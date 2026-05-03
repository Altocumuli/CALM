"""
CCD (Coherent Contextual Decoding) 在 LLaDA2.1-mini 上的实现。

核心思想（论文 "Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models"）：
  通过维护跨步预测的历史 buffer，识别跨连续步骤一致预测同一 token 的位置，
  直接接受这些"稳定"位置的 token，从而跳过多余的迭代步骤，实现自适应加速。

实现策略（与 CCD 开源代码一致）：
  - 维护最近 d 步 top-V 置信度位置及其 argmax token 的历史记录
  - 双步一致性检验：当前步 top-V ∩ 上步 top-V，且 argmax token 相同的位置直接接受
  - seek_mode=True（CCD-DS）时自适应接受所有一致位置；False 时退回均匀解码

与 LLaDA2.1 的适配：
  - 保持 block-wise 迭代结构（每个 block 内循环）
  - 全局 block attention mask（0/1 格式，与原生 generate 一致）一次性构建
  - 历史 buffer 在每个 block 内独立维护（block causal mask 导致跨 block 无上下文共享）

参考：
  - CCD 论文：Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models
  - 原始实现：https://github.com/tonyckc/CCD-DLM-code（model.py, _sample_inference_lookahead）
  - LLaDA2.1 generate 实现：modeling_llada2_moe.py
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
from collections import deque


def _llada_forward_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    forward_counter: Optional[List[int]] = None,
) -> torch.Tensor:
    """LLaDA2.1-mini 前向计算 logits。
    与 LLaDA2.1 原生 forward() 保持一致：
    - attention_mask 传入 0/1 格式，由内部 _prepare_4d_causal_attention_mask_for_sdpa 转换
    - lm_head 输出后转 float32
    forward_counter: 若传入单元素 list，则每次成功前向 +1（用于统计 TPF）。
    """
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    out = base.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    if forward_counter is not None:
        forward_counter[0] += 1
    return base.lm_head(out.last_hidden_state).float()


@dataclass
class CcdConfig:
    """CCD 解码配置参数"""

    seek_mode: bool = True  # True=动态接受（CCD-DS），False=均匀步数模式
    history_depth: int = 2  # 历史 buffer 深度（LLaDA 系列用 2，Dream 系列用 3）
    top_v: int = 4  # 每步保留 top-V 置信度位置进入 buffer

    # 继承 LLaDA2.1 原有参数
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    editing_threshold: float = 0.5
    temperature: float = 0.0
    max_post_steps: int = 16
    eos_early_stop: bool = True
    eos_id: int = 156892
    mask_id: int = 156895


def _sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """采样 token 并返回 (token_ids, probabilities)。temperature=0 为贪婪。"""
    probs = F.softmax(logits, dim=-1)
    if temperature > 0:
        scaled = F.softmax(logits / temperature, dim=-1)
        tokens = torch.multinomial(
            scaled.view(-1, scaled.size(-1)), num_samples=1
        ).view(scaled.shape[:-1])
        token_probs = torch.gather(scaled, -1, tokens.unsqueeze(-1)).squeeze(-1)
    else:
        token_probs, tokens = probs.max(dim=-1)
    return tokens, token_probs


def _neg_entropy_confidence(logits: torch.Tensor) -> torch.Tensor:
    """用负熵作为置信度（与 CCD 原论文一致）。越大越确定。"""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return torch.sum(probs * log_probs, dim=-1)


class CcdHistoryBuffer:
    """
    CCD 历史 buffer。

    维护最近 d 步 top-V 置信度位置及其 argmax token，
    用于双步一致性检验：当前步与上一步 top-V 交集中 argmax 相同的位置直接接受。

    注意：buffer 在每个 block 内独立维护，跨 block 不共享。
    """

    def __init__(self, depth: int, top_v: int, device: torch.device):
        self.depth = depth
        self.top_v = top_v
        self.device = device
        # 每个元素：(positions [k], argmax_tokens [k])
        self._history: deque = deque(maxlen=depth)

    def reset(self):
        self._history.clear()

    def update(
        self, neg_ent: torch.Tensor, tokens: torch.Tensor, active_mask: torch.Tensor
    ):
        """
        将当前步 top-V 高置信度位置及其 argmax token 加入 buffer。

        Args:
            neg_ent:     [block_len] 负熵置信度
            tokens:      [block_len] 当前步 argmax token
            active_mask: [block_len] bool，当前哪些位置是 mask
        """
        conf_at_mask = torch.where(
            active_mask,
            neg_ent,
            torch.tensor(float("-inf"), device=self.device),
        )
        k = min(self.top_v, int(active_mask.sum().item()))
        if k == 0:
            return
        _, top_positions = torch.topk(conf_at_mask, k)
        top_tokens = tokens[top_positions]  # [k]
        self._history.append((top_positions, top_tokens))

    def get_consistent_positions(
        self,
        cur_neg_ent: torch.Tensor,
        cur_tokens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        双步一致性检验：找出当前步与上一步 top-V 的交集中 argmax 相同的位置。

        Args:
            cur_neg_ent: [block_len] 当前步负熵置信度
            cur_tokens:  [block_len] 当前步 argmax token
            active_mask: [block_len] bool，当前哪些位置是 mask

        Returns:
            (positions [m], tokens [m]) 若无一致位置则返回 (None, None)
        """
        if len(self._history) == 0:
            return None, None

        # 当前步 top-V 位置
        conf_at_mask = torch.where(
            active_mask,
            cur_neg_ent,
            torch.tensor(float("-inf"), device=self.device),
        )
        k = min(self.top_v, int(active_mask.sum().item()))
        if k == 0:
            return None, None
        _, cur_top_positions = torch.topk(conf_at_mask, k)
        cur_top_set = set(cur_top_positions.tolist())

        # 上一步 top-V 位置和 token
        prev_positions, prev_tokens = self._history[-1]
        prev_pos_list = prev_positions.tolist()
        prev_token_list = prev_tokens.tolist()

        # 交集中 argmax 一致的位置
        consistent_pos = []
        consistent_tok = []
        for i, pos in enumerate(prev_pos_list):
            if pos in cur_top_set and active_mask[pos]:
                cur_tok = cur_tokens[pos].item()
                prev_tok = prev_token_list[i]
                if cur_tok == prev_tok:
                    consistent_pos.append(pos)
                    consistent_tok.append(cur_tok)

        if not consistent_pos:
            return None, None

        pos_t = torch.tensor(consistent_pos, dtype=torch.long, device=self.device)
        tok_t = torch.tensor(consistent_tok, dtype=torch.long, device=self.device)
        return pos_t, tok_t


def generate_with_ccd(
    model,
    tokenizer,
    prompt: str,
    config: Optional[CcdConfig] = None,
    thinking: bool = False,
) -> Tuple[str, int]:
    """
    使用 CCD 策略对 LLaDA2.1-mini 进行解码。

    结构与 LLaDA2.1 原生 generate 保持一致：
    - 全局 block attention mask（0/1 格式）一次性构建
    - 按 block 迭代，每个 block 内循环解码
    - 用双步一致性检验替代单纯的置信度阈值接受

    thinking=True 时开启模型内置思维链。

    Returns:
        (生成文本, forward_count)：主干前向次数，用于 TPF。
    """
    if config is None:
        config = CcdConfig()

    forward_counter: List[int] = [0]

    print(
        f"[CCD] Starting generation with seek_mode={config.seek_mode}, "
        f"history_depth={config.history_depth}, top_v={config.top_v}, thinking={thinking}"
    )

    # ── 1. 构造输入 ──────────────────────────────────────────────────────────
    chat_inp = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        thinking=thinking,
    )
    if isinstance(chat_inp, torch.Tensor):
        input_ids = chat_inp
    else:
        input_ids = chat_inp["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.long().to(model.device)

    # ── 2. 初始化序列（与原生 generate 完全一致）────────────────────────────
    prompt_length = input_ids.shape[1]
    num_blocks = (
        prompt_length + config.gen_length + config.block_length - 1
    ) // config.block_length
    total_length = num_blocks * config.block_length

    # 全局 block attention mask：0/1 格式，与原生 generate 一致
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    global_attn_mask = (
        block_mask.repeat_interleave(config.block_length, dim=0)
        .repeat_interleave(config.block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(torch.bfloat16)

    global_position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full(
        (1, total_length), config.mask_id, dtype=torch.long, device=model.device
    )
    x[:, :prompt_length] = input_ids.clone()

    # ── 3. 按 block 迭代解码 ─────────────────────────────────────────────────
    prefill_blocks = prompt_length // config.block_length

    for block_idx in range(prefill_blocks, num_blocks):
        block_start = block_idx * config.block_length
        block_end = min((block_idx + 1) * config.block_length, total_length)

        print(f"[CCD] Processing block {block_idx} (pos {block_start}:{block_end})")

        if (x[:, block_start:block_end] == config.mask_id).sum() == 0:
            print(f"[CCD] Block {block_idx} already complete, skipping")
            continue

        x = _ccd_decode_block(
            model,
            x,
            block_start,
            block_end,
            block_end,
            prompt_length,
            global_attn_mask,
            global_position_ids,
            config,
            forward_counter,
        )

        if config.eos_early_stop and config.eos_id in x[:, prompt_length:]:
            print("[CCD] EOS detected, stopping early")
            break

    # ── 4. 后处理 ────────────────────────────────────────────────────────────
    generated_part = x[:, prompt_length : prompt_length + config.gen_length]
    eos_pos = (generated_part == config.eos_id).nonzero(as_tuple=True)
    if len(eos_pos) >= 2 and len(eos_pos[1]) > 0:
        generated_tokens = generated_part[:, : eos_pos[1][0].item() + 1]
    else:
        generated_tokens = generated_part

    result_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"[CCD] Generation completed, total tokens: {generated_tokens.shape[1]}")
    return result_text.strip(), forward_counter[0]


def _ccd_decode_block(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    prompt_length: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: CcdConfig,
    forward_counter: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    在单个 block 上执行迭代式 CCD 解码。
    每个 block 独立维护历史 buffer，跨 block 不共享。
    """
    device = x.device
    block_len = block_end - block_start
    post_steps = 0

    history_buffer = CcdHistoryBuffer(
        depth=config.history_depth,
        top_v=config.top_v,
        device=device,
    )

    # 上限：block_len（最坏每次填1个）+ max_post_steps（编辑阶段）+ 缓冲
    max_iterations = block_len + config.max_post_steps + 10
    for _ in range(max_iterations):
        cur_x = x[:, :current_window_end].clone()
        block_slice = cur_x[:, block_start:block_end]
        active_mask = (block_slice == config.mask_id)[0]  # [block_len]

        if not active_mask.any():
            post_steps += 1
            if post_steps > config.max_post_steps:
                break

        old_block = block_slice.clone()

        # ── 前向计算 ──────────────────────────────────────────────────────
        cur_attn_mask = global_attn_mask[:, :, :current_window_end, :current_window_end]
        cur_pos_ids = global_position_ids[:, :current_window_end]

        with torch.no_grad():
            logits = _llada_forward_logits(
                model, cur_x, cur_attn_mask, cur_pos_ids, forward_counter
            )

        block_logits_2d = logits[0, block_start:block_end, :]  # [block_len, vocab]

        # ── 计算置信度和预测 token ────────────────────────────────────────
        neg_ent = _neg_entropy_confidence(block_logits_2d)  # [block_len]
        tokens, token_probs = _sample_tokens(block_logits_2d, config.temperature)

        # ── CCD 核心：双步一致性检验 → 接受一致 token ─────────────────────
        accepted_positions, accepted_tokens_tensor = _ccd_select_tokens(
            active_mask=active_mask,
            cur_tokens=tokens,
            cur_probs=token_probs,
            cur_neg_ent=neg_ent,
            history_buffer=history_buffer,
            config=config,
            device=device,
        )

        if len(accepted_positions) > 0:
            x[0, block_start:block_end][accepted_positions] = accepted_tokens_tensor

        # ── 编辑已生成 token（与原生 generate 一致）──────────────────────
        non_mask_non_prompt = (x[0, block_start:block_end] != config.mask_id) & (
            torch.arange(block_len, device=device)
            >= (prompt_length - block_start if block_start < prompt_length else 0)
        )
        if non_mask_non_prompt.any():
            edit_tokens, edit_probs = _sample_tokens(
                block_logits_2d, config.temperature
            )
            edit_mask = (
                non_mask_non_prompt
                & (edit_probs > config.editing_threshold)
                & (edit_tokens != x[0, block_start:block_end])
            )
            if edit_mask.any():
                x[0, block_start:block_end][edit_mask] = edit_tokens[edit_mask]

        # ── 更新历史 buffer（使用更新后的 active_mask，排除已接受位置）────
        updated_active_mask = x[0, block_start:block_end] == config.mask_id
        history_buffer.update(neg_ent, tokens, updated_active_mask)

        # ── 检查是否有实际变化 ────────────────────────────────────────────
        if torch.equal(old_block, x[:, block_start:block_end]):
            break

    return x


def _ccd_select_tokens(
    active_mask: torch.Tensor,
    cur_tokens: torch.Tensor,
    cur_probs: torch.Tensor,
    cur_neg_ent: torch.Tensor,
    history_buffer: CcdHistoryBuffer,
    config: CcdConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CCD token 选择逻辑：

    1. seek_mode=True 且 buffer 有历史时：
       - 双步一致性检验：当前步 top-V ∩ 上步 top-V，argmax token 相同 → 直接接受（自适应 budget）
    2. 退回置信度阈值驱动的单步填充（与原生 generate 一致）

    Returns:
        (accepted_positions [k], accepted_tokens [k])
    """
    empty = (
        torch.tensor([], dtype=torch.long, device=device),
        torch.tensor([], dtype=torch.long, device=device),
    )
    if not active_mask.any():
        return empty

    # ── 尝试双步一致性快速接受 ────────────────────────────────────────────
    if config.seek_mode and len(history_buffer._history) >= 1:
        pos, tok = history_buffer.get_consistent_positions(
            cur_neg_ent, cur_tokens, active_mask
        )
        if pos is not None and len(pos) > 0:
            print(f"    [CCD] Consistency-accept: {len(pos)} tokens")
            return pos, tok

    # ── 退回到置信度阈值驱动的单步填充 ──────────────────────────────────
    conf_at_mask = torch.where(
        active_mask, cur_probs, torch.tensor(float("-inf"), device=device)
    )
    high_conf = (conf_at_mask > config.threshold) & active_mask

    if high_conf.any():
        positions = high_conf.nonzero(as_tuple=True)[0]
        return positions, cur_tokens[positions]
    else:
        best_pos = conf_at_mask.argmax().unsqueeze(0)
        return best_pos, cur_tokens[best_pos]
