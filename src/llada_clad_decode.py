"""
CLAD (Consistency-guided Lookahead Adaptive Decoding)
针对 LLaDA2.1-mini 的原创解码策略优化，毕设核心创新。

════════════════════════════════════════════════════════════
核心思想
════════════════════════════════════════════════════════════

CLAD 融合了 CCD 的跨步一致性思想和 LoPA 的前瞻分支思想，
针对 LLaDA2.1 block-wise 生成结构提出一种两阶段自适应解码机制：

  **阶段一（Consistency Fast-track）**：
    与 CCD 相同，维护上一步 top-V 位置的 argmax token。
    当前步 top-V 与上步 top-V 有交集且 argmax 一致 → 直接加速接受。
    无需置信度阈值，依靠跨步稳定性保证质量。

  **阶段二（Consistency-guided Lookahead）**：
    当一致性快速路径无候选时（早期步骤 / 模型仍不确定），
    不直接退回到 LLaDA2.1 的普通阈值填充，而是先做 k 个前瞻：
    对 top-k 不确定位置各生成一个候选分支，评估每个分支的质量。

    ★ 关键创新 —— 一致性传播分支评分（Consistency Propagation Score）：
      填充位置 p 后做一次 forward，观察其他剩余 mask 位置的新预测，
      与当前步已有预测比较：预测没变的位置越多，说明填入 p 的 token
      越"自然"，其他位置在下一步更容易通过一致性检验被加速接受。

      score(B_j) = α × consistency_propagation(B_j)
                 + (1-α) × future_confidence(B_j)

    选择 score 最高的分支作为本步输出。

  **阶段三（Fallback）**：
    若前瞻分支数为 0 或不在预热期后，退回标准阈值填充。

════════════════════════════════════════════════════════════
与 CCD / LoPA 的区别
════════════════════════════════════════════════════════════

  CCD：被动等待一致性出现，出现后直接接受；不确定时仅做阈值填充
  LoPA：主动生成分支，用"剩余 mask 的未来置信度"评估分支
  CLAD：主动生成分支，用"填入后有多少位置能更快达成一致性"评估分支
        → 将一致性由"被动检测"提升为"主动引导"

════════════════════════════════════════════════════════════
参考
════════════════════════════════════════════════════════════
  - CCD 论文：Beyond Confidence: Adaptive and Coherent Decoding for DLMs
  - LoPA 论文：Scaling dLLM Inference via Lookahead Parallel Decoding
  - LLaDA2.1 generate 实现：modeling_llada2_moe.py
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass
from collections import deque


def _user_prompt_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    """单条 user 消息 → input_ids [1, L]，与 baseline apply_chat_template 一致。"""
    chat_inp = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(chat_inp, torch.Tensor):
        input_ids = chat_inp
    else:
        input_ids = chat_inp["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.long()


# ─────────────────────────────────────────────────────────────
# 共用前向函数
# ─────────────────────────────────────────────────────────────


def _llada_forward_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    forward_counter: Optional[List[int]] = None,
) -> torch.Tensor:
    """LLaDA2.1-mini 前向计算 logits。
    - attention_mask：0/1 格式，与原生 generate 一致
    - lm_head 输出转 float32，与原生 forward() 一致
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


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────


@dataclass
class DecodeStats:
    """generate_with_clad 调用期间各解码阶段的命中统计。"""

    phase1_iters: int = 0  # 阶段一（一致性快通道）命中迭代次数
    phase1_tokens: int = 0  # 阶段一累计接受 token 数
    phase2_iters: int = 0  # 阶段二触发次数（warmup 后、阶段一未命中时尝试）
    phase2_accepted: int = 0  # 阶段二成功接受次数
    phase3_iters: int = 0  # 阶段三 fallback 次数
    total_iters: int = 0  # 总迭代次数

    def hit_rates(self) -> dict:
        n = max(self.total_iters, 1)
        return {
            "phase1_hit_rate": self.phase1_iters / n,
            "phase2_trigger_rate": self.phase2_iters / n,
            "phase2_accepted_rate": self.phase2_accepted / n,
            "phase3_fallback_rate": self.phase3_iters / n,
        }


def _append_trace(trace_out: Optional[List[dict]], event: dict):
    """将结构化事件写入 trace_out（若启用）。"""
    if trace_out is not None:
        trace_out.append(event)


@dataclass
class CladConfig:
    """CLAD 解码配置参数"""

    # ── 阶段一：一致性快速通道（CCD-inspired）────────────────
    top_v: int = 4  # 每步保留 top-V 置信度位置进入 buffer
    # history_depth 固定为 2（只需比较当前步与上一步）

    # ── 阶段二：一致性传播前瞻（CLAD 创新）──────────────────
    num_lookahead: int = 2  # 前瞻分支数 k
    consistency_weight: float = 0.6  # α，一致性传播分数的权重
    lookahead_warmup: int = 3  # 前几步先积累 buffer 再启用前瞻

    # ── 继承 LLaDA2.1 参数 ────────────────────────────────────
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    editing_threshold: float = 0.5
    temperature: float = 0.0
    max_post_steps: int = 16
    eos_early_stop: bool = True
    eos_id: int = 156892
    mask_id: int = 156895


# ─────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────


def _sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (token_ids [L], token_probs [L])，temperature=0 为贪婪。"""
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
    """负熵置信度 [L]，越大越确定。"""
    probs = F.softmax(logits, dim=-1)
    return torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


# ─────────────────────────────────────────────────────────────
# 历史 Buffer（只存 argmax token，与 CCD 一致）
# ─────────────────────────────────────────────────────────────


class CladHistoryBuffer:
    """
    维护最近 1 步（depth=2 但只用最后一步）top-V 位置的 argmax token，
    供一致性快速通道使用。
    """

    def __init__(self, top_v: int, device: torch.device):
        self.top_v = top_v
        self.device = device
        # 每个元素：(positions [k], argmax_tokens [k])
        self._history: deque = deque(maxlen=2)

    def reset(self):
        self._history.clear()

    @property
    def has_history(self) -> bool:
        return len(self._history) > 0

    def update(
        self, neg_ent: torch.Tensor, tokens: torch.Tensor, active_mask: torch.Tensor
    ):
        """加入当前步 top-V 位置和 argmax token。"""
        conf_at_mask = torch.where(
            active_mask,
            neg_ent,
            torch.tensor(float("-inf"), device=self.device),
        )
        k = min(self.top_v, int(active_mask.sum().item()))
        if k == 0:
            return
        _, top_positions = torch.topk(conf_at_mask, k)
        self._history.append((top_positions, tokens[top_positions]))

    def get_consistent_positions(
        self,
        cur_neg_ent: torch.Tensor,
        cur_tokens: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        双步一致性检验：当前步 top-V ∩ 上步 top-V，且 argmax token 相同。
        返回 (positions, tokens) 或 (None, None)。
        """
        if not self.has_history:
            return None, None

        conf_at_mask = torch.where(
            active_mask,
            cur_neg_ent,
            torch.tensor(float("-inf"), device=self.device),
        )
        k = min(self.top_v, int(active_mask.sum().item()))
        if k == 0:
            return None, None
        _, cur_top = torch.topk(conf_at_mask, k)
        cur_top_set = set(cur_top.tolist())

        prev_pos, prev_tok = self._history[-1]
        consistent_pos, consistent_tok = [], []
        for pos, ptok in zip(prev_pos.tolist(), prev_tok.tolist()):
            if pos in cur_top_set and active_mask[pos]:
                if cur_tokens[pos].item() == ptok:
                    consistent_pos.append(pos)
                    consistent_tok.append(ptok)

        if not consistent_pos:
            return None, None
        return (
            torch.tensor(consistent_pos, dtype=torch.long, device=self.device),
            torch.tensor(consistent_tok, dtype=torch.long, device=self.device),
        )

    def get_prev_tokens(self) -> Optional[torch.Tensor]:
        """返回上一步所有 top-V 位置的 argmax token（用于前瞻评分）。"""
        if not self.has_history:
            return None
        return self._history[-1]  # (positions, tokens)


# ─────────────────────────────────────────────────────────────
# 一致性传播分支评分（核心创新）
# ─────────────────────────────────────────────────────────────


def _clad_branch_score(
    branch_x: torch.Tensor,  # [1, window_end] 分支序列
    branch_logits: torch.Tensor,  # [1, window_end, vocab] 分支 forward 结果
    cur_tokens: torch.Tensor,  # [block_len] 当前步（填前）的 argmax token
    block_start: int,
    block_end: int,
    config: CladConfig,
) -> float:
    """
    计算分支的综合得分：

      score = α × consistency_propagation + (1-α) × future_confidence

    consistency_propagation：在该分支中，其他剩余 mask 位置的新预测与
      当前步（填前）预测一致的比例。一致率越高，说明填入的 token 越"自然"，
      其他位置在下一步通过一致性检验被加速接受的概率越大。

    future_confidence：其他剩余 mask 位置的平均最大概率（LoPA-style）。
    """
    block_len = block_end - block_start
    branch_block = branch_x[0, block_start:block_end]
    remaining_mask = branch_block == config.mask_id

    if remaining_mask.sum() == 0:
        return 1.0  # 无剩余 mask，分支已完整，最优

    branch_block_logits = branch_logits[0, block_start:block_end]  # [block_len, vocab]
    probs = F.softmax(branch_block_logits, dim=-1)

    # future_confidence：剩余 mask 位置的平均最大概率
    max_probs = probs[remaining_mask].max(dim=-1).values
    future_conf = max_probs.mean().item()

    # consistency_propagation：剩余 mask 位置的分支新预测与当前步预测的一致率
    branch_new_tokens = probs.argmax(dim=-1)  # [block_len]
    matches = branch_new_tokens[remaining_mask] == cur_tokens[remaining_mask]
    consistency_prop = matches.float().mean().item()

    alpha = config.consistency_weight
    return alpha * consistency_prop + (1.0 - alpha) * future_conf


def _clad_branch_score_details(
    branch_x: torch.Tensor,
    branch_logits: torch.Tensor,
    cur_tokens: torch.Tensor,
    block_start: int,
    block_end: int,
    config: CladConfig,
) -> dict:
    """返回 v1 分支评分细项，便于个案分析解释。"""
    branch_block = branch_x[0, block_start:block_end]
    remaining_mask = branch_block == config.mask_id

    if remaining_mask.sum() == 0:
        return {
            "score": 1.0,
            "consistency_propagation": 1.0,
            "future_confidence": 1.0,
            "remaining_mask_count": 0,
        }

    branch_block_logits = branch_logits[0, block_start:block_end]
    probs = F.softmax(branch_block_logits, dim=-1)
    max_probs = probs[remaining_mask].max(dim=-1).values
    future_conf = max_probs.mean().item()

    branch_new_tokens = probs.argmax(dim=-1)
    matches = branch_new_tokens[remaining_mask] == cur_tokens[remaining_mask]
    consistency_prop = matches.float().mean().item()

    alpha = config.consistency_weight
    score = alpha * consistency_prop + (1.0 - alpha) * future_conf
    return {
        "score": float(score),
        "consistency_propagation": float(consistency_prop),
        "future_confidence": float(future_conf),
        "remaining_mask_count": int(remaining_mask.sum().item()),
    }


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────


def generate_with_clad(
    model,
    tokenizer,
    prompt: str,
    config: Optional[CladConfig] = None,
    stats_out: Optional[List] = None,
    trace_out: Optional[List[dict]] = None,
) -> Tuple[str, int]:
    """
    使用 CLAD 策略对 LLaDA2.1-mini 进行解码。

    stats_out: 若传入非 None 的列表，本次调用结束后会 append 一个 DecodeStats 实例。

    Returns:
        (生成文本, forward_count)：主干前向次数，用于 TPF。
    """
    if config is None:
        config = CladConfig()

    _stats = DecodeStats() if stats_out is not None else None
    forward_counter: List[int] = [0]

    print(
        f"[CLAD] Starting: top_v={config.top_v}, num_lookahead={config.num_lookahead}, "
        f"consistency_weight={config.consistency_weight}"
    )

    # ── 1. 构造输入（与 tokenizer.apply_chat_template 一致）
    input_ids = _user_prompt_input_ids(tokenizer, prompt).to(model.device)

    # ── 2. 初始化序列 ────────────────────────────────────────────────────────
    prompt_length = input_ids.shape[1]
    num_blocks = (
        prompt_length + config.gen_length + config.block_length - 1
    ) // config.block_length
    total_length = num_blocks * config.block_length

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

        print(f"[CLAD] Processing block {block_idx} (pos {block_start}:{block_end})")

        if (x[:, block_start:block_end] == config.mask_id).sum() == 0:
            continue

        x = _clad_decode_block(
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
            stats=_stats,
            trace_out=trace_out,
        )

        if config.eos_early_stop and config.eos_id in x[:, prompt_length:]:
            print("[CLAD] EOS detected, stopping early")
            break

    # ── 4. 后处理 ────────────────────────────────────────────────────────────
    generated_part = x[:, prompt_length : prompt_length + config.gen_length]
    eos_pos = (generated_part == config.eos_id).nonzero(as_tuple=True)
    if len(eos_pos) >= 2 and len(eos_pos[1]) > 0:
        generated_tokens = generated_part[:, : eos_pos[1][0].item() + 1]
    else:
        generated_tokens = generated_part

    result_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"[CLAD] Generation completed, total tokens: {generated_tokens.shape[1]}")
    if stats_out is not None and _stats is not None:
        stats_out.append(_stats)
    return result_text.strip(), forward_counter[0]


# ─────────────────────────────────────────────────────────────
# Block 级解码
# ─────────────────────────────────────────────────────────────


def _clad_decode_block(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    prompt_length: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: CladConfig,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
    trace_out: Optional[List[dict]] = None,
) -> torch.Tensor:
    """
    在单个 block 上执行 CLAD 解码。
    每个 block 独立维护历史 buffer，跨 block 不共享。
    """
    device = x.device
    block_len = block_end - block_start
    post_steps = 0
    iter_count = 0  # 用于 lookahead 预热控制

    history_buffer = CladHistoryBuffer(top_v=config.top_v, device=device)

    max_iterations = block_len + config.max_post_steps + 10
    for _ in range(max_iterations):
        cur_x = x[:, :current_window_end].clone()
        active_mask = cur_x[0, block_start:block_end] == config.mask_id  # [block_len]

        if not active_mask.any():
            post_steps += 1
            if post_steps > config.max_post_steps:
                break

        old_block = x[0, block_start:block_end].clone()

        # ── 前向计算 ──────────────────────────────────────────────────────
        cur_attn_mask = global_attn_mask[:, :, :current_window_end, :current_window_end]
        cur_pos_ids = global_position_ids[:, :current_window_end]

        with torch.no_grad():
            logits = _llada_forward_logits(
                model, cur_x, cur_attn_mask, cur_pos_ids, forward_counter
            )

        block_logits_2d = logits[0, block_start:block_end]  # [block_len, vocab]

        neg_ent = _neg_entropy_confidence(block_logits_2d)
        tokens, token_probs = _sample_tokens(block_logits_2d, config.temperature)

        # ── 阶段一：一致性快速通道 ────────────────────────────────────────
        accepted = False
        if history_buffer.has_history:
            pos, tok = history_buffer.get_consistent_positions(
                neg_ent, tokens, active_mask
            )
            if pos is not None and len(pos) > 0:
                x[0, block_start:block_end][pos] = tok
                print(f"    [CLAD] Phase-1 consistency: accepted {len(pos)} tokens")
                accepted = True
                if stats is not None:
                    stats.phase1_iters += 1
                    stats.phase1_tokens += len(pos)
                _append_trace(
                    trace_out,
                    {
                        "block": int(block_start // config.block_length),
                        "iter": int(iter_count),
                        "phase": "phase1",
                        "accepted_positions": [int(p) for p in pos.tolist()],
                        "accepted_tokens": [int(t) for t in tok.tolist()],
                        "n_tokens": int(len(pos)),
                    },
                )

        # ── 阶段二：一致性传播前瞻（预热后且阶段一未命中时启用）──────────
        if (
            not accepted
            and iter_count >= config.lookahead_warmup
            and config.num_lookahead > 0
        ):
            if stats is not None:
                stats.phase2_iters += 1
            accepted = _clad_lookahead_fill(
                model,
                x,
                logits,
                tokens,
                token_probs,
                active_mask,
                block_start,
                block_end,
                current_window_end,
                global_attn_mask,
                global_position_ids,
                config,
                device,
                forward_counter,
                stats=stats,
                trace_out=trace_out,
                block_iter=iter_count,
            )

        # ── 阶段三：退回阈值填充 ──────────────────────────────────────────
        if not accepted:
            if stats is not None:
                stats.phase3_iters += 1
            conf_at_mask = torch.where(
                active_mask, token_probs, torch.tensor(float("-inf"), device=device)
            )
            high_conf = (conf_at_mask > config.threshold) & active_mask
            if high_conf.any():
                x[0, block_start:block_end][high_conf] = tokens[high_conf]
                accepted_positions = torch.nonzero(high_conf, as_tuple=False).view(-1)
                _append_trace(
                    trace_out,
                    {
                        "block": int(block_start // config.block_length),
                        "iter": int(iter_count),
                        "phase": "phase3",
                        "fallback_mode": "threshold_multi",
                        "accepted_positions": [
                            int(p) for p in accepted_positions.tolist()
                        ],
                        "accepted_tokens": [
                            int(tokens[p].item()) for p in accepted_positions
                        ],
                        "accepted_probs": [
                            float(token_probs[p].item()) for p in accepted_positions
                        ],
                        "n_tokens": int(accepted_positions.numel()),
                    },
                )
            else:
                best_pos = conf_at_mask.argmax()
                x[0, block_start:block_end][best_pos] = tokens[best_pos]
                _append_trace(
                    trace_out,
                    {
                        "block": int(block_start // config.block_length),
                        "iter": int(iter_count),
                        "phase": "phase3",
                        "fallback_mode": "argmax_single",
                        "accepted_positions": [int(best_pos.item())],
                        "accepted_tokens": [int(tokens[best_pos].item())],
                        "accepted_probs": [float(token_probs[best_pos].item())],
                        "n_tokens": 1,
                    },
                )

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

        # ── 更新历史 buffer ───────────────────────────────────────────────
        updated_active = x[0, block_start:block_end] == config.mask_id
        history_buffer.update(neg_ent, tokens, updated_active)
        iter_count += 1
        if stats is not None:
            stats.total_iters += 1

        if torch.equal(old_block, x[0, block_start:block_end]):
            break

    return x


def _clad_lookahead_fill(
    model,
    x: torch.Tensor,
    base_logits: torch.Tensor,
    cur_tokens: torch.Tensor,
    cur_probs: torch.Tensor,
    active_mask: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: CladConfig,
    device: torch.device,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
    trace_out: Optional[List[dict]] = None,
    block_iter: Optional[int] = None,
) -> bool:
    """
    CLAD 阶段二：一致性传播前瞻。

    对 top-k 不确定位置各生成一个候选分支，
    用一致性传播分数 + 未来置信度综合评分，选择最优分支填入。

    Returns:
        True 表示成功选出分支并填入 token；False 表示无有效分支（退回阶段三）
    """
    # 从 mask 位置中选 top-k 候选（按置信度排序）
    conf_at_mask = torch.where(
        active_mask, cur_probs, torch.tensor(float("-inf"), device=device)
    )
    num_candidates = min(config.num_lookahead, int(active_mask.sum().item()))
    if num_candidates == 0:
        return False

    _, candidate_indices = torch.topk(conf_at_mask, num_candidates)

    cur_attn_mask = global_attn_mask[:, :, :current_window_end, :current_window_end]
    cur_pos_ids = global_position_ids[:, :current_window_end]

    best_score = -1.0
    best_pos = None
    best_tok = None
    candidate_traces = []

    for cand_pos in candidate_indices.tolist():
        # 构造前瞻分支：填入 cand_pos 位置的 token
        branch_x = x[:, :current_window_end].clone()
        branch_x[0, block_start + cand_pos] = cur_tokens[cand_pos].item()

        with torch.no_grad():
            branch_logits = _llada_forward_logits(
                model, branch_x, cur_attn_mask, cur_pos_ids, forward_counter
            )

        details = _clad_branch_score_details(
            branch_x,
            branch_logits,
            cur_tokens,
            block_start,
            block_end,
            config,
        )
        score = details["score"]
        candidate_traces.append(
            {
                "pos": int(cand_pos),
                "token": int(cur_tokens[cand_pos].item()),
                **details,
            }
        )

        if score > best_score:
            best_score = score
            best_pos = cand_pos
            best_tok = cur_tokens[cand_pos].item()

    if best_pos is None:
        return False

    x[0, block_start + best_pos] = best_tok
    print(
        f"    [CLAD] Phase-2 lookahead: filled pos={best_pos} "
        f"tok={best_tok} score={best_score:.3f}"
    )
    if stats is not None:
        stats.phase2_accepted += 1
    _append_trace(
        trace_out,
        {
            "block": int(block_start // config.block_length),
            "iter": int(block_iter) if block_iter is not None else None,
            "phase": "phase2",
            "accepted_positions": [int(best_pos)],
            "accepted_tokens": [int(best_tok)],
            "n_tokens": 1,
            "winner_score": float(best_score),
            "candidate_scores": candidate_traces,
        },
    )
    return True
