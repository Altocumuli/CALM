"""
LocalLeap baseline for LLaDA2.1-mini.

This is a direct local-determinism baseline adapted to the block-wise LLaDA2.1
decoding loop used in this project. Unlike CALM, anchors are selected only by
current-step confidence; no cross-step consistency is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from llada_clad_decode import (
    _append_trace,
    _llada_forward_logits,
    _sample_tokens,
    _user_prompt_input_ids,
)


@dataclass
class DecodeStats:
    """LocalLeap decoding statistics."""

    local_leap_iters: int = 0
    local_leap_tokens: int = 0
    phase3_iters: int = 0
    total_iters: int = 0

    def hit_rates(self) -> dict:
        n = max(self.total_iters, 1)
        local_tokens_per_iter = self.local_leap_tokens / n
        return {
            "local_leap_hit_rate": self.local_leap_iters / n,
            # Backward-compatible name; semantically this is relaxed-neighbor
            # extra tokens per iteration, excluding anchors and forced top-1.
            "local_leap_token_rate": local_tokens_per_iter,
            "local_leap_tokens_per_iter": local_tokens_per_iter,
            "local_leap_token_count": self.local_leap_tokens,
            "decode_total_iters": self.total_iters,
            "phase3_fallback_rate": self.phase3_iters / n,
        }


@dataclass
class LocalLeapConfig:
    """LocalLeap decoding configuration."""

    anchor_threshold: float = 0.90
    relaxed_threshold: float = 0.75
    local_radius: int = 4

    # inherited LLaDA2.1 params
    gen_length: int = 2048
    block_length: int = 32
    threshold: float = 0.7
    editing_threshold: float = 0.5
    temperature: float = 0.0
    max_post_steps: int = 16
    eos_early_stop: bool = True
    eos_id: int = 156892
    mask_id: int = 156895


def _apply_localleap_accept(
    x: torch.Tensor,
    token_probs: torch.Tensor,
    tokens: torch.Tensor,
    active_mask: torch.Tensor,
    block_start: int,
    config: LocalLeapConfig,
) -> dict:
    """Apply LocalLeap transfer rule within one active block.

    The rule mirrors the reference implementation:
    1. Always accept the current top-1 masked position to guarantee progress.
    2. Positions above anchor_threshold are high-confidence anchors.
    3. If at least one anchor exists, positions in anchor neighborhoods use
       relaxed_threshold; all other masked positions use anchor_threshold.
    """

    device = token_probs.device
    conf = torch.where(
        active_mask,
        token_probs,
        torch.tensor(float("-inf"), device=device),
    )
    if not active_mask.any():
        return {
            "accepted_positions": [],
            "accepted_tokens": [],
            "accepted_probs": [],
            "anchor_positions": [],
            "relaxed_positions": [],
            "local_positions": [],
            "forced_top1_position": None,
            "fired": False,
        }

    best_pos = int(conf.argmax().item())
    anchor_mask = (conf >= config.anchor_threshold) & active_mask
    anchor_positions = torch.nonzero(anchor_mask, as_tuple=False).view(-1).tolist()

    neighbor_positions = set()
    if anchor_positions:
        block_len = active_mask.shape[0]
        for pos in anchor_positions:
            left = max(0, int(pos) - config.local_radius)
            right = min(block_len, int(pos) + config.local_radius + 1)
            for neighbor_pos in range(left, right):
                if active_mask[neighbor_pos].item():
                    neighbor_positions.add(int(neighbor_pos))

    accepted = {best_pos}
    local_positions = set()
    relaxed_positions = set()

    active_positions = torch.nonzero(active_mask, as_tuple=False).view(-1).tolist()
    for pos in active_positions:
        prob = float(token_probs[pos].item())
        if pos in neighbor_positions:
            if prob >= config.relaxed_threshold:
                accepted.add(int(pos))
                if prob < config.anchor_threshold:
                    # Count only tokens accepted because of the relaxed local
                    # rule. Anchors and the forced top-1 progress token are not
                    # neighbor-extra tokens and should not be compared to CALM's
                    # neighbor-token statistic.
                    if int(pos) != best_pos:
                        local_positions.add(int(pos))
                        relaxed_positions.add(int(pos))
        elif prob >= config.anchor_threshold:
            accepted.add(int(pos))

    accepted_positions = sorted(accepted)
    accepted_tokens = [int(tokens[pos].item()) for pos in accepted_positions]
    accepted_probs = [float(token_probs[pos].item()) for pos in accepted_positions]

    for pos, tok in zip(accepted_positions, accepted_tokens):
        x[0, block_start + pos] = tok

    return {
        "accepted_positions": accepted_positions,
        "accepted_tokens": accepted_tokens,
        "accepted_probs": accepted_probs,
        "anchor_positions": [int(pos) for pos in anchor_positions],
        "relaxed_positions": sorted(relaxed_positions),
        "local_positions": sorted(local_positions),
        "forced_top1_position": best_pos,
        "fired": len(anchor_positions) > 0,
    }


def _localleap_decode_block(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    current_window_end: int,
    prompt_length: int,
    global_attn_mask: torch.Tensor,
    global_position_ids: torch.Tensor,
    config: LocalLeapConfig,
    forward_counter: Optional[List[int]] = None,
    stats: Optional[DecodeStats] = None,
    trace_out: Optional[List[dict]] = None,
) -> torch.Tensor:
    device = x.device
    block_len = block_end - block_start
    post_steps = 0
    iter_count = 0
    max_iterations = block_len + config.max_post_steps + 10

    for _ in range(max_iterations):
        cur_x = x[:, :current_window_end].clone()
        active_mask = cur_x[0, block_start:block_end] == config.mask_id
        if not active_mask.any():
            post_steps += 1
            if post_steps > config.max_post_steps:
                break

        old_block = x[0, block_start:block_end].clone()
        cur_attn_mask = global_attn_mask[:, :, :current_window_end, :current_window_end]
        cur_pos_ids = global_position_ids[:, :current_window_end]

        with torch.no_grad():
            logits = _llada_forward_logits(
                model, cur_x, cur_attn_mask, cur_pos_ids, forward_counter
            )

        block_logits_2d = logits[0, block_start:block_end]
        tokens, token_probs = _sample_tokens(block_logits_2d, config.temperature)
        accept_info = _apply_localleap_accept(
            x,
            token_probs,
            tokens,
            active_mask,
            block_start,
            config,
        )

        if stats is not None:
            if accept_info["fired"]:
                stats.local_leap_iters += 1
                stats.local_leap_tokens += len(accept_info["local_positions"])
            else:
                stats.phase3_iters += 1

        _append_trace(
            trace_out,
            {
                "block": int(block_start // config.block_length),
                "iter": int(iter_count),
                "phase": "localleap" if accept_info["fired"] else "fallback",
                "accepted_positions": accept_info["accepted_positions"],
                "accepted_tokens": accept_info["accepted_tokens"],
                "accepted_probs": accept_info["accepted_probs"],
                "anchor_positions": accept_info["anchor_positions"],
                "local_positions": accept_info["local_positions"],
                "relaxed_positions": accept_info["relaxed_positions"],
                "forced_top1_position": accept_info["forced_top1_position"],
                "n_tokens": int(len(accept_info["accepted_positions"])),
            },
        )

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

        iter_count += 1
        if stats is not None:
            stats.total_iters += 1

        if torch.equal(old_block, x[0, block_start:block_end]):
            break

    return x


def generate_with_localleap(
    model,
    tokenizer,
    prompt: str,
    config: Optional[LocalLeapConfig] = None,
    stats_out: Optional[List] = None,
    trace_out: Optional[List[dict]] = None,
) -> Tuple[str, int]:
    """Generate with the LocalLeap baseline."""
    if config is None:
        config = LocalLeapConfig()

    _stats = DecodeStats() if stats_out is not None else None
    forward_counter: List[int] = [0]

    print(
        f"[LocalLeap] anchor={config.anchor_threshold:.2f} "
        f"relaxed={config.relaxed_threshold:.2f} radius={config.local_radius}"
    )

    try:
        _dev = next(model.parameters()).device
    except StopIteration:
        _dev = torch.device("cpu")

    input_ids = _user_prompt_input_ids(tokenizer, prompt).to(_dev)
    prompt_length = input_ids.shape[1]
    num_blocks = (
        prompt_length + config.gen_length + config.block_length - 1
    ) // config.block_length
    total_length = num_blocks * config.block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=_dev))
    global_attn_mask = (
        block_mask.repeat_interleave(config.block_length, dim=0)
        .repeat_interleave(config.block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(torch.bfloat16)
    global_position_ids = torch.arange(total_length, device=_dev).unsqueeze(0)

    x = torch.full((1, total_length), config.mask_id, dtype=torch.long, device=_dev)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // config.block_length
    for block_idx in range(prefill_blocks, num_blocks):
        block_start = block_idx * config.block_length
        block_end = min((block_idx + 1) * config.block_length, total_length)
        print(f"[LocalLeap] block {block_idx} ({block_start}:{block_end})")

        if (x[:, block_start:block_end] == config.mask_id).sum() == 0:
            continue

        x = _localleap_decode_block(
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
            print("[LocalLeap] EOS, stop")
            break

    generated_part = x[:, prompt_length : prompt_length + config.gen_length]
    eos_pos = (generated_part == config.eos_id).nonzero(as_tuple=True)
    if len(eos_pos) >= 2 and len(eos_pos[1]) > 0:
        generated_tokens = generated_part[:, : eos_pos[1][0].item() + 1]
    else:
        generated_tokens = generated_part

    result_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"[LocalLeap] done, tokens={generated_tokens.shape[1]}")

    if stats_out is not None and _stats is not None:
        stats_out.append(_stats)
    return result_text.strip(), forward_counter[0]
