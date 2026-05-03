"""
评测 benchmark 推理结果的准确率与效率。

当前评测方式：
- LLM-as-a-Judge：使用 Qwen3.5-9B 等模型对「题目 + 参考答案 + 模型回答」进行正确性判断。

效率指标说明：
- avg_gen_time_sec        平均每条样本的生成时间（秒），保留作参考
- avg_output_len          平均输出 token 数
- throughput_tok_per_sec  整体吞吐量（tokens/sec）= 总输出 token 数 / 总生成时间
- correct_throughput      正确吞吐量（correct_tokens/sec）= 正确样本的总输出 token 数 / 所有样本的总生成时间
                          综合反映"又快又准"的能力，适合跨方法横向对比
- tpf / macro_tpf         TPF（Tokens Per Forward）= 每条输出 token 数 / 解码中主干前向次数（forward_count）。
                          需在 run_benchmark_llada2 写入 forward_count 后才有；旧 jsonl 无该字段则报告中不显示。
- diffusion_steps         扩散步数统计（当前与 forward_count 同口径）；报告给出平均扩散步数与总扩散步数。
- speedup_vs_baseline     相对 baseline 的加速比（需在报告中手动对比）

用法示例：
    # 评测单个文件
    python src/evaluate_benchmark_results.py \\
        --results_file experiments/runs/xxx.jsonl \\
        --judge_model_path /path/to/judge/model

    # 评测整个目录
    python src/evaluate_benchmark_results.py \\
        --results_dir experiments/runs/ \\
        --judge_model_path /path/to/judge/model

评测结果默认保存到 dlm/experiments/evals/ 目录。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Override this with --judge_model_path or CALM_JUDGE_MODEL_PATH.
DEFAULT_JUDGE_MODEL_PATH = os.environ.get(
    "CALM_JUDGE_MODEL_PATH", "Qwen/Qwen3.5-9B"
)

# 默认评测报告目录：本脚本位于 dlm/src/，与 src 同级的 experiments/evals/
# （不根据 results 路径推断，避免 runs/子目录导致误写到 runs/evals/）
DEFAULT_EVALS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "evals"

PHASE_RATE_KEYS = [
    "phase1_hit_rate",
    "calm_neighbor_hit_rate",
    "calm_neighbor_token_rate",
    "phase2_trigger_rate",
    "phase2_accepted_rate",
    "local_leap_hit_rate",
    "local_leap_token_rate",
    "o2_hit_rate",
    "phase3_fallback_rate",
]

TOKEN_PER_ITER_COUNT_KEYS = {
    "calm_neighbor_token_rate": "calm_neighbor_token_count",
    "local_leap_token_rate": "local_leap_token_count",
}

# ---------------------------------------------------------------------------
# 0. 日志初始化（stdout/stderr tee 到文件，judge prompt/response 写入独立 judge log）
# ---------------------------------------------------------------------------

_script_name = os.path.splitext(os.path.basename(__file__))[0]
_log_base_dir = Path(__file__).resolve().parents[1] / "log"
_eval_log_dir = _log_base_dir / "eval"
_eval_log_dir.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# 主日志：tee stdout/stderr（与 run 脚本风格一致）
_main_log_path = _eval_log_dir / f"{_ts}_{_script_name}.log"
_main_log_file = _main_log_path.open("w", encoding="utf-8")

# Judge 详细日志：每条样本的 prompt + judge 输出，便于人工核查
_judge_log_path = _eval_log_dir / f"{_ts}_{_script_name}_judge_details.log"
_judge_log_file = _judge_log_path.open("w", encoding="utf-8")


class _Tee:
    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


sys.stdout = _Tee(sys.__stdout__, _main_log_file)
sys.stderr = _Tee(sys.__stderr__, _main_log_file)
print(f"[log] 主日志输出已同时写入: {_main_log_path}")
print(f"[log] Judge 详细日志写入:   {_judge_log_path}")


def _log_judge(sample_id: str, prompt: str, response: str, correct: bool):
    """将单条样本的 judge prompt 和判断结果写入 judge 详细日志。"""
    sep = "=" * 80
    _judge_log_file.write(f"\n{sep}\n")
    _judge_log_file.write(f"[Sample ID] {sample_id}\n")
    _judge_log_file.write(f"[Judgment]  {'CORRECT' if correct else 'INCORRECT'}\n")
    _judge_log_file.write(f"[Judge Raw Response]\n{response}\n")
    _judge_log_file.write(f"[Prompt Sent to Judge]\n{prompt}\n")
    _judge_log_file.flush()


# ---------------------------------------------------------------------------
# 1. LLM-as-a-Judge（Qwen3.5-9B 等）
# ---------------------------------------------------------------------------


def _load_judge_model(model_path: str):
    """延迟加载 Judge 模型与分词器，仅在 --use_judge 时调用。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Judge model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to(torch.float32)
    model.eval()
    return model, tokenizer


def _build_judge_prompt(
    question: str,
    reference_answer: str,
    model_answer: str,
    answer_key: Optional[str] = None,
    choices: Optional[list] = None,
) -> str:
    """
    构造 Judge 的 user 消息，要求输出 CORRECT/INCORRECT。

    对于选择题（answer_key 不为 None），会额外提供正确答案字母和所有选项，
    并告知 Judge 模型回答中可能包含分析过程，只需关注最终选择的字母是否正确。
    """
    if answer_key is not None:
        # ARC 等选择题专用 prompt
        choices_text = ""
        if choices:
            choices_text = "\n".join(f"  {c['label']}. {c['text']}" for c in choices)
        return (
            "You are an expert judge for multiple-choice science questions. "
            "The model's answer may contain reasoning steps followed by a final answer. "
            "Your task: determine whether the model ultimately chose the CORRECT option.\n\n"
            "The correct answer is: {key}. {ref}\n\n"
            "---\nQuestion:\n{question}\n\n"
            "---\nAll choices:\n{choices}\n\n"
            "---\nModel's answer:\n{model_ans}\n\n"
            "---\n"
            "Does the model's final answer match option {key}? "
            "Reply with exactly one word: CORRECT or INCORRECT."
        ).format(
            key=answer_key,
            ref=reference_answer[:500] if reference_answer else "",
            question=question[:4000],
            choices=choices_text,
            model_ans=model_answer[:8000] if model_answer else "(empty)",
        )
    else:
        # 数学/代码等开放式题目
        return (
            "You are an expert judge for math and reasoning problems. "
            "Given the following problem, the reference (gold) answer, and the model's answer, "
            "determine whether the model's answer is correct. "
            "Consider numerical equality and equivalent reasoning (e.g., different wording but same result). "
            "Reply with exactly one word: CORRECT or INCORRECT.\n\n"
            "---\nProblem:\n{question}\n\n"
            "---\nReference (gold) answer:\n{ref}\n\n"
            "---\nModel's answer:\n{model_ans}\n\n"
            "---\nYour judgment (CORRECT or INCORRECT):"
        ).format(
            question=question[:4000],
            ref=reference_answer[:2000] if reference_answer else "(none)",
            model_ans=model_answer[:8000] if model_answer else "(empty)",
        )


def _judge_single_sample(
    model,
    tokenizer,
    sample: Dict,
    device: Optional[str] = None,
    max_new_tokens: int = 2048,
) -> Dict:
    """
    使用 LLM 对单条样本进行正确性判断。
    返回与 evaluate_single_sample 结构兼容的 dict，其中 correct 来自 Judge 输出。

    注意：max_new_tokens 设为 2048 而非 32，因为 Qwen3.5-9B 默认开启 thinking 模式，
    会先输出思维链再给出 CORRECT/INCORRECT，32 token 会导致输出被截断、判断全部失败。
    同时在 apply_chat_template 中显式关闭 enable_thinking（enable_thinking=False），
    让 Judge 直接输出结论，既省时间又更稳定。
    """
    import torch

    question = sample.get("question", "")
    reference_text = sample.get("reference_answer", "")
    model_text = sample.get("model_answer", "")
    answer_key = sample.get("answer_key")  # ARC 选择题有此字段，其他题目为 None
    choices = sample.get("choices")  # ARC 选择题有此字段，其他题目为 None

    user_content = _build_judge_prompt(
        question,
        reference_text,
        model_text,
        answer_key=answer_key,
        choices=choices,
    )
    messages = [{"role": "user", "content": user_content}]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,  # Judge 任务不需要思维链，关闭以节省时间并避免截断
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        if hasattr(model, "device"):
            device = next(model.parameters()).device
        else:
            device = device or "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
    except Exception as e:
        response = f"ERROR: {e}"

    response_upper = response.strip().upper()
    idx_c = response_upper.find("CORRECT")
    idx_i = response_upper.find("INCORRECT")
    if idx_i >= 0 and (idx_c < 0 or idx_i < idx_c):
        correct = False
    elif idx_c >= 0:
        correct = True
    else:
        correct = False

    # 写入 judge 详细日志，便于人工核查判断是否准确
    _log_judge(
        sample_id=sample.get("id", "unknown"),
        prompt=user_content,
        response=response.strip(),
        correct=correct,
    )

    return {
        "id": sample.get("id", "unknown"),
        "benchmark": sample.get("benchmark", "unknown"),
        "reference_answer": None,
        "model_answer": None,
        "correct": correct,
        "reference_text": reference_text,
        "model_text": model_text[:200] + ("..." if len(model_text) > 200 else ""),
        "extraction_success": True,
        "judge_response": response.strip()[:200],
    }


# ---------------------------------------------------------------------------
# 3. 批量评测与统计
# ---------------------------------------------------------------------------


def load_results_from_file(file_path: Path) -> List[Dict]:
    """从 jsonl 文件加载结果"""
    results = []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    results.append(sample)
                except json.JSONDecodeError as e:
                    print(f"[Warning] JSON decode error at {file_path}:{line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
    except Exception as e:
        print(f"[Error] Error reading {file_path}: {e}")

    return results


def evaluate_results_file(
    file_path: Path,
    judge_model,
    judge_tokenizer,
) -> Dict:
    """
    使用 LLM-as-a-Judge 评测单个结果文件。

    Returns:
        {
            'file_path': str,
            'total_samples': int,
            'correct_samples': int,
            'accuracy': float,
            'by_benchmark': {benchmark: {'total': int, 'correct': int, 'accuracy': float}},
            'judge_details': [...],
            # 以及效率相关字段（平均生成时间 / 平均输出长度等）
        }
    """
    samples = load_results_from_file(file_path)

    if not samples:
        return {
            "file_path": str(file_path),
            "total_samples": 0,
            "correct_samples": 0,
            "accuracy": 0.0,
            "by_benchmark": {},
            "judge_details": [],
            # 效率相关统计（空文件时全为 None/0）
            "avg_gen_time": None,
            "avg_output_len": None,
            "throughput_tok_per_sec": None,
            "correct_throughput": None,
            "sum_gen_time": 0.0,
            "count_gen_time": 0,
            "sum_output_len": 0,
            "count_output_len": 0,
            "sum_correct_output_len": 0,
            "count_throughput": 0,
            "avg_tpf": None,
            "macro_tpf": None,
            "sum_forward_count": 0,
            "sum_tpf_output_len": 0,
            "count_tpf_samples": 0,
            "sum_per_sample_tpf": 0.0,
            "avg_diffusion_steps": None,
            "sum_diffusion_steps": 0,
            "count_diffusion_steps": 0,
            # 阶段命中率统计（空文件）
            "phase_rate_avgs": {},
            "phase_rate_sums": {k: 0.0 for k in PHASE_RATE_KEYS},
            "phase_rate_counts": {k: 0 for k in PHASE_RATE_KEYS},
            "token_per_iter_sums": {k: 0.0 for k in TOKEN_PER_ITER_COUNT_KEYS},
            "token_per_iter_iters": {k: 0 for k in TOKEN_PER_ITER_COUNT_KEYS},
        }

    # 评测每个样本（正确性相关）
    sample_results = []
    for i, sample in enumerate(samples):
        result = _judge_single_sample(judge_model, judge_tokenizer, sample)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [Judge] {i + 1}/{len(samples)}")
        sample_results.append(result)

    # 计算总体正确性统计
    total_samples = len(sample_results)
    correct_samples = sum(1 for r in sample_results if r["correct"])
    accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

    # 计算解码效率相关统计（时间 / token 长度 / 吞吐量）
    # 结果文件来自 run_benchmark_llada2.py，包含 gen_time_sec / output_token_len 字段
    # 用 id 建立 judge 结果的索引，方便后续按正确性拆分统计
    judge_correct_by_id = {r["id"]: r["correct"] for r in sample_results}

    sum_gen_time = 0.0
    count_gen_time = 0
    sum_output_len = 0
    count_output_len = 0
    # 正确样本的输出 token 总数（用于计算正确吞吐量的分子）
    sum_correct_output_len = 0
    count_throughput = 0  # 同时拥有有效 t 和 out_len 的样本数，用于判断是否能算吞吐量

    # TPF（Tokens Per Forward）：需 jsonl 含 forward_count（新 run_benchmark 写入）
    sum_forward_count = 0
    sum_tpf_output_len = 0  # 仅含同时有 forward_count 与有效 output_len 的样本
    count_tpf_samples = 0
    sum_per_sample_tpf = 0.0  # 用于 mean( output_len / forward_count )
    sum_diffusion_steps = 0
    count_diffusion_steps = 0
    phase_rate_sums = {k: 0.0 for k in PHASE_RATE_KEYS}
    phase_rate_counts = {k: 0 for k in PHASE_RATE_KEYS}
    token_per_iter_sums = {k: 0.0 for k in TOKEN_PER_ITER_COUNT_KEYS}
    token_per_iter_iters = {k: 0 for k in TOKEN_PER_ITER_COUNT_KEYS}

    for sample in samples:
        t = sample.get("gen_time_sec", None)
        out_len = sample.get("output_token_len", None)
        is_correct = judge_correct_by_id.get(sample.get("id"), False)

        if isinstance(t, (int, float)) and t > 0:
            sum_gen_time += float(t)
            count_gen_time += 1

        if isinstance(out_len, int) and out_len >= 0:
            sum_output_len += int(out_len)
            count_output_len += 1

        if (
            isinstance(t, (int, float))
            and t > 0
            and isinstance(out_len, int)
            and out_len >= 0
        ):
            count_throughput += 1
            if is_correct:
                sum_correct_output_len += int(out_len)

        fw = sample.get("forward_count", None)
        if isinstance(fw, int) and fw > 0 and isinstance(out_len, int) and out_len >= 0:
            sum_forward_count += fw
            sum_tpf_output_len += int(out_len)
            count_tpf_samples += 1
            sum_per_sample_tpf += float(out_len) / float(fw)

        # diffusion_steps 优先使用显式字段，旧文件回退 forward_count
        ds = sample.get("diffusion_steps", fw)
        if isinstance(ds, int) and ds >= 0:
            sum_diffusion_steps += ds
            count_diffusion_steps += 1

        # 阶段统计：hit/fallback 字段为 0~1 浮点数；token-rate 字段为 tokens/iter。
        for key in PHASE_RATE_KEYS:
            v = sample.get(key, None)
            if isinstance(v, (int, float)):
                phase_rate_sums[key] += float(v)
                phase_rate_counts[key] += 1

        total_iters = sample.get("decode_total_iters", None)
        if isinstance(total_iters, int) and total_iters > 0:
            for rate_key, count_key in TOKEN_PER_ITER_COUNT_KEYS.items():
                token_count = sample.get(count_key, None)
                if isinstance(token_count, (int, float)):
                    token_per_iter_sums[rate_key] += float(token_count)
                    token_per_iter_iters[rate_key] += total_iters

    avg_gen_time = sum_gen_time / count_gen_time if count_gen_time > 0 else None
    avg_output_len = sum_output_len / count_output_len if count_output_len > 0 else None
    # 吞吐量 = 总输出 token 数 / 总生成时间
    throughput_tok_per_sec = (
        sum_output_len / sum_gen_time
        if sum_gen_time > 0 and count_throughput > 0
        else None
    )
    # 正确吞吐量 = 正确样本的总输出 token 数 / 所有样本的总生成时间
    # 精确定义：只统计判断为正确的样本贡献的 token，分母仍是全部时间（体现整体代价）
    correct_throughput = (
        sum_correct_output_len / sum_gen_time
        if sum_gen_time > 0 and count_throughput > 0
        else None
    )

    # TPF：per-sample 均值；macro = Σoutput_token / Σforward_count（与论文整体口径一致）
    avg_tpf = sum_per_sample_tpf / count_tpf_samples if count_tpf_samples > 0 else None
    macro_tpf = (
        sum_tpf_output_len / sum_forward_count if sum_forward_count > 0 else None
    )
    avg_diffusion_steps = (
        sum_diffusion_steps / count_diffusion_steps
        if count_diffusion_steps > 0
        else None
    )
    phase_rate_avgs = {
        k: phase_rate_sums[k] / phase_rate_counts[k]
        for k in PHASE_RATE_KEYS
        if phase_rate_counts[k] > 0
    }
    for key in TOKEN_PER_ITER_COUNT_KEYS:
        if token_per_iter_iters[key] > 0:
            # Prefer macro tokens/iteration when raw counts are available.
            phase_rate_avgs[key] = token_per_iter_sums[key] / token_per_iter_iters[key]

    # 按 benchmark 分组统计（正确率）
    by_benchmark = {}
    for result in sample_results:
        benchmark = result["benchmark"]
        if benchmark not in by_benchmark:
            by_benchmark[benchmark] = {"total": 0, "correct": 0}

        by_benchmark[benchmark]["total"] += 1
        if result["correct"]:
            by_benchmark[benchmark]["correct"] += 1

    # 计算每个 benchmark 的准确率
    for benchmark in by_benchmark:
        total = by_benchmark[benchmark]["total"]
        correct = by_benchmark[benchmark]["correct"]
        by_benchmark[benchmark]["accuracy"] = correct / total if total > 0 else 0.0

    return {
        "file_path": str(file_path),
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "accuracy": accuracy,
        "by_benchmark": by_benchmark,
        "judge_details": sample_results,
        # 效率相关统计
        "avg_gen_time": avg_gen_time,
        "avg_output_len": avg_output_len,
        "throughput_tok_per_sec": throughput_tok_per_sec,
        "correct_throughput": correct_throughput,
        # 中间量，供 generate_report 跨文件汇总时使用
        "sum_gen_time": sum_gen_time,
        "count_gen_time": count_gen_time,
        "sum_output_len": sum_output_len,
        "count_output_len": count_output_len,
        "sum_correct_output_len": sum_correct_output_len,
        "count_throughput": count_throughput,
        "avg_tpf": avg_tpf,
        "macro_tpf": macro_tpf,
        "sum_forward_count": sum_forward_count,
        "sum_tpf_output_len": sum_tpf_output_len,
        "count_tpf_samples": count_tpf_samples,
        "avg_diffusion_steps": avg_diffusion_steps,
        "sum_diffusion_steps": sum_diffusion_steps,
        "count_diffusion_steps": count_diffusion_steps,
        # 阶段命中率统计
        "phase_rate_avgs": phase_rate_avgs,
        "phase_rate_sums": phase_rate_sums,
        "phase_rate_counts": phase_rate_counts,
        "token_per_iter_sums": token_per_iter_sums,
        "token_per_iter_iters": token_per_iter_iters,
    }


# ---------------------------------------------------------------------------
# 4. 报告生成
# ---------------------------------------------------------------------------


def generate_report(
    evaluation_results: List[Dict],
    use_judge: bool = False,
) -> str:
    """生成评测报告。use_judge 为 True 时在标题中标注为 LLM Judge 评测。"""
    if not evaluation_results:
        return "No results to evaluate."

    report_lines = []
    report_lines.append("=" * 80)
    title = "LLaDA2.1-mini Benchmark Evaluation Report"
    title += " (LLM-as-a-Judge)"
    report_lines.append(title)
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    # 整体统计（正确性）
    total_samples = sum(r["total_samples"] for r in evaluation_results)
    total_correct = sum(r["correct_samples"] for r in evaluation_results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # 整体统计（解码效率）：从各文件的中间量汇总，保证加权正确
    total_sum_gen_time = sum(r.get("sum_gen_time", 0.0) for r in evaluation_results)
    total_count_gen_time = sum(r.get("count_gen_time", 0) for r in evaluation_results)
    total_sum_output_len = sum(r.get("sum_output_len", 0) for r in evaluation_results)
    total_count_output_len = sum(
        r.get("count_output_len", 0) for r in evaluation_results
    )
    total_sum_correct_output_len = sum(
        r.get("sum_correct_output_len", 0) for r in evaluation_results
    )
    total_count_throughput = sum(
        r.get("count_throughput", 0) for r in evaluation_results
    )
    total_sum_forward = sum(r.get("sum_forward_count", 0) for r in evaluation_results)
    total_sum_tpf_output = sum(
        r.get("sum_tpf_output_len", 0) for r in evaluation_results
    )
    total_count_tpf = sum(r.get("count_tpf_samples", 0) for r in evaluation_results)
    total_sum_diffusion_steps = sum(
        r.get("sum_diffusion_steps", 0) for r in evaluation_results
    )
    total_count_diffusion_steps = sum(
        r.get("count_diffusion_steps", 0) for r in evaluation_results
    )
    total_phase_rate_sums = {
        k: sum(r.get("phase_rate_sums", {}).get(k, 0.0) for r in evaluation_results)
        for k in PHASE_RATE_KEYS
    }
    total_phase_rate_counts = {
        k: sum(r.get("phase_rate_counts", {}).get(k, 0) for r in evaluation_results)
        for k in PHASE_RATE_KEYS
    }
    total_token_per_iter_sums = {
        k: sum(r.get("token_per_iter_sums", {}).get(k, 0.0) for r in evaluation_results)
        for k in TOKEN_PER_ITER_COUNT_KEYS
    }
    total_token_per_iter_iters = {
        k: sum(r.get("token_per_iter_iters", {}).get(k, 0) for r in evaluation_results)
        for k in TOKEN_PER_ITER_COUNT_KEYS
    }
    sum_avg_tpf_weighted = sum(
        r["avg_tpf"] * r["count_tpf_samples"]
        for r in evaluation_results
        if r.get("avg_tpf") is not None and r.get("count_tpf_samples", 0) > 0
    )

    overall_avg_gen_time = (
        total_sum_gen_time / total_count_gen_time if total_count_gen_time > 0 else None
    )
    overall_avg_output_len = (
        total_sum_output_len / total_count_output_len
        if total_count_output_len > 0
        else None
    )
    overall_throughput = (
        total_sum_output_len / total_sum_gen_time
        if total_sum_gen_time > 0 and total_count_throughput > 0
        else None
    )
    # 正确吞吐量 = 正确样本的总输出 token 数 / 所有样本的总生成时间
    overall_correct_throughput = (
        total_sum_correct_output_len / total_sum_gen_time
        if total_sum_gen_time > 0 and total_count_throughput > 0
        else None
    )
    overall_macro_tpf = (
        total_sum_tpf_output / total_sum_forward
        if total_sum_forward > 0 and total_sum_tpf_output > 0
        else None
    )
    overall_avg_tpf = (
        sum_avg_tpf_weighted / total_count_tpf
        if total_count_tpf > 0 and sum_avg_tpf_weighted > 0
        else None
    )
    overall_avg_diffusion_steps = (
        total_sum_diffusion_steps / total_count_diffusion_steps
        if total_count_diffusion_steps > 0
        else None
    )
    overall_phase_rate_avgs = {
        k: total_phase_rate_sums[k] / total_phase_rate_counts[k]
        for k in PHASE_RATE_KEYS
        if total_phase_rate_counts[k] > 0
    }
    for key in TOKEN_PER_ITER_COUNT_KEYS:
        if total_token_per_iter_iters[key] > 0:
            overall_phase_rate_avgs[key] = (
                total_token_per_iter_sums[key] / total_token_per_iter_iters[key]
            )

    report_lines.append(f"\n📊 Overall Statistics (Accuracy):")
    report_lines.append(f"  Total Samples: {total_samples}")
    report_lines.append(f"  Correct Answers: {total_correct}")
    report_lines.append(f"  Overall Accuracy: {overall_accuracy:.1%}")

    has_efficiency = any(
        v is not None
        for v in [
            overall_avg_gen_time,
            overall_avg_output_len,
            overall_throughput,
            overall_macro_tpf,
        ]
    )
    if has_efficiency:
        report_lines.append(f"\n⚙️ Overall Efficiency Statistics:")
        if overall_avg_gen_time is not None:
            report_lines.append(
                f"  Avg Generation Time per Sample: {overall_avg_gen_time:.3f} sec"
            )
        if overall_avg_output_len is not None:
            report_lines.append(
                f"  Avg Output Token Length: {overall_avg_output_len:.1f} tokens"
            )
        if overall_throughput is not None:
            report_lines.append(f"  Throughput: {overall_throughput:.2f} tokens/sec")
        if overall_correct_throughput is not None:
            report_lines.append(
                f"  Correct Throughput: {overall_correct_throughput:.2f} correct_tokens/sec"
                f"  (= throughput × accuracy, higher is better)"
            )
        if overall_macro_tpf is not None:
            report_lines.append(
                f"  TPF (macro): {overall_macro_tpf:.2f} tokens/forward  "
                f"(= Σ output tokens / Σ forward passes; higher is better)"
            )
            if overall_avg_tpf is not None:
                report_lines.append(
                    f"  TPF (mean per sample): {overall_avg_tpf:.2f} tokens/forward"
                )
        if overall_avg_diffusion_steps is not None:
            report_lines.append(
                f"  Avg Diffusion Steps per Sample: {overall_avg_diffusion_steps:.1f}"
            )
            report_lines.append(f"  Total Diffusion Steps: {total_sum_diffusion_steps}")
        elif total_count_tpf == 0:
            report_lines.append(
                "  TPF: (not available — re-run benchmark with latest run_benchmark_llada2.py "
                "to record forward_count in jsonl)"
            )
        if overall_phase_rate_avgs:
            report_lines.append("  Decode Phase Statistics:")
            phase_labels = {
                "phase1_hit_rate": "Phase-1 hit rate",
                "calm_neighbor_hit_rate": "CALM neighbor hit rate",
                "calm_neighbor_token_rate": "CALM neighbor tokens/iter",
                "phase2_trigger_rate": "Phase-2 trigger rate",
                "phase2_accepted_rate": "Phase-2 accepted rate",
                "local_leap_hit_rate": "Local-leap hit rate",
                "local_leap_token_rate": "Local-leap relaxed-neighbor tokens/iter",
                "o2_hit_rate": "O2 extra-accept rate",
                "phase3_fallback_rate": "Phase-3 fallback rate",
            }
            token_per_iter_keys = {
                "calm_neighbor_token_rate",
                "local_leap_token_rate",
            }
            for key in PHASE_RATE_KEYS:
                if key in overall_phase_rate_avgs:
                    value = overall_phase_rate_avgs[key]
                    if key in token_per_iter_keys:
                        report_lines.append(f"    {phase_labels[key]}: {value:.2f}")
                    else:
                        report_lines.append(f"    {phase_labels[key]}: {value:.1%}")

    # 按文件统计
    report_lines.append(f"\n📁 By File:")
    for result in evaluation_results:
        file_name = Path(result["file_path"]).name
        report_lines.append(f"\n  📄 {file_name}")
        report_lines.append(f"    Samples: {result['total_samples']}")
        report_lines.append(f"    Correct: {result['correct_samples']}")
        report_lines.append(f"    Accuracy: {result['accuracy']:.1%}")
        extraction_rate = result.get("extraction_success_rate")
        if extraction_rate is not None:
            report_lines.append(f"    Extraction Success: {extraction_rate:.1%}")

        has_file_efficiency = any(
            result.get(k) is not None
            for k in ["avg_gen_time", "avg_output_len", "throughput_tok_per_sec"]
        )
        if has_file_efficiency:
            report_lines.append("    Efficiency:")
            if result.get("avg_gen_time") is not None:
                report_lines.append(
                    f"      Avg Generation Time:  {result['avg_gen_time']:.3f} sec"
                )
            if result.get("avg_output_len") is not None:
                report_lines.append(
                    f"      Avg Output Length:    {result['avg_output_len']:.1f} tokens"
                )
            if result.get("throughput_tok_per_sec") is not None:
                report_lines.append(
                    f"      Throughput:           {result['throughput_tok_per_sec']:.2f} tokens/sec"
                )
            if result.get("correct_throughput") is not None:
                report_lines.append(
                    f"      Correct Throughput:   {result['correct_throughput']:.2f} correct_tokens/sec"
                )
            if result.get("macro_tpf") is not None:
                report_lines.append(
                    f"      TPF (macro):          {result['macro_tpf']:.2f} tokens/forward"
                )
                if result.get("avg_tpf") is not None:
                    report_lines.append(
                        f"      TPF (mean/sample):    {result['avg_tpf']:.2f} tokens/forward"
                    )
            if result.get("avg_diffusion_steps") is not None:
                report_lines.append(
                    f"      Avg Diffusion Steps:  {result['avg_diffusion_steps']:.1f}"
                )
                report_lines.append(
                    f"      Total Diffusion Steps:{result.get('sum_diffusion_steps', 0)}"
                )
            phase_rate_avgs = result.get("phase_rate_avgs", {})
            if phase_rate_avgs:
                report_lines.append("      Decode Phase Statistics:")
                phase_labels = {
                    "phase1_hit_rate": "Phase-1 hit rate",
                    "calm_neighbor_hit_rate": "CALM neighbor hit rate",
                    "calm_neighbor_token_rate": "CALM neighbor tokens/iter",
                    "phase2_trigger_rate": "Phase-2 trigger rate",
                    "phase2_accepted_rate": "Phase-2 accepted rate",
                    "local_leap_hit_rate": "Local-leap hit rate",
                    "local_leap_token_rate": "Local-leap relaxed-neighbor tokens/iter",
                    "o2_hit_rate": "O2 extra-accept rate",
                    "phase3_fallback_rate": "Phase-3 fallback rate",
                }
                token_per_iter_keys = {
                    "calm_neighbor_token_rate",
                    "local_leap_token_rate",
                }
                for key in PHASE_RATE_KEYS:
                    if key in phase_rate_avgs:
                        value = phase_rate_avgs[key]
                        if key in token_per_iter_keys:
                            report_lines.append(f"        {phase_labels[key]}: {value:.2f}")
                        else:
                            report_lines.append(f"        {phase_labels[key]}: {value:.1%}")

        # 按 benchmark 统计
        if result["by_benchmark"]:
            for benchmark, stats in result["by_benchmark"].items():
                report_lines.append(
                    f"      🎯 {benchmark}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})"
                )

    # 失败分析（仅规则匹配模式下有此字段，judge 模式下跳过）
    total_failed_extractions = sum(
        len(r.get("failed_extractions", [])) for r in evaluation_results
    )
    total_incorrect = sum(
        len(r.get("incorrect_samples", [])) for r in evaluation_results
    )

    if total_failed_extractions > 0:
        report_lines.append(
            f"\n❌ Failed Answer Extractions ({total_failed_extractions} samples):"
        )
        count = 0
        for result in evaluation_results:
            for failed in result.get("failed_extractions", [])[:3]:
                count += 1
                report_lines.append(
                    f"  {count}. [{failed['id']}] Reference: {failed['reference_answer']}"
                )
                report_lines.append(f"     Model text: {failed['model_text']}")
            if len(result.get("failed_extractions", [])) > 3:
                report_lines.append(
                    f"     ... and {len(result['failed_extractions']) - 3} more"
                )

    if total_incorrect > 0:
        report_lines.append(
            f"\n🚫 Incorrect Answers (Extraction Successful, {total_incorrect} samples):"
        )
        count = 0
        for result in evaluation_results:
            for incorrect in result.get("incorrect_samples", [])[:3]:
                count += 1
                report_lines.append(
                    f"  {count}. [{incorrect['id']}] Expected: {incorrect['reference_answer']}, Got: {incorrect['model_answer']}"
                )
            if len(result.get("incorrect_samples", [])) > 3:
                report_lines.append(
                    f"     ... and {len(result['incorrect_samples']) - 3} more"
                )

    report_lines.append("\n" + "=" * 80)

    return "\n".join(report_lines)


def save_report(report: str, output_file: Path):
    """保存报告到文件"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Report] Saved to: {output_file}")


# ---------------------------------------------------------------------------
# 5. 主程序
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="评测 LLaDA2.1-mini benchmark 推理结果"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results_file",
        type=str,
        help="单个结果文件路径 (jsonl格式)",
    )
    group.add_argument(
        "--results_dir",
        type=str,
        help="结果文件目录，将评测所有 .jsonl 文件",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告文件路径（默认自动生成）",
    )
    parser.add_argument(
        "--judge_model_path",
        type=str,
        default=DEFAULT_JUDGE_MODEL_PATH,
        help=f"Judge 模型路径（默认: {DEFAULT_JUDGE_MODEL_PATH}）",
    )

    args = parser.parse_args()

    # 预加载 Judge 模型（本脚本仅支持 LLM-as-a-Judge 评测）
    judge_model, judge_tokenizer = None, None
    print(f"[Judge] Loading judge model from: {args.judge_model_path}")
    try:
        judge_model, judge_tokenizer = _load_judge_model(args.judge_model_path)
        print("[Judge] Model loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to load judge model: {e}")
        return

    # 收集要评测的文件
    files_to_evaluate = []

    if args.results_file:
        file_path = Path(args.results_file)
        if not file_path.exists():
            print(f"[Error] File not found: {file_path}")
            return
        files_to_evaluate = [file_path]

    elif args.results_dir:
        dir_path = Path(args.results_dir)
        if not dir_path.exists():
            print(f"[Error] Directory not found: {dir_path}")
            return

        # 查找所有 .jsonl 文件
        jsonl_files = list(dir_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"[Error] No .jsonl files found in: {dir_path}")
            return

        files_to_evaluate = sorted(jsonl_files)

    print(f"[Evaluate] Found {len(files_to_evaluate)} file(s) to evaluate:")
    for file_path in files_to_evaluate:
        print(f"  - {file_path}")

    # 评测每个文件
    evaluation_results = []
    for file_path in files_to_evaluate:
        print(f"\n[Evaluating] {file_path.name}...")
        result = evaluate_results_file(
            file_path,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
        )
        evaluation_results.append(result)
        print(
            f"  ✓ {result['total_samples']} samples, {result['accuracy']:.1%} accuracy"
        )

    # 生成报告
    report = generate_report(evaluation_results, use_judge=True)

    # 输出报告
    print("\n" + report)

    # 保存报告（默认输出到与 src/ 同级的 experiments/evals/）
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.results_file:
            input_name = Path(args.results_file).stem
        else:
            input_name = "report"
        DEFAULT_EVALS_DIR.mkdir(parents=True, exist_ok=True)
        output_file = DEFAULT_EVALS_DIR / f"evaluation_{input_name}_{timestamp}.txt"

    save_report(report, output_file)


if __name__ == "__main__":
    main()
