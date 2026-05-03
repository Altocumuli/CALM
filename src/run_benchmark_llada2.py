import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Configure GPU visibility and cache locations outside the script, e.g.:
#   HIP_VISIBLE_DEVICES=0 HF_HOME=/path/to/cache python src/run_benchmark_llada2.py ...
os.environ.setdefault("DLM_DATA_PARALLEL", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_rope_utils

# CCD baseline
from llada_ccd_decode import generate_with_ccd, CcdConfig

# CALM decoder
from llada_calm_decode import generate_with_calm, CalmConfig

# LocalLeap baseline
from llada_localleap_decode import generate_with_localleap, LocalLeapConfig


"""Run LLaDA2.1-mini benchmarks with baseline, CCD, CALM, and LocalLeap decoding.

The script writes JSONL records with generated text and efficiency metrics such as
wall time, output length, forward count, and decode-stage statistics when available.

Example:
    python src/run_benchmark_llada2.py --benchmark gsm8k_test_only --decode_mode calm \
        --calm_radius 2 --calm_max_accept 1 --calm_tau_start 0.70 --calm_tau_end 0.50
"""


# ---------------------------------------------------------------------------
# 1. 日志与环境初始化（与 test_llada2_1_mini.py 风格一致）
# ---------------------------------------------------------------------------

_script_name = os.path.splitext(os.path.basename(__file__))[0]
_log_dir = (Path(__file__).resolve().parents[1] / "log" / "run").as_posix()
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(
    _log_dir,
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_script_name}.log",
)
_log_file = open(_log_path, "w", encoding="utf-8")


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


sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
print(f"[log] 输出已同时写入: {_log_path}")


# ---------------------------------------------------------------------------
# 2. 环境与 RoPE 修复（与 test_llada2_1_mini.py 保持一致）
# ---------------------------------------------------------------------------


def _compute_default_rope_parameters(
    config, device=None, seq_len=None, layer_type=None
):
    """标准 RoPE（无 scaling），与 transformers 其它 ROPE 初始化函数签名一致。"""
    base = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    partial = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float() / dim)
    )
    return inv_freq, 1.0


if "default" not in getattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", {}):
    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = (
        _compute_default_rope_parameters
    )


# ---------------------------------------------------------------------------
# 2. 模型加载（只加载一次，后续循环复用）
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("CALM_MODEL_ID", "inclusionAI/LLaDA2.1-mini")


def load_llada_model_and_tokenizer():
    """
    加载 LLaDA2.1-mini。
    - 若设置 DLM_CPU=1，则在 CPU 上单卡推理；
    - 否则：
      - 若设置 DLM_DATA_PARALLEL=1 且当前可见 GPU 数 > 1，则使用 DataParallel；
      - 否则在当前可见 GPU 上单卡推理（通过 HIP_VISIBLE_DEVICES 控制）。
    """
    use_cpu = bool(os.environ.get("DLM_CPU"))
    use_dp = bool(os.environ.get("DLM_DATA_PARALLEL"))

    if use_cpu or not torch.cuda.is_available():
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map=None,
        )
        model = model.to(torch.bfloat16).to(device)
    else:
        num_devices = torch.cuda.device_count()
        if use_dp and num_devices > 1:
            # 数据并行：在每张可见 GPU 上放一份完整模型副本，由 DataParallel 自动把 batch 里的不同样本分配到不同 GPU。
            base_device = torch.device("cuda:0")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                device_map=None,
            )
            model = model.to(torch.bfloat16).to(base_device)
            model = torch.nn.DataParallel(model)
            print(
                f"[devices] DataParallel on {list(range(num_devices))} (logical device ids)"
            )
        else:
            # 默认：仍然交给 device_map="auto" 在可见卡之间自动放置/切分
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.bfloat16,
            )
            # 打印实际 device 映射，方便你检查到底用了哪些卡
            try:
                device_map = getattr(model, "hf_device_map", None)
                print(f"[devices] hf_device_map={device_map}")
            except Exception:
                pass

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. 通用工具：读 jsonl、写 jsonl
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]  # 指向 dlm/
BENCH_ROOT = ROOT / "experiments" / "benchmarks"
RUN_ROOT = ROOT / "experiments" / "runs"


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_completed_sample_ids(path: Path) -> set:
    """读取已有结果文件中的 sample id，用于续跑时跳过已完成样本。"""
    if not path.is_file():
        return set()
    done_ids = set()
    for row in read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is not None:
            done_ids.add(sample_id)
    return done_ids


# ---------------------------------------------------------------------------
# 4. 构造 chat 输入并调用 generate（baseline 策略）
# ---------------------------------------------------------------------------


def run_generate_single(
    model,
    tokenizer,
    prompt: str,
    decode_mode: str = "baseline",
    gen_length: int = 2048,
    clad_overrides: Optional[Dict] = None,
    stats_out: Optional[List] = None,
) -> Tuple[str, int]:
    """
    使用指定的解码策略进行生成。

    clad_overrides: dict，可覆盖 CALM / LocalLeap 超参数。
    stats_out: 若传入非 None 的列表，CALM / LocalLeap 运行后会 append DecodeStats 实例，
        可从中读取各阶段命中率。

    Returns:
        (生成的文本, forward_count)
    """
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    ov = clad_overrides or {}

    if decode_mode == "baseline":
        return _run_baseline_generate(
            actual_model, tokenizer, prompt, gen_length=gen_length
        )
    if decode_mode == "ccd":
        return _run_ccd_generate(actual_model, tokenizer, prompt, gen_length=gen_length)
    if decode_mode == "calm":
        return _run_calm_generate(
            actual_model,
            tokenizer,
            prompt,
            gen_length=gen_length,
            overrides=ov,
            stats_out=stats_out,
        )
    if decode_mode == "localleap":
        return _run_localleap_generate(
            actual_model,
            tokenizer,
            prompt,
            gen_length=gen_length,
            overrides=ov,
            stats_out=stats_out,
        )
    raise ValueError(
        f"Unsupported decode_mode: {decode_mode}. "
        "Supported modes: ['baseline', 'ccd', 'calm', 'localleap']"
    )


def _run_baseline_generate(
    model, tokenizer, prompt: str, gen_length: int = 2048
) -> Tuple[str, int]:
    """Baseline LLaDA2.1-mini 解码策略。forward 次数通过对主干 ``model.model`` 注册 hook 统计。"""
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
    input_ids = input_ids.long()

    # 将输入移动到模型设备
    main_device = None
    try:
        main_device = next(model.parameters()).device
    except StopIteration:
        main_device = None
    if main_device is not None and main_device.type != "cpu":
        input_ids = input_ids.to(main_device)

    inner = getattr(model, "model", None)
    forward_count = [0]
    hook_handle = None
    if inner is not None:

        def _count_forward(_module, _inp, _out):
            forward_count[0] += 1

        hook_handle = inner.register_forward_hook(_count_forward)

    try:
        with torch.no_grad():
            generated_tokens = model.generate(
                inputs=input_ids,
                eos_early_stop=True,
                gen_length=gen_length,
                block_length=32,
                threshold=0.7,
                editing_threshold=0.5,
                temperature=0.0,
                max_post_steps=16,
            )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return text.strip(), forward_count[0]



def _run_ccd_generate(
    model, tokenizer, prompt: str, gen_length: int = 2048
) -> Tuple[str, int]:
    """CCD (Coherent and Consistent Decoding) 双步一致性加速解码策略"""
    ccd_config = CcdConfig(
        seek_mode=True,
        history_depth=2,
        top_v=4,
        gen_length=gen_length,
        block_length=32,
        threshold=0.7,
        editing_threshold=0.5,
        temperature=0.0,
        max_post_steps=16,
        eos_early_stop=True,
        eos_id=156892,
        mask_id=156895,
    )
    text, n_fw = generate_with_ccd(model, tokenizer, prompt, ccd_config)
    return text, n_fw



def _run_calm_generate(
    model,
    tokenizer,
    prompt: str,
    gen_length: int = 2048,
    overrides: Optional[Dict] = None,
    stats_out: Optional[List] = None,
) -> Tuple[str, int]:
    """CALM：Phase-1 consistency anchors + local neighborhood acceptance + fallback。"""
    ov = overrides or {}
    calm_config = CalmConfig(
        top_v=ov.get("top_v", 4),
        neighbor_radius=ov.get("neighbor_radius", 1),
        max_neighbor_accept_per_anchor=ov.get("max_neighbor_accept_per_anchor", 1),
        anchor_mode=ov.get("calm_anchor_mode", "consistency"),
        random_anchor_seed=ov.get("calm_random_anchor_seed", 0),
        local_threshold_start=ov.get("local_threshold_start", 0.90),
        local_threshold_end=ov.get("local_threshold_end", 0.72),
        local_threshold_gamma=ov.get("local_threshold_gamma", 1.0),
        gen_length=gen_length,
        block_length=32,
        threshold=0.7,
        editing_threshold=0.5,
        temperature=0.0,
        max_post_steps=16,
        eos_early_stop=True,
        eos_id=156892,
        mask_id=156895,
    )
    return generate_with_calm(
        model, tokenizer, prompt, calm_config, stats_out=stats_out
    )


def _run_localleap_generate(
    model,
    tokenizer,
    prompt: str,
    gen_length: int = 2048,
    overrides: Optional[Dict] = None,
    stats_out: Optional[List] = None,
) -> Tuple[str, int]:
    """LocalLeap：current-confidence anchor + relaxed local neighborhood propagation。"""
    ov = overrides or {}
    localleap_config = LocalLeapConfig(
        anchor_threshold=ov.get("localleap_anchor_threshold", 0.90),
        relaxed_threshold=ov.get("localleap_relaxed_threshold", 0.75),
        local_radius=ov.get("localleap_radius", 4),
        gen_length=gen_length,
        block_length=32,
        threshold=0.7,
        editing_threshold=0.5,
        temperature=0.0,
        max_post_steps=16,
        eos_early_stop=True,
        eos_id=156892,
        mask_id=156895,
    )
    return generate_with_localleap(
        model, tokenizer, prompt, localleap_config, stats_out=stats_out
    )


# ---------------------------------------------------------------------------
# 5. 针对不同 benchmark 的适配
# ---------------------------------------------------------------------------


def iter_gsm8k_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "math" / "gsm8k_small.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_gsm8k_test_only_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "math" / "gsm8k_testOnly.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}。请先运行: python dlm/src/download_benchmarks.py --dataset gsm8k_test_only"
        )
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_aime2025_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "math" / "aime2025_all.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_math500_examples(max_examples: int | None = None):
    """Hendrycks MATH 测试集 500 题（download_benchmarks.py --dataset math500）。"""
    path = BENCH_ROOT / "math" / "math500.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}。请先运行: python dlm/src/download_benchmarks.py --dataset math500"
        )
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_humaneval_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "code" / "humaneval_all.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_mbpp_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "code" / "mbpp_sanitized.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_arc_easy_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "reasoning" / "arc_easy_300.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def iter_arc_challenge_examples(max_examples: int | None = None):
    path = BENCH_ROOT / "reasoning" / "arc_challenge_300.jsonl"
    for i, ex in enumerate(read_jsonl(path)):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def _format_arc_prompt(ex: dict) -> str:
    """
    将 ARC 样本格式化为模型输入 prompt。
    要求模型给出完整的分析过程，并在最后明确指出答案字母，
    便于后续 LLM Judge 判断正确性，同时保留推理过程供错误分析。
    """
    choices_text = "\n".join(f"  {c['label']}. {c['text']}" for c in ex["choices"])
    return (
        "Answer the following multiple-choice science question. "
        "Think through the problem step by step, then clearly state your final answer "
        "as one of the options (A, B, C, or D).\n\n"
        f"Question: {ex['question']}\n\n"
        f"Choices:\n{choices_text}"
    )


# ---------------------------------------------------------------------------
# 6. 主流程
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run LLaDA2.1-mini benchmarks with baseline/CCD/CALM/LocalLeap decoding"
    )
    _all_benchmarks = [
        "gsm8k_small",
        "gsm8k_test_only",
        "math500",
        "aime2025_all",
        "humaneval_all",
        "mbpp_sanitized",
        "arc_easy",
        "arc_challenge",
    ]
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        choices=_all_benchmarks,
        help="要跑的基准数据集标识",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=_all_benchmarks,
        help="要跑的多个基准数据集（用空格分隔）",
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="baseline",
        choices=["baseline", "ccd", "calm", "localleap"],
        help="Decoding strategy: baseline / ccd / calm / localleap",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="最多跑多少条样本（默认全部）",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="按数据集原始顺序切片的起始下标，0-based，包含该下标",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="按数据集原始顺序切片的结束下标，0-based，不包含该下标",
    )
    parser.add_argument(
        "--resume_file",
        type=str,
        default=None,
        help="续跑已有 jsonl 结果文件：读取已完成样本 id，跳过后继续 append 到该文件",
    )
    parser.add_argument(
        "--calm_radius",
        type=int,
        default=None,
        metavar="R",
        help="覆盖 CALM 的 neighbor_radius（默认 1）",
    )
    parser.add_argument(
        "--calm_max_accept",
        type=int,
        default=None,
        metavar="M",
        help="覆盖 CALM 的 max_neighbor_accept_per_anchor（默认 1）",
    )
    parser.add_argument(
        "--calm_tau_start",
        type=float,
        default=None,
        metavar="tau_s",
        help="覆盖 CALM 的 local_threshold_start（默认 0.90）",
    )
    parser.add_argument(
        "--calm_tau_end",
        type=float,
        default=None,
        metavar="tau_e",
        help="覆盖 CALM 的 local_threshold_end（默认 0.72）",
    )
    parser.add_argument(
        "--calm_tau_gamma",
        type=float,
        default=None,
        metavar="gamma",
        help="覆盖 CALM 的 local_threshold_gamma（默认 1.0）",
    )
    parser.add_argument(
        "--calm_anchor_mode",
        type=str,
        choices=["consistency", "random"],
        default=None,
        help="CALM anchor 消融：consistency 使用跨步一致 anchor；random 使用匹配数量的随机 anchor（默认 consistency）",
    )
    parser.add_argument(
        "--calm_random_seed",
        type=int,
        default=None,
        metavar="seed",
        help="CALM random-anchor 消融的随机种子（默认 0）",
    )
    parser.add_argument(
        "--localleap_anchor_threshold",
        type=float,
        default=None,
        metavar="thr_anchor",
        help="覆盖 LocalLeap 的 anchor_threshold（默认 0.90）",
    )
    parser.add_argument(
        "--localleap_relaxed_threshold",
        type=float,
        default=None,
        metavar="thr_relaxed",
        help="覆盖 LocalLeap 的 relaxed_threshold（默认 0.75）",
    )
    parser.add_argument(
        "--localleap_radius",
        type=int,
        default=None,
        metavar="R",
        help="覆盖 LocalLeap 的 local_radius（默认 4）",
    )
    args = parser.parse_args()

    # 构建超参数覆盖字典（仅含显式传入的参数）
    clad_overrides: Dict = {}
    if args.calm_radius is not None:
        clad_overrides["neighbor_radius"] = args.calm_radius
    if args.calm_max_accept is not None:
        clad_overrides["max_neighbor_accept_per_anchor"] = args.calm_max_accept
    if args.calm_tau_start is not None:
        clad_overrides["local_threshold_start"] = args.calm_tau_start
    if args.calm_tau_end is not None:
        clad_overrides["local_threshold_end"] = args.calm_tau_end
    if args.calm_tau_gamma is not None:
        clad_overrides["local_threshold_gamma"] = args.calm_tau_gamma
    if args.calm_anchor_mode is not None:
        clad_overrides["calm_anchor_mode"] = args.calm_anchor_mode
    if args.calm_random_seed is not None:
        clad_overrides["calm_random_anchor_seed"] = args.calm_random_seed
    if args.localleap_anchor_threshold is not None:
        clad_overrides["localleap_anchor_threshold"] = args.localleap_anchor_threshold
    if args.localleap_relaxed_threshold is not None:
        clad_overrides["localleap_relaxed_threshold"] = args.localleap_relaxed_threshold
    if args.localleap_radius is not None:
        clad_overrides["localleap_radius"] = args.localleap_radius

    # 参数验证：必须指定 --benchmark 或 --benchmarks 中的一个
    if not args.benchmark and not args.benchmarks:
        parser.error("必须指定 --benchmark 或 --benchmarks 中的一个")
    if args.benchmark and args.benchmarks:
        parser.error("不能同时指定 --benchmark 和 --benchmarks")
    if args.resume_file and args.benchmarks:
        parser.error("--resume_file 目前仅支持与单个 --benchmark 一起使用")
    if (args.start_index is None) != (args.end_index is None):
        parser.error("--start_index 和 --end_index 必须同时指定")
    if args.start_index is not None:
        if args.start_index < 0 or args.end_index <= args.start_index:
            parser.error("--start_index/--end_index 必须满足 0 <= start < end")
        if args.max_examples is not None:
            parser.error("--start_index/--end_index 与 --max_examples 不能同时使用，避免切片语义混淆")
    range_tag = ""
    if args.start_index is not None:
        range_tag = f"_idx{args.start_index:04d}-{args.end_index:04d}"

    # 确定要处理的 benchmark 列表
    if args.benchmark:
        benchmarks_to_run = [args.benchmark]
    else:
        benchmarks_to_run = args.benchmarks

    print(f"[main] 将依次处理以下 benchmarks: {benchmarks_to_run}")

    # 一次性加载模型，多个 benchmark 共用
    model, tokenizer = load_llada_model_and_tokenizer()

    total_processed = 0

    # 循环处理每个 benchmark
    for benchmark_name in benchmarks_to_run:
        print(f"\n{'='*60}")
        print(f"开始处理 benchmark: {benchmark_name}")
        print(f"{'='*60}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 若有超参数覆盖，在文件名中追加 tag，便于区分消融实验结果
        _ov_tag = ""
        if clad_overrides and args.decode_mode in ("calm", "localleap"):
            parts = []
            if args.decode_mode == "calm":
                if "calm_anchor_mode" in clad_overrides:
                    parts.append(f"anchor_{clad_overrides['calm_anchor_mode']}")
                if "calm_random_anchor_seed" in clad_overrides:
                    parts.append(f"seed{clad_overrides['calm_random_anchor_seed']}")
                if "neighbor_radius" in clad_overrides:
                    parts.append(f"r{clad_overrides['neighbor_radius']}")
                if "max_neighbor_accept_per_anchor" in clad_overrides:
                    parts.append(f"m{clad_overrides['max_neighbor_accept_per_anchor']}")
                if "local_threshold_start" in clad_overrides:
                    parts.append(
                        f"ts{clad_overrides['local_threshold_start']:.2f}".replace(
                            ".", ""
                        )
                    )
                if "local_threshold_end" in clad_overrides:
                    parts.append(
                        f"te{clad_overrides['local_threshold_end']:.2f}".replace(
                            ".", ""
                        )
                    )
                if "local_threshold_gamma" in clad_overrides:
                    parts.append(
                        f"g{clad_overrides['local_threshold_gamma']:.2f}".replace(
                            ".", ""
                        )
                    )
            if args.decode_mode == "localleap":
                if "localleap_radius" in clad_overrides:
                    parts.append(f"r{clad_overrides['localleap_radius']}")
                if "localleap_anchor_threshold" in clad_overrides:
                    parts.append(
                        f"ta{clad_overrides['localleap_anchor_threshold']:.2f}".replace(
                            ".", ""
                        )
                    )
                if "localleap_relaxed_threshold" in clad_overrides:
                    parts.append(
                        f"tr{clad_overrides['localleap_relaxed_threshold']:.2f}".replace(
                            ".", ""
                        )
                    )
            if parts:
                _ov_tag = "_" + "_".join(parts)
        if args.resume_file:
            out_path = Path(args.resume_file)
        else:
            out_path = (
                RUN_ROOT
                / f"{ts}_llada2_{benchmark_name}_decode={args.decode_mode}{_ov_tag}{range_tag}.jsonl"
            )
        print(f"[run] 输出结果将写入: {out_path}")
        if args.start_index is not None:
            print(
                f"[run] 样本切片: index in [{args.start_index}, {args.end_index}) "
                f"（0-based，按 benchmark 原始 jsonl 顺序）"
            )
        if clad_overrides:
            print(f"[run] Decode hyperparameter overrides: {clad_overrides}")

        completed_ids = set()
        if args.resume_file:
            if not out_path.is_file():
                parser.error(f"--resume_file 指向的文件不存在: {out_path}")
            existing_rows = list(read_jsonl(out_path))
            if existing_rows:
                first = existing_rows[0]
                file_benchmark = first.get("benchmark")
                file_decode_mode = first.get("decode_mode")
                if file_benchmark != benchmark_name:
                    parser.error(
                        f"--resume_file benchmark 不匹配：文件中是 {file_benchmark}，当前是 {benchmark_name}"
                    )
                if file_decode_mode != args.decode_mode:
                    parser.error(
                        f"--resume_file decode_mode 不匹配：文件中是 {file_decode_mode}，当前是 {args.decode_mode}"
                    )
                file_overrides = first.get("clad_overrides")
                if (file_overrides or None) != (clad_overrides or None):
                    parser.error(
                        "--resume_file 的 clad_overrides 与当前命令不一致，"
                        f"文件中是 {file_overrides}，当前是 {clad_overrides or None}"
                    )
            completed_ids = load_completed_sample_ids(out_path)
            print(
                f"[resume] 已完成样本数: {len(completed_ids)}，将跳过这些 id 并继续追加写入"
            )

        # 根据 benchmark 名称选择对应的迭代器
        if benchmark_name == "gsm8k_small":
            iterator = iter_gsm8k_examples(args.max_examples)
        elif benchmark_name == "gsm8k_test_only":
            iterator = iter_gsm8k_test_only_examples(args.max_examples)
        elif benchmark_name == "aime2025_all":
            iterator = iter_aime2025_examples(args.max_examples)
        elif benchmark_name == "math500":
            iterator = iter_math500_examples(args.max_examples)
        elif benchmark_name == "humaneval_all":
            iterator = iter_humaneval_examples(args.max_examples)
        elif benchmark_name == "mbpp_sanitized":
            iterator = iter_mbpp_examples(args.max_examples)
        elif benchmark_name == "arc_easy":
            iterator = iter_arc_easy_examples(args.max_examples)
        elif benchmark_name == "arc_challenge":
            iterator = iter_arc_challenge_examples(args.max_examples)
        else:
            print(f"[错误] 未知 benchmark: {benchmark_name}，跳过")
            continue

        # 处理当前 benchmark 的所有样本
        processed = 0
        skipped_completed = 0
        skipped_out_of_range = 0
        for sample_index, ex in enumerate(iterator):
            if args.start_index is not None:
                if sample_index < args.start_index:
                    skipped_out_of_range += 1
                    continue
                if sample_index >= args.end_index:
                    break

            sample_id = ex.get("id")
            if sample_id in completed_ids:
                skipped_completed += 1
                continue

            # 根据 benchmark 类型获取问题/提示和参考答案
            if benchmark_name in [
                "gsm8k_small",
                "gsm8k_test_only",
                "aime2025_all",
                "math500",
            ]:
                # 数学基准测试：使用 "question" 字段
                q = ex["question"]
                ref_ans = ex.get("answer")
            elif benchmark_name in ["humaneval_all", "mbpp_sanitized"]:
                # 代码基准测试：使用 "prompt" 字段，构造英文代码生成任务
                if benchmark_name == "humaneval_all":
                    q = (
                        "Complete the following Python function. "
                        "Return only valid Python code for the function implementation, without any explanations.\n\n"
                        f"{ex['prompt']}"
                    )
                else:
                    q = (
                        "Write a Python function that solves the following task.\n"
                        f"Task description: {ex['prompt']}\n\n"
                        "Return only the complete Python function implementation, without any explanations."
                    )
                ref_ans = ex.get("reference_code")
            elif benchmark_name in ["arc_easy", "arc_challenge"]:
                # ARC 选择题：格式化题目 + 选项，参考答案为正确选项的文本
                q = _format_arc_prompt(ex)
                ref_ans = ex.get("answer_text")
            else:
                print(f"[错误] 未知 benchmark 类型: {benchmark_name}，跳过样本")
                continue

            print(f"\n[{benchmark_name}|id={sample_id}] 开始生成...")
            # 与真实送入模型的 chat 模板一致（用于统计 prompt token 数）
            chat_inp = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if isinstance(chat_inp, torch.Tensor):
                _inp = chat_inp
            else:
                _inp = chat_inp["input_ids"]
            if not isinstance(_inp, torch.Tensor):
                _inp = torch.tensor(_inp, dtype=torch.long)
            if _inp.dim() == 1:
                _inp = _inp.unsqueeze(0)
            input_token_len = int(_inp.shape[-1])

            # 扩散模型 forward 序列长度 ≈ input + gen_length；2048 为默认生成空间。
            gen_length = 2048

            t_start = time.time()
            _sample_stats: List = []
            gen_text, forward_count = run_generate_single(
                model,
                tokenizer,
                prompt=q,
                decode_mode=args.decode_mode,
                gen_length=gen_length,
                clad_overrides=clad_overrides if clad_overrides else None,
                stats_out=(
                    _sample_stats
                    if args.decode_mode
                    in ("calm", "localleap")
                    else None
                ),
            )
            gen_time_sec = float(time.time() - t_start)

            # 统计输出 token 长度（只对生成的文本再次分词，评估解码效率）
            try:
                output_ids_for_len = tokenizer(
                    gen_text, add_special_tokens=True, return_tensors="pt"
                )["input_ids"]
                output_token_len = int(output_ids_for_len.shape[-1])
            except Exception:
                # 兜底：如果分词失败，长度记为 -1
                output_token_len = -1

            print(
                f"[生成完成] time={gen_time_sec:.3f}s, "
                f"in_len={input_token_len}, out_len={output_token_len}, "
                f"forwards={forward_count} | "
                f"{gen_text[:120].replace(chr(10), ' ')}{'...' if len(gen_text) > 120 else ''}"
            )

            # 构造输出记录，根据基准测试类型使用不同的字段名
            record = {
                "id": sample_id,
                "benchmark": benchmark_name,
                "decode_mode": args.decode_mode,
                "sample_index": sample_index,
                "model_answer": gen_text,
                # 供评测脚本使用的效率指标
                "input_token_len": input_token_len,
                "output_token_len": output_token_len,
                "gen_time_sec": gen_time_sec,
                "gen_length": gen_length,
                "forward_count": forward_count,
            }
            # 若有超参数覆盖，记录实际使用的参数（消融实验追溯用）
            if clad_overrides:
                record["clad_overrides"] = clad_overrides
            # 写入 CLAD 解码阶段命中率（由 DecodeStats 提供）
            if _sample_stats:
                record.update(_sample_stats[0].hit_rates())

            # 根据基准测试类型添加特定字段
            if benchmark_name in [
                "gsm8k_small",
                "gsm8k_test_only",
                "aime2025_all",
                "math500",
            ]:
                # 数学基准测试：保留原有字段名
                record["question"] = ex["question"]
                record["reference_answer"] = ref_ans
                if benchmark_name == "math500":
                    # 供 level-5 子集等分层分析
                    if ex.get("level") is not None:
                        record["level"] = ex["level"]
                    if ex.get("subject") is not None:
                        record["subject"] = ex["subject"]
            elif benchmark_name in ["humaneval_all", "mbpp_sanitized"]:
                # 代码基准测试：使用更具描述性的字段名
                record["prompt"] = ex["prompt"]  # 原始任务描述
                record["formatted_question"] = q  # 我们构造的完整提示
                record["reference_code"] = ref_ans  # 参考代码实现
                # 保存测试用例，供后续代码执行评估使用
                if benchmark_name == "humaneval_all":
                    record["tests"] = ex["tests"]
                    record["entry_point"] = ex["entry_point"]
                else:  # mbpp_sanitized
                    record["tests"] = ex["tests"]
            elif benchmark_name in ["arc_easy", "arc_challenge"]:
                # ARC 选择题：保存题目、选项、正确答案 key 和文本
                record["question"] = ex["question"]
                record["choices"] = ex["choices"]
                record["answer_key"] = ex.get("answer_key")
                record["reference_answer"] = ref_ans  # 正确选项的文本，供 judge 评测
            append_jsonl(out_path, [record])
            processed += 1

        print(
            f"[{benchmark_name}] 完成，新增样本数: {processed}，"
            f"跳过已完成样本数: {skipped_completed}，"
            f"切片范围外跳过: {skipped_out_of_range}，结果保存在: {out_path}"
        )
        total_processed += processed

    print(f"\n[全部完成] 总共处理样本数: {total_processed}")
    print("所有结果文件已保存在 dlm/experiments/runs/ 目录下")


if __name__ == "__main__":
    main()
