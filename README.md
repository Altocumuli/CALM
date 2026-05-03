# CALM: Consistency-Anchored Local Mask Propagation

This repository contains the benchmark, decoding, and evaluation code for:

**CALM: Consistency-Anchored Local Mask Propagation for Improved Diffusion Decoding**

CALM is a lightweight decoding strategy for diffusion language models. It treats cross-step consistent predictions as anchors of locally stabilized structure, and performs local mask propagation only around those anchors under an annealed confidence threshold.

## Contents

- `src/run_benchmark_llada2.py`: benchmark runner for LLaDA2.1-mini.
- `src/evaluate_benchmark_results.py`: accuracy and efficiency evaluation script.
- `src/llada_calm_decode.py`: CALM decoding implementation.
- `src/llada_ccd_decode.py`: CCD baseline implementation.
- `src/llada_localleap_decode.py`: LocalLeap baseline implementation.
- `src/llada_clad_decode.py`: shared helper utilities used by CALM and LocalLeap.

## Setup

```bash
pip install -r requirements.txt
```

The paper experiments use `LLaDA2.1-mini` as the backbone model and a judge model for answer evaluation. You can override model locations without editing code:

```bash
export CALM_MODEL_ID=inclusionAI/LLaDA2.1-mini
export CALM_JUDGE_MODEL_PATH=/path/to/judge/model
```

## Run Benchmarks

Run CALM on GSM8K:

```bash
python src/run_benchmark_llada2.py \
  --benchmark gsm8k_test_only \
  --decode_mode calm \
  --calm_radius 2 \
  --calm_max_accept 1 \
  --calm_tau_start 0.70 \
  --calm_tau_end 0.50
```

Run CCD:

```bash
python src/run_benchmark_llada2.py \
  --benchmark gsm8k_test_only \
  --decode_mode ccd
```

Run LocalLeap:

```bash
python src/run_benchmark_llada2.py \
  --benchmark gsm8k_test_only \
  --decode_mode localleap \
  --localleap_radius 2 \
  --localleap_anchor_threshold 0.70 \
  --localleap_relaxed_threshold 0.50
```

## Evaluate Results

```bash
python src/evaluate_benchmark_results.py \
  --results_file experiments/runs/YOUR_RESULT.jsonl \
  --judge_model_path /path/to/judge/model
```

Evaluation reports are written to `experiments/evals/` by default.

## Citation

Citation information will be added after release.
