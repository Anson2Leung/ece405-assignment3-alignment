#!/usr/bin/env bash
# sft_experiment2.sh
# for filter correct answeres

set -euo pipefail

OUTPUT_ROOT="/home/ansonl32/koa_scratch/ECE405/assignment3/sft_expert_experiment"
OUT_DIR="${OUTPUT_ROOT}/exp2_full"

MODEL_ID="/home/ansonl32/koa_scratch/ECE405/assignment3/Qwen/Qwen2.5-Math-1.5B"
MATH_PATH="/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet"
PROMPT_FILE="/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/prompts/r1_zero.prompt"
FILTERED_PATH="/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math_filtered_correct.parquet"

POLICY_DEVICE="cuda:0"
VLLM_DEVICE="cuda:1"
SEED=42

# Filter dataset
echo "══════════════════════════════════════════════════════════════"
echo " Step 1: Filtering dataset to correctly-answered examples"
echo "══════════════════════════════════════════════════════════════"

mkdir -p ./outputs

uv run python filter_correct.py \
    --math_path   "${MATH_PATH}"   \
    --prompt_file "${PROMPT_FILE}" \
    --model_id    "${MODEL_ID}"    \
    --output_path "${FILTERED_PATH}" \
    --device      "${POLICY_DEVICE}" \
    --vllm_mem    0.60              \
    --batch_size  16                \
    --max_tokens  1024              \
    --temperature 0.0               \
    --seed        "${SEED}"

echo ""
echo "Filtering done.  Filtered parquet saved to: ${FILTERED_PATH}"
echo ""

echo "══════════════════════════════════════════════════════════════"
echo " Step 2: Training SFT on filtered dataset"
echo "══════════════════════════════════════════════════════════════"

uv run python sft.py \
    --model_id         "${MODEL_ID}"     \
    --math_path        "${FILTERED_PATH}" \
    --prompt_file      "${PROMPT_FILE}"  \
    --output_dir       "${OUT_DIR}"      \
    --n_sft_steps      200              \
    --batch_size       1                \
    --grad_accum_steps 4                \
    --lr               5e-5             \
    --warmup_steps     20               \
    --max_seq_len      512              \
    --max_samples      -1               \
    --eval_every       25               \
    --num_val_generate 200              \
    --num_to_log       8                \
    --policy_device    "${POLICY_DEVICE}" \
    --vllm_device      "${VLLM_DEVICE}"  \
    --seed             "${SEED}"         \
    --wandb_project    "ece405_assignment3"        \
    --wandb_run_name   "sft_filtered_full_lr2e-5"

echo ""
echo "Experiment 2 complete."