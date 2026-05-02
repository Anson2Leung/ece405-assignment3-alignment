#!/usr/bin/env bash
# sft_experiment1.sh

set -euo pipefail
OUTPUT_ROOT="/home/ansonl32/koa_scratch/ECE405/assignment3/sft_experiment"

MODEL_ID="/home/ansonl32/koa_scratch/ECE405/assignment3/Qwen/Qwen2.5-Math-1.5B"
MATH_PATH="/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet"
PROMPT_FILE="/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/prompts/r1_zero.prompt"
WANDB_PROJECT="ece405_assignment3"

LR=2e-5
BATCH_SIZE=1
GRAD_ACCUM=8
N_STEPS=200 
EVAL_EVERY=25
MAX_SEQ_LEN=256
WARMUP_STEPS=20
SEED=42

POLICY_DEVICE="cuda:0"
VLLM_DEVICE="cuda:1"

#  Dataset sizes to sweep 
SIZES=(128 256 512 1024 -1)

for SIZE in "${SIZES[@]}"; do

    if [ "$SIZE" -eq -1 ]; then
        RUN_NAME="sft_full_lr${LR}"
        OUT_DIR="${OUTPUT_ROOT}/exp1_full"
    else
        RUN_NAME="sft_n${SIZE}_lr${LR}"
        OUT_DIR="${OUTPUT_ROOT}/exp1_n${SIZE}"
    fi

    echo "══════════════════════════════════════════════════════════════"
    echo " Starting run: max_samples=${SIZE}   output → ${OUT_DIR}"
    echo "══════════════════════════════════════════════════════════════"

    uv run python sft.py \
        --model_id        "${MODEL_ID}"     \
        --math_path       "${MATH_PATH}"    \
        --prompt_file     "${PROMPT_FILE}"  \
        --output_dir      "${OUT_DIR}"      \
        --n_sft_steps     "${N_STEPS}"      \
        --batch_size      "${BATCH_SIZE}"   \
        --grad_accum_steps "${GRAD_ACCUM}"  \
        --lr              "${LR}"           \
        --warmup_steps    "${WARMUP_STEPS}" \
        --max_seq_len     "${MAX_SEQ_LEN}"  \
        --max_samples     "${SIZE}"         \
        --eval_every      "${EVAL_EVERY}"   \
        --num_val_generate 200              \
        --num_to_log       8                \
        --policy_device   "${POLICY_DEVICE}" \
        --vllm_device     "${VLLM_DEVICE}"  \
        --seed            "${SEED}"         \
        --wandb_project   "${WANDB_PROJECT}" \
        --wandb_run_name  "${RUN_NAME}"

    echo "Run complete: ${RUN_NAME}"
    echo ""
done

echo "All Experiment 1 runs finished."
echo "Open W&B project '${WANDB_PROJECT}' and group by run name to compare curves."