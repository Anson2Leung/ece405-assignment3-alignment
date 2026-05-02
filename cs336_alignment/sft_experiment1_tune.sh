#!/usr/bin/env bash
# sft_experiment1_tune.sh

set -euo pipefail
OUTPUT_ROOT="/home/ansonl32/koa_scratch/ECE405/assignment3/sft_tuning"

MODEL_ID="/home/ansonl32/koa_scratch/ECE405/assignment3/Qwen/Qwen2.5-Math-1.5B"
MATH_PATH="/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet"
PROMPT_FILE="/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/prompts/r1_zero.prompt"
WANDB_PROJECT="ece405_assignment3"

SIZE=-1
BATCH_SIZE=1
N_STEPS=200
EVAL_EVERY=50         
MAX_SEQ_LEN=256
WARMUP_STEPS=20
SEED=42

POLICY_DEVICE="cuda:0"
VLLM_DEVICE="cuda:1"

export WANDB_PROJECT="${WANDB_PROJECT}"

# learning rate to sweep
LEARNING_RATES=( 1e-5 2e-5 5e-5 )

# gradient accumulation steps to sweep
GRAD_ACCUM_STEPS=( 4 8 16 )

echo "Starting Hyperparameter Tuning on Full Dataset (SIZE=${SIZE})"
echo "Testing Learning Rates: ${LEARNING_RATES[*]}"
echo "Testing Gradient Accumulation Steps: ${GRAD_ACCUM_STEPS[*]}"

for LR in "${LEARNING_RATES[@]}"; do
    for GRAD_ACCUM in "${GRAD_ACCUM_STEPS[@]}"; do
    
        RUN_NAME="sft_full_lr${LR}_bs${GRAD_ACCUM}"
        OUT_DIR="${OUTPUT_ROOT}/lr${LR}_bs${GRAD_ACCUM}"

        echo "══════════════════════════════════════════════════════════════"
        echo " Starting run: LR=${LR}, Eff_Batch=${GRAD_ACCUM}   output → ${OUT_DIR}"
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

    done
done

echo "All tuning runs completed successfully!"