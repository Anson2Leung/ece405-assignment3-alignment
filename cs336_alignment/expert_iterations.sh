#!/usr/bin/env bash
# expert_iterations.sh

set -euo pipefail

MODEL_ID="/home/ansonl32/koa_scratch/ECE405/assignment3/Qwen/Qwen2.5-Math-1.5B"
MATH_PATH="/home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet"
PROMPT_FILE="/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/prompts/r1_zero.prompt"
WANDB_PROJECT="ece405_assignment3"

OUTPUT_DIR="/home/ansonl32/koa_scratch/ECE405/assignment3/sft_expert_experiment"
mkdir -p "${OUTPUT_DIR}"

# Sweep parameters
DB_SIZES=(512 1024 2048)
ROLLOUTS=(4 8)
EPOCHS=(1 3)

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment

for DB in "${DB_SIZES[@]}"; do
    for G in "${ROLLOUTS[@]}"; do
        for E in "${EPOCHS[@]}"; do
            RUN_NAME="expertIteration_db${DB}_G${G}_E${E}"
            echo "Running: ${RUN_NAME}"
            
            uv run python expert_iterations.py \
                --model_id "${MODEL_ID}" \
                --math_path "${MATH_PATH}" \
                --prompt_file "${PROMPT_FILE}" \
                --db_size "${DB}" \
                --num_rollouts "${G}" \
                --sft_epochs "${E}" \
                --n_ei_steps 5 \
                --vllm_mem 0.4 \
                --max_seq_len 1024 \
                --max_grad_norm 1.0 \
                --output_dir "${OUTPUT_DIR}" \
                --wandb_project "${WANDB_PROJECT}" \
                --wandb_run_name  "${RUN_NAME}"
        done
    done
done