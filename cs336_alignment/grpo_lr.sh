#!/bin/bash
#SBATCH --job-name=grpo_lr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NV-A30:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/grpo_lr_%A_%a.out
#SBATCH --error=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/grpo_lr_%A_%a.err

cd /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment
export WANDB_API_KEY=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LRS=(1e-6 5e-6 1e-5 5e-5 1e-4)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="/home/ansonl32/koa_scratch/ECE405/assignment3/qwen_grpo_final/grpo_lr${LR}"
RUN_NAME="GRPO_lr_${LR}"
mkdir -p "${OUT_DIR}"

echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}: GRPO learning_rate=${LR}"

ROLLOUT_BATCH=64    # Reduced from 256
TRAIN_BATCH=64      # Reduced from 256
GRAD_ACCUM=32       # Reduced from 128

uv run python grpo_train.py \
    --model-id                    /home/ansonl32/koa_scratch/ECE405/assignment3/Qwen/Qwen2.5-Math-1.5B \
    --data-path                   /home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet \
    --output-dir                  "${OUT_DIR}" \
    --n-grpo-steps                200 \
    --learning-rate               "${LR}" \
    --advantage-eps               1e-6 \
    --rollout-batch-size          "${ROLLOUT_BATCH}" \
    --group-size                  8 \
    --sampling-temperature        1.0 \
    --sampling-min-tokens         4 \
    --sampling-max-tokens         1024 \
    --epochs-per-rollout-batch    1 \
    --train-batch-size            "${TRAIN_BATCH}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --gpu-memory-utilization      0.70 \
    --loss-type                   reinforce_with_baseline \
    --use-std-normalization       \
    --cliprange                   0.2 \
    --eval-interval               10 \
    --seed                        42 \
    --wandb-project "ece405_assignment3" \
    --wandb-run-name  "${RUN_NAME}"

echo "Done: GRPO lr=${LR}"