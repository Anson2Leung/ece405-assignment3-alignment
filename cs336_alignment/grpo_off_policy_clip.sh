#!/bin/bash
#SBATCH --job-name=grpo_off_policy_clip
#SBATCH --partition=gpu,kill-shared
#SBATCH --gres=gpu:NV-A30:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/grpo_off_policy_clip_%A_%a.out
#SBATCH --error=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/grpo_off_policy_clip_%A_%a.err

cd /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment
export WANDB_API_KEY=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs

CLIP_TYPES=("grpo_clip" "grpo_no_clip")
CLIP_TYPE=${CLIP_TYPES[$SLURM_ARRAY_TASK_ID]}

LR=1e-5
STEPS=200
EPOCHS=4
ROLLOUT_BATCH=64
TRAIN_BATCH=32
GRAD_ACCUM=16
LEN_NORM="--use-length-normalization"
STD_NORM="--no-use-std-normalization"

RUN_NAME="GRPO_offpolicy_${CLIP_TYPE}_ep${EPOCHS}_tb${TRAIN_BATCH}"
OUT_DIR="/home/ansonl32/koa_scratch/ECE405/assignment3/qwen_grpo_final/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}: CLIP_TYPE=${CLIP_TYPE}  Epochs=${EPOCHS}"

uv run python grpo_train.py \
    --model-id                    /home/ansonl32/koa_scratch/ECE405/assignment3/Qwen/Qwen2.5-Math-1.5B \
    --data-path                   /home/ansonl32/ECE405/ece405-assignment3-alignment/data/math/math.parquet \
    --output-dir                  "${OUT_DIR}" \
    --n-grpo-steps                "${STEPS}" \
    --learning-rate               "${LR}" \
    --advantage-eps               1e-6 \
    --rollout-batch-size          "${ROLLOUT_BATCH}" \
    --group-size                  8 \
    --sampling-temperature        1.0 \
    --sampling-min-tokens         4 \
    --sampling-max-tokens         1024 \
    --epochs-per-rollout-batch    "${EPOCHS}" \
    --train-batch-size            "${TRAIN_BATCH}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --gpu-memory-utilization      0.70 \
    --loss-type                   "${CLIP_TYPE}" \
    ${STD_NORM} \
    ${LEN_NORM} \
    --cliprange                   0.2 \
    --eval-interval               10 \
    --seed                        42 \
    --wandb-project               "ece405_assignment3" \
    --wandb-run-name              "${RUN_NAME}"

echo "Done: Off Policy Clip=${CLIP_TYPE}"