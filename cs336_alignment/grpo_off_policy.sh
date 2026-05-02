#!/bin/bash
#SBATCH --job-name=grpo_off_policy
#SBATCH --partition=gpu,kill-shared
#SBATCH --gres=gpu:NV-A30:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/grpo_offpolicy_%A_%a.out
#SBATCH --error=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/grpo_offpolicy_%A_%a.err

cd /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment
export WANDB_API_KEY=""
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p outputs/logs

# sweep epoch and train batch size
# epoch 1 is not necessary as it was done in last experimentgrpo_std_norm
ROLLOUT_BATCH=64
EPOCHS_ARRAY=(2 2 4 4)
TRAIN_BATCH_ARRAY=(32 64 32 64) # to maintain microbatch = 2

EPOCH=${EPOCHS_ARRAY[$SLURM_ARRAY_TASK_ID]}
TRAIN_BATCH=${TRAIN_BATCH_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Keep micro_train_batch_size = 2 to fit in VRAM. 
# GRAD_ACCUM = TRAIN_BATCH / 2
GRAD_ACCUM=$((TRAIN_BATCH / 2))
LR=1e-5
STEPS=45 # 45 looked nice to the grad_norm and entropy graphs
LOSS_TYPE="grpo_clip"
LEN_NORM="--use-length-normalization"
STD_NORM="--no-use-std-normalization"


RUN_NAME="GRPO_offpolicy_ep${EPOCH}_tb${TRAIN_BATCH}_lr${LR}"
OUT_DIR="/home/ansonl32/koa_scratch/ECE405/assignment3/qwen_grpo_final/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}: EPOCHS=${EPOCH} TRAIN_BATCH=${TRAIN_BATCH} GRAD_ACCUM=${GRAD_ACCUM}"

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
    --epochs-per-rollout-batch    "${EPOCH}" \
    --train-batch-size            "${TRAIN_BATCH}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --gpu-memory-utilization      0.70 \
    --loss-type                   "${LOSS_TYPE}" \
    ${STD_NORM} \
    ${LEN_NORM} \
    --cliprange                   0.2 \
    --eval-interval               10 \
    --seed                        42 \
    --wandb-project               "ece405_assignment3" \
    --wandb-run-name              "${RUN_NAME}"

echo "Done: EPOCH=${EPOCH} TRAIN_BATCH=${TRAIN_BATCH}"