#!/bin/bash
#SBATCH --job-name=expert_iterations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NV-A30:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/ei_job_%j.out
#SBATCH --error=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/ei_job_%j.err

mkdir -p /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs

export WANDB_API_KEY=""

echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
 
cd /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment
 
echo "=========================================================="
echo "Executing expert_iterations.sh..."
echo "=========================================================="
./expert_iterations.sh
 
echo "Expert iteration experiment completed successfully."