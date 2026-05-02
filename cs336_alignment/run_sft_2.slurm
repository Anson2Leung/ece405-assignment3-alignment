#!/bin/bash
#SBATCH --job-name=sft_experiment1_tune
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:NV-A30:2
#SBATCH --time=12:00:00
#SBATCH --output=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/sft_tune_job_%j.out
#SBATCH --error=/home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs/sft_tune_job_%j.err
 
mkdir -p /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment/outputs/logs

echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export WANDB_API_KEY="" 

cd /home/ansonl32/ECE405/ece405-assignment3-alignment/cs336_alignment

# Run Experiment 1 tuning
echo "=========================================================="
echo "Executing sft_experiment1_tune.sh..."
echo "=========================================================="
./sft_experiment1_tune.sh

echo "Completed successfully."