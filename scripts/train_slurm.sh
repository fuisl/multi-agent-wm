#!/bin/bash
# Train on the full A100 of the gpu partition. Extra args go to train.py (Hydra overrides):
#   sbatch scripts/train_slurm.sh data=multipusht output_model_name=lewm_mpt2a subdir=lewm_mpt2a
#SBATCH --job-name=lewm-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=14
# memory: measured peak 20.5 GB with 12 workers, prefetch 1
#SBATCH --mem=24G
#SBATCH --time=36:00:00
#SBATCH --output=data/outputs/slurm/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
srun .venv/bin/python train.py trainer.devices=1 num_workers=12 loader.prefetch_factor=1 "$@"
