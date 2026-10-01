#!/bin/bash
# Train on the full A100 of the gpu partition. Extra args go to train.py (Hydra overrides):
#   sbatch scripts/train_slurm.sh data=multipusht output_model_name=lewm_mpt2a subdir=lewm_mpt2a
#SBATCH --job-name=lewm-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=14
# memory: measured peak 26 GB with 12 workers, prefetch 1, non-persistent workers (the peak is the
# train -> val switch; persistent workers keep both pools alive there and were OOM-killed at 24 GB).
# 28 GB was OOM-killed at the 20th switch (MaxRSS 27.4 GB), so leave ~10 GB of headroom
#SBATCH --mem=36G
#SBATCH --time=36:00:00
#SBATCH --output=data/outputs/slurm/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
# temp files (DataLoader worker dirs, wandb, torch compile) go in the repo, not /tmp: a full root
# disk made worker startup fail at the train -> val switch and hung the job
export TMPDIR="$SLURM_SUBMIT_DIR/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
# a crashed step (e.g. OOM-killed at the train -> val switch) is retried inside the same
# allocation; train.py resumes from the newest last.ckpt under checkpoints/<subdir>/spt, so at most one epoch is lost
for attempt in 1 2 3; do
    srun .venv/bin/python train.py trainer.devices=1 num_workers=${NUM_WORKERS:-12} loader.prefetch_factor=1 loader.persistent_workers=false "$@" && exit 0
    echo "attempt $attempt failed (exit $?), resuming from last.ckpt" >&2
done
exit 1
