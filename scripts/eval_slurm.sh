#!/bin/bash
# Run an eval script on one H100 of the main partition. Extra args go to the script (Hydra overrides):
#   sbatch scripts/eval_slurm.sh eval.py --config-name=pusht policy=lewm/pusht        # -> data/lewm/
#   sbatch scripts/eval_slurm.sh eval_multi.py env.n_agents=1 output.dir=multipusht/1a_single
#SBATCH --job-name=lewm-eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=data/outputs/slurm/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$SLURM_SUBMIT_DIR/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
srun .venv/bin/python "$@"
