#!/bin/bash
# Pack several GPU runs into one allocation: every line of a jobs file is one python command
# (script + Hydra overrides), and they all run at once on the same GPU (an eval uses a few GB).
#   sbatch scripts/parallel_slurm.sh jobs/1a_parity.txt
# Each line's output goes to data/outputs/slurm/<job id>_<line no>.log
#SBATCH --job-name=lewm-par
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=data/outputs/slurm/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
export TMPDIR="$SLURM_SUBMIT_DIR/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
n=0
pids=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    n=$((n + 1))
    log="data/outputs/slurm/${SLURM_JOB_ID}_${n}.log"
    echo "[$n] $line -> $log"
    eval ".venv/bin/python $line" > "$log" 2>&1 &
    pids+=($!)
done < "$1"
fail=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "[$((i + 1))] failed" >&2; fail=1; }
done
exit $fail
