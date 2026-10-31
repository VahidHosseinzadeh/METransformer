#!/bin/bash
# One METransformer Moving MNIST run on Bridges-2 (PSC), default allocation. Submit from the
# repo root, with data/MNIST/raw in place and .venv built by scripts/setup_bridges2.sh (PSC's
# PyTorch module plus wandb and pytest):
#
#   sbatch --job-name=met_ms1 submit_bridges2.sbatch met 1
#   SMOKE=1 RUN_TESTS=1 sbatch --job-name=met_smoke --time=00:30:00 submit_bridges2.sbatch met 1
#   sbatch --gres=gpu:h100-80:1 --time=08:00:00 --job-name=met_ms1 submit_bridges2.sbatch met 1
#
# V100-32 by default: one run takes 565 ms per step there, about 15 h for 50 epochs at 1 SU per
# GPU-hour, and keeps the GPU busy. An H100 hour costs 2 SU.
# Runs write to runs/ in the repo, under $HOME: the project's /ocean space is shared by the
# whole allocation and was full on 2026-10-31. The job exits with Python's exit code, and a
# rerun with the same arguments resumes from the save dir's checkpoint.

#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=20:00:00
#SBATCH --output=slurm_%x_%j.log

cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate || { echo ">>> no .venv in $SLURM_SUBMIT_DIR"; exit 1; }
echo ">>> job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ">>> $(python -V) at $(which python)"
if [ -n "$RUN_TESTS" ]; then
  python -m pytest -q tests || { echo ">>> tests failed"; exit 1; }
fi
bash run_mm.sh "$@"
status=$?
echo ">>> exit status $status"
exit $status
