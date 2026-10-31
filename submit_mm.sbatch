#!/bin/bash
# One METransformer Moving MNIST run as a Slurm job. Submit from the repo root, from a shell
# with the environment active (the job inherits it):
#
#   source ~/.bashrc; conda activate torch2.2
#   sbatch --job-name=met_ms1 submit_mm.sbatch met 1
#   SMOKE=1 sbatch --job-name=met_smoke --time=01:00:00 submit_mm.sbatch met 1
#
# The job exits with Python's exit code, so sacct shows crashes as FAILED. A rerun with the
# same arguments resumes from the save dir's checkpoint.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%j.log

echo ">>> job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ">>> python $(which python)"
if [ -n "$RUN_TESTS" ]; then
  python -m pytest -q tests || { echo ">>> tests failed"; exit 1; }
fi
bash run_mm.sh "$@"
status=$?
echo ">>> exit status $status"
exit $status
