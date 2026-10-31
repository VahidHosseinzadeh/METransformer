#!/bin/bash
# Moving MNIST next-frame prediction with METransformer on the MELSTM paper protocol.
#
#   bash run_mm.sh <arm> [model_seed]
#
# arms:
#   met         phase-correlation slot tracking (min template, frame pairs for t < 4)
#   tf          the plain transformer: the same network without a lift (one slot, no transport)
#   met_oracle  the dataset's true velocities: the ceiling of the lift (diagnostic)
#   met_pair    MELSTM's frame-pair velocities, continuity-assigned
#
# SMOKE=1 trains 300 steps of one epoch and scores small test sets (wandb offline).
# DATA_ROOT is the torchvision MNIST root (default ./data); NUM_WORKERS the data workers
# (default 4); EXTRA_ARGS is appended.
set -e
ARM=${1:?usage: bash run_mm.sh met|tf|met_oracle|met_pair [model_seed]}
MODEL_SEED=${2:-1}

case $ARM in
  met)        MODEL=(--lift motion --velocity track) ;;
  tf)         MODEL=(--lift none) ;;
  met_oracle) MODEL=(--lift motion --velocity oracle) ;;
  met_pair)   MODEL=(--lift motion --velocity pair) ;;
  *) echo "unknown arm: $ARM (met|tf|met_oracle|met_pair)"; exit 1 ;;
esac

SAVE_DIR=runs/${ARM}_ms${MODEL_SEED}
RUN=(--epochs 50 --wandb_mode online)
if [ -n "$SMOKE" ]; then
  SAVE_DIR=smoke/${ARM}_ms${MODEL_SEED}
  RUN=(--epochs 1 --max_train_samples 9600 --max_val_samples 1280 --n_test 1000
       --n_len_gen 200 --image_every 1 --wandb_mode offline)
fi

ARGS=(
  --root "${DATA_ROOT:-data}"
  --model_seed "$MODEL_SEED"
  --batch_size 32 --lr 1e-3 --final_lr 1e-5 --warmup_epochs 1 --grad_clip 1.0
  --num_workers "${NUM_WORKERS:-4}"
  --save_dir "$SAVE_DIR"
  --wandb_name "${ARM}_ms${MODEL_SEED}" --wandb_group "$ARM"
  "${MODEL[@]}" "${RUN[@]}"
)
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

echo ">>> $(date)  arm=$ARM  model_seed=$MODEL_SEED  save_dir=$SAVE_DIR  host=$(hostname)"
exec python -m moving_mnist.train "${ARGS[@]}"
