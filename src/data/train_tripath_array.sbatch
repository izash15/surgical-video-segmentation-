#!/bin/bash
#SBATCH --job-name=tripath
#SBATCH --output=logs/tripath_%A_%a.out
#SBATCH --error=logs/tripath_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-12

set -euo pipefail

REPO=/home/rmosad/surgical-video-segmentation-
ROOT=/home/rmosad/DSAD_pped/set3
LIST=$ROOT/organs.txt
PY=/home/rmosad/.conda/envs/svs/bin/python

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
cd "$REPO/src"

mkdir -p $REPO/logs

# Pick organ for this task
ORGAN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")

# Optional skip
if [[ "$ORGAN" == "ss" ]]; then
  echo "Skipping organ: $ORGAN"
  exit 0
fi

echo "Organ: $ORGAN"
echo "GPU:"
nvidia-smi || true

cd $REPO

# Save dir per organ
SAVE_DIR="$REPO/experiments/set3UNet/UNet_256_${ORGAN}E80"
mkdir -p "$SAVE_DIR"

$PY src/training/train_EB.py \
  --data-root "$ROOT/$ORGAN" \
  --images-subdir images \
  --masks-subdir masks \
  --list-train "$ROOT/$ORGAN/splits/train.txt" \
  --list-val "$ROOT/$ORGAN/splits/val.txt" \
  --num-classes 2 \
  --img-size 256 256 \
  --save-dir "$SAVE_DIR" \
  --num-workers 4 \
  --epochs 80 \
  --lr 1e-4 \
  --weight-decay 1e-3
