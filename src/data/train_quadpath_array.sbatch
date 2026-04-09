#!/bin/bash
#SBATCH --job-name=quadpath_stacked
#SBATCH --output=/home/rmosad/surgical-video-segmentation-/logs/quadpath_%A_%a.out
#SBATCH --error=/home/rmosad/surgical-video-segmentation-/logs/quadpath_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-12

set -euo pipefail

REPO=/home/rmosad/surgical-video-segmentation-
ROOT=/home/rmosad/DSAD_pped/set3
LIST=$ROOT/organs.txt
PY=/home/rmosad/.conda/envs/svs2/bin/python

mkdir -p "$REPO/logs"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/rmosad/.cache/huggingface
export TRANSFORMERS_CACHE=/home/rmosad/.cache/huggingface

cd "$REPO/src"

# Pick organ for this task
ORGAN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")

if [[ -z "${ORGAN:-}" ]]; then
  echo "No organ found for task id ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

# Optional skip
if [[ "$ORGAN" == "ss" ]]; then
  echo "Skipping organ: $ORGAN"
  exit 0
fi

echo "Organ: $ORGAN"
echo "Host: $(hostname)"
echo "Python: $PY"
echo "GPU:"
nvidia-smi || true

# sanity import
$PY - <<'PYCODE'
from mmcv.ops import ModulatedDeformConv2d
from models.ViT import MiTB0Encoder
from models.ViTStacked import QuadPathUNetStacked
print("mmcv + MiTB0Encoder + QuadPathUNetStacked import OK")
PYCODE

SAVE_DIR="$REPO/experiments/set3QuadPathStacked/QuadPathStacked_256_${ORGAN}E80"
mkdir -p "$SAVE_DIR"

$PY training/train_EB.py \
  --data-root "$ROOT/$ORGAN" \
  --images-subdir images \
  --masks-subdir masks \
  --list-train "$ROOT/$ORGAN/splits/train.txt" \
  --list-val "$ROOT/$ORGAN/splits/val.txt" \
  --num-classes 2 \
  --img-size 256 256 \
  --save-dir "$SAVE_DIR" \
  --batch-size 4 \
  --num-workers 0 \
  --epochs 80 \
  --lr 1e-4 \
  --weight-decay 1e-3