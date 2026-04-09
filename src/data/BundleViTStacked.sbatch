#!/bin/bash
#SBATCH --job-name=quad_aw
#SBATCH --output=/home/idavis3/surgical-video-segmentation-/logs/quad_aw_%j.out
#SBATCH --error=/home/idavis3/surgical-video-segmentation-/logs/quad_aw_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

REPO=/home/idavis3/surgical-video-segmentation-
ROOT=/home/idavis3/DSAD_pped/set3
ORGAN=abdominal_wall
PY=/home/idavis3/miniconda3/envs/svs/bin/python

mkdir -p "$REPO/logs"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
cd "$REPO/src"

echo "============================================"
echo "Job:    $SLURM_JOB_ID"
echo "Organ:  $ORGAN"
echo "Host:   $(hostname)"
echo "Time:   $(date)"
echo "============================================"

echo "--- GPU Info ---"
nvidia-smi || true

echo "--- Python/Torch ---"
$PY -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"

echo "--- Sanity Imports ---"
$PY -c "
from mmcv.ops import ModulatedDeformConv2d
print('mmcv OK')
from models.ViTStacked import QuadPathUNetStacked
print('QuadPathUNetStacked import OK')
from models.ViT import MiTB0Encoder
print('MiTB0Encoder import OK')
import torch
m = QuadPathUNetStacked(in_ch=3, num_classes=2, base_ch=32).cuda()
x = torch.randn(1, 3, 256, 256).cuda()
out = m(x)
print(f'Smoke test PASSED — output shape: {out.shape}')
del m, x, out
torch.cuda.empty_cache()
"

echo "--- Starting Training ---"

SAVE_DIR="$REPO/experiments/set3QuadPath/QuadPath_256_${ORGAN}_E80"
mkdir -p "$SAVE_DIR"

$PY training/train_EB.py \
  --data-root "$ROOT/$ORGAN" \
  --images-subdir images \
  --masks-subdir masks \
  --list-train "$ROOT/$ORGAN/splits/train.txt" \
  --list-val   "$ROOT/$ORGAN/splits/val.txt" \
  --num-classes 2 \
  --img-size 256 256 \
  --save-dir "$SAVE_DIR" \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 80 \
  --lr 1e-4 \
  --weight-decay 1e-3

echo "--- Done ---"
echo "Time: $(date)"