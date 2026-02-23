#!/bin/bash

# Usage: ./batch_preprocess.sh <root_dir>
# Example: ./batch_preprocess.sh ../DSAD

ROOT_DIR=/home/rmosad/DSAD/
OUT_ROOT=/home/rmosad/DSAD_pped/set3

# Loop through all subdirectories in the given root directory
for A in "$ROOT_DIR"/*/; do
  A_NAME=$(basename "$A")
  echo "Processing folder: $A_NAME"

  /home/rmosad/.conda/envs/svs/bin/python dsad_preprocess.py \
    --src-root "${ROOT_DIR}/${A_NAME}" \
    --train '1, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17, 19, 22, 23, 24, 25, 27, 28, 29, 30, 31' \
    --val '2,7,11,13,14,18,20,32' \
    --image-out "${OUT_ROOT}/${A_NAME}/images" \
    --mask-out "${OUT_ROOT}/${A_NAME}/masks" \
    --lists-out "${OUT_ROOT}/${A_NAME}/splits"

  echo "Finished processing ${A_NAME}"
  echo "----------------------------------------"
done