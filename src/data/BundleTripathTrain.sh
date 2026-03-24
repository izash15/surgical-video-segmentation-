#!/usr/bin/env bash
set -euo pipefail

# --- User editable section ---

ROOT_DIR="/home/idavis3/DSAD_pped/set3"
REPO_DIR="/home/idavis3/surgical-video-segmentation-/src/"
PY="/home/idavis3/.conda/envs/svs/bin/python"

# List of folder names to skip
SKIP_LIST=("ss")

# --- End of user editable section ---

for A in "$ROOT_DIR"/*/; do
    A_NAME=$(basename "$A")

    # skip list check
    skip=false
    for SKIP_ITEM in "${SKIP_LIST[@]}"; do
        if [[ "$A_NAME" == "$SKIP_ITEM" ]]; then
            skip=true
            break
        fi
    done

    if [[ "$skip" == true ]]; then
        echo "Skipping folder: $A_NAME"
        echo "----------------------------------------"
        continue
    fi

    echo "Training on folder: $A_NAME"

    SAVE_DIR="${REPO_DIR}/experiments/set3UNet/UNet_256_${A_NAME}E80"
    mkdir -p "$SAVE_DIR"

    "$PY" "${REPO_DIR}/training/train_EB.py" \
        --data-root "${ROOT_DIR}/${A_NAME}" \
        --images-subdir images \
        --masks-subdir masks \
        --list-train "${ROOT_DIR}/${A_NAME}/splits/train.txt" \
        --list-val "${ROOT_DIR}/${A_NAME}/splits/val.txt" \
        --num-classes 2 \
        --img-size 256 256 \
        --save-dir "$SAVE_DIR" \
        --num-workers 2 \
        --epochs 80 \
        --lr 1e-4 \
        --weight-decay 1e-3

    echo "Finished training on ${A_NAME}"
    echo "---------------------"
done