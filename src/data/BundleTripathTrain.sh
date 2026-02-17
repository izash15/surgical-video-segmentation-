#!/usr/bin/env bash

# --- User editable section ---

# Usage: ./batch_train_tripath.sh <root_dir>
# Example: ./batch_train_tripath.sh ../DSAD_pped

ROOT_DIR=../DSAD_pped/set3/

# EB: Set9 Anno: Set1 
# List of folder names to skip (space-separated)
# Example: SKIP_LIST=("ureter" "fat" "spleen")
SKIP_LIST=("ss")

# --- End of user editable section ---

# Loop through all subdirectories in the given root directory
for A in "$ROOT_DIR"/*/; do
    # Remove trailing slash and get the folder name only
    A_NAME=$(basename "$A")

    # Check if folder name is in SKIP_LIST
    skip=false
    for SKIP_ITEM in "${SKIP_LIST[@]}"; do
        if [[ "$A_NAME" == "$SKIP_ITEM" ]]; then
            skip=true
            break
        fi
    done

    if [ "$skip" = true ]; then
        echo "Skipping folder: $A_NAME"
        echo "----------------------------------------"
        continue
    fi

    echo "Training on folder: $A_NAME"

    python train_EB.py \
        --data-root "${ROOT_DIR}/${A_NAME}" \
        --images-subdir images \
        --masks-subdir masks \
        --list-train "${ROOT_DIR}/${A_NAME}/splits/train.txt" \
        --list-val "${ROOT_DIR}/${A_NAME}/splits/val.txt" \
        --num-classes 2 \
        --img-size 256 256 \
        --save-dir "../experiments/set3UNet/UNet_256_${A_NAME}E80" \
        --num-workers 2 \
        --epochs 80 \
        --lr 1e-4 \
        --weight-decay 1e-3

    echo "Finished training on ${A_NAME}"
    echo "---------------------"
done