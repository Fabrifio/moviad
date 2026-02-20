#!/bin/bash
set -e

SEEDS=(1)
DIAG_FLAGS=(--no-diag)

CATEGORIES=(pill)

RESULTS_DIR="/home/fgenilotti/Desktop/Workspace/Research/vad-edge/moviad/outputs"
DATA_PATH="/home/fgenilotti/Downloads/mvtec"

for CATEGORY in "${CATEGORIES[@]}"; do
  for DIAG_FLAG in "${DIAG_FLAGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      # filename suffix
      if [ "$DIAG_FLAG" = "--diag" ]; then
        SUFFIX="_diag"
      else
        SUFFIX=""
      fi

      echo "Running category=${CATEGORY}, seed=${SEED}, diag_flag=${DIAG_FLAG}..."

      python main_scripts/main_padim.py \
        --train \
        --test \
        --results_dirpath "${RESULTS_DIR}" \
        --data_path "${DATA_PATH}" \
        --categories "${CATEGORY}" \
        --backbone mobilenet_v2 \
        --ad_layers 10 13 16 \
        --device cuda:0 \
        --seeds "${SEED}" \
        ${DIAG_FLAG} \
        --save_path "./outputs/patch_${CATEGORY}${SUFFIX}_s${SEED}.pt"

      echo "Finished category=${CATEGORY}, seed=${SEED}, diag_flag=${DIAG_FLAG}"
      echo "---------------------------------------------"

    done
  done
done
