#!/bin/bash
set -e

SEEDS=(1)
DIAG_FLAGS=(--no-diag)

CATEGORIES=("pill")

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
        --mode "train" \
        --results_dirpath "${RESULTS_DIR}" \
        --data_path "${DATA_PATH}" \
        --categories "${CATEGORY}" \
        --backbone mobilenet_v2 \
        --ad_layers 7 10 13 \
        --variant low-rank \
        --device cuda:0 \
        --seeds "${SEED}" \
        --save_path "./outputs/padim.pt"

      echo "Finished category=${CATEGORY}, seed=${SEED}, diag_flag=${DIAG_FLAG}"
      echo "---------------------------------------------"

    done
  done
done
