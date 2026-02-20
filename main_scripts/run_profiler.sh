#!/bin/bash
set -e

CATEGORIES=(pill)
SEEDS=(1)

DIAG_FLAGS=(--diag --no-diag)

BATCH_SIZES=(1)
REPEATS=3

DATA_PATH="/home/fgenilotti/Downloads/mvtec"

for CATEGORY in "${CATEGORIES[@]}"; do
  for DIAG_FLAG in "${DIAG_FLAGS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for RUN in $(seq 1 $REPEATS); do

          # detect suffix from flag
          if [ "$DIAG_FLAG" = "--diag" ]; then
            SUFFIX="_diag"
          else
            SUFFIX=""
          fi

          echo "Run ${RUN}/${REPEATS} | cat=${CATEGORY}, seed=${SEED}, diag_flag=${DIAG_FLAG}, batch=${BATCH_SIZE}"

          MODEL_PATH="./outputs/mvtec/outputs_mobilenetv2_l4-7-10/patch_${CATEGORY}${SUFFIX}_s${SEED}.pt"

          if [ ! -f "$MODEL_PATH" ]; then
            echo "Model not found: $MODEL_PATH — skipping"
            continue
          fi

          python main_scripts/run_inference_profiler_cpu.py \
            --train \
            --test \
            --data_path "${DATA_PATH}" \
            --categories "${CATEGORY}" \
            --backbone mobilenet_v2 \
            --ad_layers 4 7 10 \
            --device cpu \
            --seeds "${SEED}" \
            --batch_size "${BATCH_SIZE}" \
            --save_path "${MODEL_PATH}"

          echo "Completed run ${RUN}"
          echo "----------------------------------------------------"

        done
      done
    done
  done
done