#!/bin/bash
set -e

DATASET_PATH="/home/fgenilotti/Downloads/mvtec"

MODELS=(padim patchcore fastflow cfa dinomaly stfpm rd4ad ssnet)
BACKBONE="mobilenet_v2"

AD_LAYERS=(4 7 10)

SEEDS=(1)
BATCH_SIZES=(1)
REPEATS=3

IMG_SIZE="224 224"

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for RUN in $(seq 1 $REPEATS); do

        echo "Run ${RUN}/${REPEATS} | model=${MODEL}, seed=${SEED}, batch=${BATCH_SIZE}"

        python main_scripts/profiler.py \
          --model "${MODEL}" \
          --dataset_path "${DATASET_PATH}" \
          --backbone_model_name "${BACKBONE}" \
          --ad_layers_idxs "${AD_LAYERS[@]}" \
          --device cpu \
          --seeds "${SEED}" \
          --batch_size "${BATCH_SIZE}" \
          --num_batches 1 \
          --img_input_size ${IMG_SIZE} \
          --model_path "./checkpoints/${MODEL}.pt"

        echo "Completed run ${RUN} for ${MODEL}"
        echo "----------------------------------------------------"

      done
    done
  done
done