#!/bin/bash
set -e

CYAN='\033[0;36m'
NC='\033[0m'

DATASET_PATH="/home/fgenilotti/Downloads/anovox/Anovox_Sample/Anovox"

MODELS=(padim patchcore fastflow cfa dinomaly stfpm rd4ad ssnet)
BACKBONE="mobilenet_v2"

AD_LAYERS=(10 13 16)

SEEDS=(1)
BATCH_SIZES=(1)
REPEATS=3

IMG_SIZE=(224 224)

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for RUN in $(seq 1 $REPEATS); do

        echo -e "${CYAN}>> Run ${RUN}/${REPEATS} | model=${MODEL}, seed=${SEED}, batch=${BATCH_SIZE}${NC}"

        python main_scripts/profiler.py \
          --model "${MODEL}" \
          --dataset_path "${DATASET_PATH}" \
          --backbone_model_name "${BACKBONE}" \
          --ad_layers_idxs "${AD_LAYERS[@]}" \
          --device cpu \
          --seeds "${SEED}" \
          --batch_size "${BATCH_SIZE}" \
          --num_batches 1000 \
          --img_input_size ${IMG_SIZE[0]} ${IMG_SIZE[1]} \
          --save_path "/home/fgenilotti/Downloads/padim_mobile.pt"

        echo -e "${CYAN}>> Completed run ${RUN} for ${MODEL}${NC}"
        echo "--------------------------------------------------"

      done
    done
  done
done