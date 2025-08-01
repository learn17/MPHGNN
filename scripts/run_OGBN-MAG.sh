#!/bin/bash

SEED=1
GPU=0
DATASET=mag
METHOD=MPHGNN
USE_NRL=False 
TRAIN_STRATEGY=common
USE_INPUT=True
ALL_FEAT=True 
INPUT_DROP_RATE=0.0
DROP_RATE=0.5 
HIDDEN_SIZE=512
SQUASH_K=5
EPOCHS=200
MAX_PATIENCE=30
EMBEDDING_SIZE=512
USE_LABEL=False
EVEN_ODD="all"

STDDEV_VALUES=(0.1 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 1.9 2.1 2.3 2.5 2.7 2.9 3.1 3.3 3.5)
ROUNDS_VALUES=(1 2 3 4 5 6 7 8 9 10 11 12)

for STDDEV in "${STDDEV_VALUES[@]}"; do
  for R in "${ROUNDS_VALUES[@]}"; do
    RPS="circulant_${STDDEV}_${R}"
    echo "Running with RPS = ${RPS}"

    python -u main.py \
      --method ${METHOD} \
      --dataset ${DATASET} \
      --use_nrl ${USE_NRL} \
      --use_label ${USE_LABEL} \
      --even_odd ${EVEN_ODD} \
      --train_strategy ${TRAIN_STRATEGY} \
      --use_input ${USE_INPUT} \
      --input_drop_rate ${INPUT_DROP_RATE} \
      --drop_rate ${DROP_RATE} \
      --hidden_size ${HIDDEN_SIZE} \
      --squash_k ${SQUASH_K} \
      --num_epochs ${EPOCHS} \
      --max_patience ${MAX_PATIENCE} \
      --embedding_size ${EMBEDDING_SIZE} \
      --use_all_feat ${ALL_FEAT} \
      --output_dir outputs/${DATASET}/seed${SEED}/circulant_${STDDEV}_${R}/ \
      --seed ${SEED} \
      --gpus ${GPU} \
      --rps ${RPS}
  done
done