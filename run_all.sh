#!/usr/bin/env bash

DATA_PATH="/Users/dmitry/Downloads/курсач/timeeval_data"
COLLECTIONS=("NASA-SMAP" "NASA-MSL" "SMD" "SWaT" "WADI")
CONFIGS=("chronos" "moirai" "timesfm" "ttm")

for model in "${CONFIGS[@]}"; do
  for collection in "${COLLECTIONS[@]}"; do
    echo "====================================="
    echo "Running $model on $collection"
    echo "====================================="

    python src/run_model.py \
      --config "configs/${model}.yaml" \
      --collection "$collection" \
      --data-path "$DATA_PATH"
  done
done