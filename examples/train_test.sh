#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

CONFIG_FILE=${1-config.env}

set -o allexport
source $CONFIG_FILE
set +o allexport

if [ -f "$PARAMS_PATH" ]; then
  echo ">> $PARAMS_PATH already exists, skipping training"
else
  python examples/semmacape.py --train_limit $TRAIN_LIMIT --params_path $PARAMS_PATH \
    | tee -a $LOG_FILE
fi

python examples/test_semmacape.py \
  --test_limit $TEST_LIMIT \
  --threshold $THRESHOLD \
  --iou_threshold $IOU_THRESHOLD \
  --params_path $PARAMS_PATH \
  | tee -a $LOG_FILE
