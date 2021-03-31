#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

CONFIG_FILE=${1-./configs/config.env}

set -o allexport
source $CONFIG_FILE
set +o allexport

echo ">> Config: "
cat $CONFIG_FILE | tee -a $LOG_FILE
echo "THRESHOLD=$THRESHOLD" | tee -a $LOG_FILE

if [ -f "$PARAMS_PATH" ]; then
  echo ">> $PARAMS_PATH already exists, skipping training"
else
	# We want to use the Deep-SVDD version
	if [[ "${PADEEP-0}" == "1" ]]; then
		PYTHONPATH=deep_svdd/src/ python examples/padeep.py \
			--train_folder ./data/semmacape/ \
			--test_folder ./data/semmacape \
			--oe_folder ./data/coco/ \
			--oe_frequency $OE_FREQUENCY \
			--n_epochs $N_EPOCHS \
			--ae_n_epochs $AE_N_EPOCHS \
			--pretrain \
			--train_limit $TRAIN_LIMIT \
			--params_path $PARAMS_PATH \
			| tee -a $LOG_FILE
	else # Use the regular PaDiM version
		python examples/semmacape.py --train_limit $TRAIN_LIMIT --params_path $PARAMS_PATH $EXTRA_FLAGS \
			| tee -a $LOG_FILE
	fi
fi

python examples/test_semmacape.py \
  --test_limit $TEST_LIMIT \
  --threshold $THRESHOLD \
  --iou_threshold $IOU_THRESHOLD \
  --params_path $PARAMS_PATH \
  --min_area $MIN_AREA \
  --use_nms \
  ${EXTRA_FLAGS-} \
  | tee -a $LOG_FILE
