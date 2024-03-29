#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mail-type=FAIL,END

if [[ ! -z ${SLURM_JOBID+z} ]];
then
	echo "Setting up SLURM env"
	setcuda 10.2
	conda activate python37
else
	echo "Not a SLURM job"
fi

set -o errexit
set -o pipefail
set -o nounset

CONFIG_FILE=${1-./configs/config.env}

set -o allexport
source $CONFIG_FILE
set +o allexport

echo ">> Config: "
cat $CONFIG_FILE | tee -a $LOG_FILE

TRAIN_FOLDER=${TRAIN_FOLDER-./data/semmacape/416_empty/}
TEST_FOLDER=${TEST_FOLDER-/share/projects/semmacape/Data_Semmacape_2/416_non_empty_filtered/}

echo "Starting script"
echo $(date)

python examples/main.py \
  --train_folder $TRAIN_FOLDER \
  --test_folder $TEST_FOLDER \
  --train_limit $TRAIN_LIMIT \
  --test_limit $TEST_LIMIT \
  --params_path $PARAMS_PATH \
  --oe_folder ./data/coco/ \
  --oe_frequency ${OE_FREQUENCY-2} \
  --n_epochs ${N_EPOCHS-0} \
  --ae_n_epochs ${AE_N_EPOCHS-0} \
  --pretrain \
  --n_svdds ${N_SVDDS-1} \
  --iou_threshold $IOU_THRESHOLD \
  --min_area $MIN_AREA \
  --use_nms \
  ${EXTRA_FLAGS-} \
  | tee -a $LOG_FILE

echo $(date)
