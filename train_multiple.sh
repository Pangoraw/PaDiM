#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

CONFIG_FILE=./configs/config_without_threshold.env

for THRESHOLD_VALUE in 3 4 5 6 7 8 9
do
	export THRESHOLD=0.$THRESHOLD_VALUE
	./examples/train_test.sh $CONFIG_FILE
done

