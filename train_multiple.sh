#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

CONFIG_FILE=config_without_threshold.env

for THRESHOLD_VALUE in 0.3 0.5 0.6 0.7
do
	export THRESHOLD=$THRESHOLD_VALUE
	./examples/train_test.sh $CONFIG_FILE
done

