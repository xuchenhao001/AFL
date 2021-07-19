#!/bin/bash
user=$1
addr=$2
test_name=$3
model=$4
dataset=$5
is_iid=$6
dataset_train_size=$7
fade=$9


FIRST_CHAR=$(echo ${test_name} | cut -c1-1)
FOLLOWING_CHAR=$(echo ${test_name} | cut -c2-)
PYTHON_CMD="python3 -u ${test_name}.py"
PS_NAME="[${FIRST_CHAR}]${FOLLOWING_CHAR}.py"

if [[ "$is_iid" == true ]]; then
  PYTHON_CMD="$PYTHON_CMD --iid"
fi
if [[ ! -z "$model" ]]; then
  PYTHON_CMD="$PYTHON_CMD --model=$model"
fi
if [[ ! -z "$dataset" ]]; then
  PYTHON_CMD="$PYTHON_CMD --dataset=$dataset"
fi
if [[ ! -z "$dataset_train_size" ]]; then
  PYTHON_CMD="$PYTHON_CMD --dataset_train_size=$dataset_train_size"
fi
if [[ ! -z "$fade" ]]; then
  PYTHON_CMD="$PYTHON_CMD --fade=$fade"
fi

ssh ${user}@${addr} "kill -9 \$(ps -ef|grep '$PS_NAME'|awk '{print \$2}')"
ssh ${user}@${addr} "(cd $PWD/../federated-learning/; $PYTHON_CMD) > $PWD/../server_${addr}.log 2>&1 &"

