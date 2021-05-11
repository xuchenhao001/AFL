#!/bin/bash
model=$1
dataset=$2
is_iid=$3
dataset_train_size=$4
dataset_test_size=$5


source ../fabric-network/network.config


PYTHON_CMD="python3 -u fed_async.py"
PS_NAME="[f]ed_async.py"

if [[ "$is_iid" == true ]]; then
  PYTHON_CMD="$PYTHON_CMD --is_iid"
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
if [[ ! -z "$dataset_test_size" ]]; then
  PYTHON_CMD="$PYTHON_CMD --dataset_test_size=$dataset_test_size"
fi
echo "$PYTHON_CMD"

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })

  ssh ${HostUser}@${addrIN[0]} "kill -9 \$(ps -ef|grep '$PS_NAME'|awk '{print \$2}')"
  ssh ${HostUser}@${addrIN[0]} "(cd $PWD/../federated-learning/; $PYTHON_CMD) > $PWD/../server_${addrIN[0]}.log 2>&1 &"
done

