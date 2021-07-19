#!/bin/bash

model=$1
dataset=$2
is_iid=$3
dataset_train_size=$4

source ../fabric-network/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })

  ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_avg" "$model" "$dataset" "$is_iid" "$dataset_train_size"
done

