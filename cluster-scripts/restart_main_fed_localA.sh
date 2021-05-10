#!/bin/bash
model=$1
dataset=$2

source ../fabric-network/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })

  ssh ${HostUser}@${addrIN[0]} "kill -9 \$(ps -ef|grep '[m]ain_fed_localA.py'|awk '{print \$2}')"
  ssh ${HostUser}@${addrIN[0]} "(cd $PWD/../federated-learning/; python3 -u main_fed_localA.py --model=${model} --dataset=${dataset}) > $PWD/../server_${addrIN[0]}.log 2>&1 &"
done
