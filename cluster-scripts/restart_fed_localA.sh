#!/bin/bash
model=$1
dataset=$2

source ../fabric-network/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })

  ssh ${HostUser}@${addrIN[0]} "kill -9 \$(ps -ef|grep '[f]ed_localA.py'|awk '{print \$2}')"
  #ssh ${HostUser}@${addrIN[0]} "(cd $PWD/../federated-learning/; python3 -u fed_localA.py --iid --model=${model} --dataset=${dataset}) > $PWD/../server_${addrIN[0]}.log 2>&1 &"
  ssh ${HostUser}@${addrIN[0]} "(cd $PWD/../federated-learning/; python3 -u fed_localA.py --model=${model} --dataset=${dataset}) > $PWD/../server_${addrIN[0]}.log 2>&1 &"
done

