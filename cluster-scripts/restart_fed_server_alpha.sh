#!/bin/bash
model=$1
dataset=$2
alpha=$3

source ../fabric-samples/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })

  ssh ${HostUser}@${addrIN[0]} "kill -9 \$(ps -ef|grep '[f]ed_server_alpha.py'|awk '{print \$2}')"
  ssh ${HostUser}@${addrIN[0]} "kill -9 \$(ps -ef|grep '[h]raftd'|awk '{print \$2}')"
  ssh ${HostUser}@${addrIN[0]} "(cd ~/EASC/federated-learning/; python3 -u fed_server_alpha.py --model=${model} --dataset=${dataset} --hyperpara=${alpha}) > ~/EASC/server_${addrIN[0]}.log 2>&1 &"
done
