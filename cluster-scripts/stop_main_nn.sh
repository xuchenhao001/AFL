#!/bin/bash

source ../fabric-samples/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  
  ssh ${HostUser}@${addrIN[0]} "kill -9 \$(ps -ef|grep '[m]ain_nn.py'|awk '{print \$2}')"
done

