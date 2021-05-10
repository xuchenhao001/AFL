#!/bin/bash

source ../fabric-samples/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  
  ssh ${HostUser}@${addrIN[0]} "kill -9 \$(ps -ef|grep '[f]ed_server.py'|awk '{print \$2}')"
  ssh ${HostUser}@${addrIN[0]} "kill -9 \$(ps -ef|grep '[h]raftd'|awk '{print \$2}')"
done

