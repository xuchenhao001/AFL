#!/bin/bash

source ../fabric-samples/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  
  ssh ${HostUser}@${addrIN[0]} "cd ~/EASC/ && git pull"
done

