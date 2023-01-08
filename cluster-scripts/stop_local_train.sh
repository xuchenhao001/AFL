#!/bin/bash

source ../fabric-network/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  
  ./stop_core.sh ${HostUser} ${addrIN[0]} "local_train" 
done


