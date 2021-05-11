#!/bin/bash

source ../fabric-network/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  
  ./stop_core.sh ${HostUser} ${addrIN[0]} "fed_localA"
done


