#!/bin/bash

source ../fabric-network/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  scp ../fabric-network/network.config ${HostUser}@${addrIN[0]}:$PWD/../fabric-network/network.config
done

