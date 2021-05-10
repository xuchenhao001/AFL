#!/bin/bash

source ../fabric-samples/network.config

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  scp ../fabric-samples/network.config ${HostUser}@${addrIN[0]}:~/EASC/fabric-samples/network.config
done

