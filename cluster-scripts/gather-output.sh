#!/bin/bash

source ../fabric-samples/network.config

rm -rf output/
mkdir -p output/

for i in "${!PeerAddress[@]}"; do
  addrIN=(${PeerAddress[i]//:/ })
  
  scp ${HostUser}@${addrIN[0]}:~/EASC/federated-learning/result-record_*.txt output/
done

