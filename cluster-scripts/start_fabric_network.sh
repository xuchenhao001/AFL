#!/bin/bash

# set -x

cd $PWD/../fabric-network/
./network.sh up createChannel && ./network.sh deployCC
cd -
