#!/bin/bash

PS_NAME="[n]ode ./bin/www"

kill -9 $(ps -ef|grep "$PS_NAME"|awk '{print $2}')

cd $PWD/../blockchain-server/
nohup npm start > $PWD/../blockchain-server/blockchain-server.log 2>&1 &
cd -

