#!/bin/bash

# set -x

source ./test.config
source ../fabric-network/network.config

function killOldProcesses() {
    # kill all old processes
    ./stop_fed_async.sh
    ./stop_fed_sync.sh
    ./stop_fed_localA.sh
    ./stop_local_train.sh
}

function cleanOutput() {
    # clean all old outputs
    ./clean-output.sh
}

function clean() {
    killOldProcesses
    cleanOutput
}

function arrangeOutput(){
    model=$1
    dataset=$2
    expname=$3
    ./gather-output.sh
    mkdir -p "${model}-${dataset}"
    mv output/ "${model}-${dataset}/${expname}"
}


function testFinish() {
    fileName=$1
    while : ; do
        count=$(ps -ef|grep ${fileName}|wc -l)
        if [[ $count -eq 0 ]]; then
            break
        fi
        echo "[`date`] Process still active, sleep 60 seconds"
        sleep 60
    done
}


function main() {
    for i in "${!TestSchema[@]}"; do
        schema=(${TestSchema[i]//-/ })
        model=${schema[0]}
        dataset=${schema[1]}
        is_iid=${IS_IID}
        echo "[`date`] ALL_NODE_TEST UNDER: ${model} - ${dataset}"

        # fed_async
        if [[ ! -d "${model}-${dataset}/fed_async" ]]; then
            echo "[`date`] ## fed_async start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_async" "$model" "$dataset" "$is_iid" "$dataset_train_size"
            done
            sleep 300
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_async.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_async"
            echo "[`date`] ## fed_async done ##"
        fi

        # fed_async_f05
        if [[ ! -d "${model}-${dataset}/fed_async_f05" ]]; then
            echo "[`date`] ## fed_async_f05 start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_async" "$model" "$dataset" "$is_iid" "$dataset_train_size" "0.5"
            done
            sleep 300
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_async.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_async_f05"
            echo "[`date`] ## fed_async_f05 done ##"
        fi

        # fed_async_f10
        if [[ ! -d "${model}-${dataset}/fed_async_f10" ]]; then
            echo "[`date`] ## fed_async_f10 start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_async" "$model" "$dataset" "$is_iid" "$dataset_train_size" "1.0"
            done
            sleep 300
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_async.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_async_f10"
            echo "[`date`] ## fed_async_f10 done ##"
        fi

        # fed_async_f15
        if [[ ! -d "${model}-${dataset}/fed_async_f15" ]]; then
            echo "[`date`] ## fed_async_f15 start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_async" "$model" "$dataset" "$is_iid" "$dataset_train_size" "1.5"
            done
            sleep 300
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_async.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_async_f15"
            echo "[`date`] ## fed_async_f15 done ##"
        fi
    done
}

main

