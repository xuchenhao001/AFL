#!/bin/bash

# set -x

source ./test.config
source ../fabric-network/network.config

function killOldProcesses() {
    # kill all old processes
    ./stop_fed_async.sh
    ./stop_fed_sync.sh
    ./stop_fed_avg.sh
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
        fade=${FADE}
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

        # fed_sync
        if [[ ! -d "${model}-${dataset}/fed_sync" ]]; then
            echo "[`date`] ## fed_sync start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_sync" "$model" "$dataset" "$is_iid" "$dataset_train_size"
            done
            sleep 300
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_sync.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_sync"
            echo "[`date`] ## fed_sync done ##"
        fi

        # fed_avg
        if [[ ! -d "${model}-${dataset}/fed_avg" ]]; then
            echo "[`date`] ## fed_avg start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_avg" "$model" "$dataset" "$is_iid" "$dataset_train_size"
            done
            sleep 300
            # detect test finish or not
            testFinish "[f]ed_avg.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_avg"
            echo "[`date`] ## fed_avg done ##"
        fi

        # fed_localA
        if [[ ! -d "${model}-${dataset}/fed_localA" ]]; then
            echo "[`date`] ## fed_localA start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_localA" "$model" "$dataset" "$is_iid" "$dataset_train_size"
            done
            sleep 300
            # detect test finish or not
            testFinish "[f]ed_localA.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_localA"
            echo "[`date`] ## fed_localA done ##"
        fi

        # local_train
        if [[ ! -d "${model}-${dataset}/local_train" ]]; then
            echo "[`date`] ## local_train start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "local_train" "$model" "$dataset" "$is_iid" "$dataset_train_size"
            done
            sleep 300
            # detect test finish or not
            testFinish "[l]ocal_train.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "local_train"
            echo "[`date`] ## local_train done ##"
        fi

    done
}

main


