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
    ./stop_fed_befl.sh
    ./stop_fed_asofed.sh
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


function waitFinish() {
    fileName=$1
    sleep 1500
}


function main() {
    for i in "${!TestSchema[@]}"; do
        schema=(${TestSchema[i]//-/ })
        model=${schema[0]}
        dataset=${schema[1]}
        is_iid=${IS_IID}
        fade=${FADE}
        echo "[`date`] ALL_NODE_TEST UNDER: ${model} - ${dataset}"

        # fed_befl
        if [[ ! -d "${model}-${dataset}/fed_befl" ]]; then
            echo "[`date`] ## fed_befl start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_befl" "$model" "$dataset" "$is_iid" "$dataset_train_size"
            done
            sleep 300
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            waitFinish "[f]ed_befl.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_befl"
            # clean
            clean
            echo "[`date`] ## fed_befl.py done ##"
        fi

        # fed_asofed
        if [[ ! -d "${model}-${dataset}/fed_asofed" ]]; then
            echo "[`date`] ## fed_asofed start ##"
            # clean
            clean
            # run test
            for i in "${!PeerAddress[@]}"; do
              addrIN=(${PeerAddress[i]//:/ })
              dataset_train_size=${TrainDataSize[i]}
              ./restart_core.sh ${HostUser} ${addrIN[0]} "fed_asofed" "$model" "$dataset" "$is_iid" "$dataset_train_size"
            done
            sleep 300
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            waitFinish "[f]ed_asofed.py"
            # gather output, move to the right directory
            arrangeOutput ${model} ${dataset} "fed_asofed"
            # clean
            clean
            echo "[`date`] ## fed_asofed done ##"
        fi

    done
}

main > full_test.log 2>&1 &

