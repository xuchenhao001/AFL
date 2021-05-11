#!/bin/bash

# set -x

source ./test.config

function killOldProcesses() {
    # kill all old processes
    ./stop_fed_async.sh
    ./stop_fed_sync.sh
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
        echo "[`date`] Process still active, sleep 300 seconds"
        sleep 300
    done
    sleep 60  # wait 60 seconds to make sure all nodes are finished.
}

function main() {
    for i in "${!TestSchema[@]}"; do
        schema=(${TestSchema[i]//-/ })
        echo "[`date`] ALL_NODE_TEST UNDER: ${schema[0]} - ${schema[1]}"

        # fed_async
        if [[ ! -d "${schema[0]}-${schema[1]}/fed_async" ]]; then
            echo "[`date`] ## fed_async start ##"
            # clean
            clean
            # run test
            ./restart_fed_async.sh ${schema[0]} ${schema[1]}
            sleep 180
            # detect test finish or not
            testFinish "[f]ed_async.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "fed_async"
            echo "[`date`] ## fed_async done ##"
        fi
        

        # fed_sync
        if [[ ! -d "${schema[0]}-${schema[1]}/fed_sync" ]]; then
            echo "[`date`] ## fed_sync start ##"
            # clean
            clean
            # run test
            ./restart_fed_sync.sh ${schema[0]} ${schema[1]}
            sleep 180
            # detect test finish or not
            testFinish "[f]ed_sync.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "fed_sync"
            echo "[`date`] ## fed_sync done ##"
        fi
        

        # local_train
        if [[ ! -d "${schema[0]}-${schema[1]}/local_train" ]]; then
            echo "[`date`] ## local_train start ##"
            # clean
            clean
            # run test
            ./restart_local_train.sh ${schema[0]} ${schema[1]}
            sleep 180
            # detect test finish or not
            testFinish "[l]ocal_train.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "local_train"
            echo "[`date`] ## local_train done ##"
        fi

    done
}

main


