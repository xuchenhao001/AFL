#!/bin/bash

# set -x

source ./test.config

function killOldProcesses() {
    # kill all old processes
    ./stop_fed_server_alpha.sh
    ./stop_fed_server.sh
    ./stop_main_fed_localA.sh
    ./stop_main_fed.sh
    ./stop_main_nn.sh
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

        # 1. fed_server
        if [[ ! -d "${schema[0]}-${schema[1]}/fed_server" ]]; then
            echo "[`date`] ## fed_server start ##"
            # clean
            clean
            # run test
            ./restart_fed_server.sh ${schema[0]} ${schema[1]}
            sleep 60
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_server.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "fed_server"
            echo "[`date`] ## fed_server done ##"
        fi

        # 2. fed_server_alpha
        echo "[`date`] ## fed_server_alpha start ##"

        if [[ ! -d "${schema[0]}-${schema[1]}/fed_server_alpha_025" ]]; then
            echo "[`date`] ## fed_server_alpha_025 start ##"
            # clean
            clean
            # run test
            ./restart_fed_server_alpha.sh ${schema[0]} ${schema[1]} "0.25"
            sleep 180
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_server_alpha.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "fed_server_alpha_025"
            echo "[`date`] # fed_server_alpha_025 done #"
        fi

        if [[ ! -d "${schema[0]}-${schema[1]}/fed_server_alpha_050" ]]; then
            echo "[`date`] ## fed_server_alpha_050 start ##"
            # clean
            clean
            # run test
            ./restart_fed_server_alpha.sh ${schema[0]} ${schema[1]} "0.5"
            sleep 180
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_server_alpha.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "fed_server_alpha_050"
            echo "[`date`] # fed_server_alpha_050 done #"
        fi
        
        if [[ ! -d "${schema[0]}-${schema[1]}/fed_server_alpha_075" ]]; then
            echo "[`date`] ## fed_server_alpha_075 start ##"
            # clean
            clean
            # run test
            ./restart_fed_server_alpha.sh ${schema[0]} ${schema[1]} "0.75"
            sleep 180
            curl -i -X GET 'http://localhost:8888/messages'
            # detect test finish or not
            testFinish "[f]ed_server_alpha.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "fed_server_alpha_075"
            echo "[`date`] # fed_server_alpha_075 done #"
        fi
        
        echo "[`date`] ## fed_server_alpha done ##"

        # 3. main_fed_localA
        if [[ ! -d "${schema[0]}-${schema[1]}/main_fed_localA" ]]; then
            echo "[`date`] ## main_fed_localA start ##"
            # clean
            clean
            # run test
            ./restart_main_fed_localA.sh ${schema[0]} ${schema[1]}
            sleep 180
            # detect test finish or not
            testFinish "[m]ain_fed_localA.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "main_fed_localA"
            echo "[`date`] ## main_fed_localA done ##"
        fi
        

        # 4. main_fed
        if [[ ! -d "${schema[0]}-${schema[1]}/main_fed" ]]; then
            echo "[`date`] ## main_fed start ##"
            # clean
            clean
            # run test
            ./restart_main_fed.sh ${schema[0]} ${schema[1]}
            sleep 180
            # detect test finish or not
            testFinish "[m]ain_fed.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "main_fed"
            echo "[`date`] ## main_fed done ##"
        fi
        

        # 4. main_nn
        if [[ ! -d "${schema[0]}-${schema[1]}/main_nn" ]]; then
            echo "[`date`] ## main_nn start ##"
            # clean
            clean
            # run test
            ./restart_main_nn.sh ${schema[0]} ${schema[1]}
            sleep 180
            # detect test finish or not
            testFinish "[m]ain_nn.py"
            # gather output, move to the right directory
            arrangeOutput ${schema[0]} ${schema[1]} "main_nn"
            echo "[`date`] ## main_nn done ##"
        fi

    done
}

main

