#!/bin/bash

source ./network.config

CHANNEL_NAME="$1"
CC_SRC_LANGUAGE="$2"
VERSION="$3"
DELAY="$4"
MAX_RETRY="$5"
VERBOSE="$6"
: ${CHANNEL_NAME:="mychannel"}
: ${CC_SRC_LANGUAGE:="golang"}
: ${VERSION:="1"}
: ${DELAY:="3"}
: ${MAX_RETRY:="5"}
: ${VERBOSE:="false"}
CC_SRC_LANGUAGE=`echo "$CC_SRC_LANGUAGE" | tr [:upper:] [:lower:]`

if [ "$CC_SRC_LANGUAGE" = "go" -o "$CC_SRC_LANGUAGE" = "golang" ] ; then
  CC_RUNTIME_LANGUAGE=golang
  CC_SRC_PATH="./chaincode/fabcar/go/"

  echo Vendoring Go dependencies ...
  pushd ./chaincode/fabcar/go
  GO111MODULE=on go mod vendor
  popd
  echo Finished vendoring Go dependencies

elif [ "$CC_SRC_LANGUAGE" = "javascript" ]; then
  CC_RUNTIME_LANGUAGE=node # chaincode runtime language is node.js
  CC_SRC_PATH="../chaincode/fabcar/javascript/"

elif [ "$CC_SRC_LANGUAGE" = "java" ]; then
  CC_RUNTIME_LANGUAGE=java
  CC_SRC_PATH="../chaincode/fabcar/java/build/install/fabcar"

  echo Compiling Java code ...
  pushd ../chaincode/fabcar/java
  ./gradlew installDist
  popd
  echo Finished compiling Java code

elif [ "$CC_SRC_LANGUAGE" = "typescript" ]; then
  CC_RUNTIME_LANGUAGE=node # chaincode runtime language is node.js
  CC_SRC_PATH="../chaincode/fabcar/typescript/"

  echo Compiling TypeScript code into JavaScript ...
  pushd ../chaincode/fabcar/typescript
  npm install
  npm run build
  popd
  echo Finished compiling TypeScript code into JavaScript

else
  echo The chaincode language ${CC_SRC_LANGUAGE} is not supported by this script
  echo Supported chaincode languages are: go, java, javascript, and typescript
  exit 1
fi

# import utils
. scripts/envVar.sh

function packageChaincode() {
  ORG=$1
  setGlobals $ORG
  set -x
  peer lifecycle chaincode package fabcar.tar.gz --path ${CC_SRC_PATH} --lang ${CC_RUNTIME_LANGUAGE} --label fabcar_${VERSION} >&log.txt
  res=$?
  set +x
  cat log.txt
  verifyResult $res "Chaincode packaging on peer0.org${ORG} has failed"
  echo "===================== Chaincode is packaged on peer0.org${ORG} ===================== "
  echo
}

# installChaincode PEER ORG
function installChaincode() {
  ORG=$1
  setGlobals $ORG
  set -x
  peer lifecycle chaincode install fabcar.tar.gz >&log.txt
  res=$?
  set +x
  cat log.txt
  verifyResult $res "Chaincode installation on peer0.org${ORG} has failed"
  echo "===================== Chaincode is installed on peer0.org${ORG} ===================== "
  echo
}

# queryInstalled PEER ORG
function queryInstalled() {
  ORG=$1
  setGlobals $ORG
  set -x
  peer lifecycle chaincode queryinstalled >&log.txt
  res=$?
  set +x
  cat log.txt
  PACKAGE_ID=$(sed -n "/fabcar_${VERSION}/{s/^Package ID: //; s/, Label:.*$//; p;}" log.txt)
  verifyResult $res "Query installed on peer0.org${ORG} has failed"
  echo "===================== Query installed successful on peer0.org${ORG} on channel ===================== "
  echo
}

# approveForMyOrg VERSION PEER ORG
function approveForMyOrg() {
  ORG=$1
  setGlobals $ORG
  set -x
  peer lifecycle chaincode approveformyorg --signature-policy $SIGNATURE_POLICY -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile $ORDERER_CA --channelID $CHANNEL_NAME --name fabcar --version ${VERSION} --init-required --package-id ${PACKAGE_ID} --sequence ${VERSION} >&log.txt
  set +x
  cat log.txt
  verifyResult $res "Chaincode definition approved on peer0.org${ORG} on channel '$CHANNEL_NAME' failed"
  echo "===================== Chaincode definition approved on peer0.org${ORG} on channel '$CHANNEL_NAME' ===================== "
  echo
}

# checkCommitReadiness VERSION PEER ORG
function checkCommitReadiness() {
  ORG=$1
  shift 1
  setGlobals $ORG
  echo "===================== Checking the commit readiness of the chaincode definition on peer0.org${ORG} on channel '$CHANNEL_NAME'... ===================== "
  local rc=1
  local COUNTER=1
  # continue to poll
  # we either get a successful response, or reach MAX RETRY
  while [ $rc -ne 0 -a $COUNTER -lt $MAX_RETRY ] ; do
    sleep $DELAY
    echo "Attempting to check the commit readiness of the chaincode definition on peer0.org${ORG}, Retry after $DELAY seconds."
    set -x
    peer lifecycle chaincode checkcommitreadiness --signature-policy $SIGNATURE_POLICY --channelID $CHANNEL_NAME --name fabcar --version ${VERSION} --sequence ${VERSION} --output json --init-required >&log.txt
    res=$?
    set +x
    let rc=0
    for var in "$@"
    do
      grep "$var" log.txt &>/dev/null || let rc=1
    done
    COUNTER=$(expr $COUNTER + 1)
  done
  cat log.txt
  if test $rc -eq 0; then
    echo "===================== Checking the commit readiness of the chaincode definition successful on peer0.org${ORG} on channel '$CHANNEL_NAME' ===================== "
  else
    echo "!!!!!!!!!!!!!!! After $MAX_RETRY attempts, Check commit readiness result on peer0.org${ORG} is INVALID !!!!!!!!!!!!!!!!"
    echo
    exit 1
  fi
}

# commitChaincodeDefinition VERSION PEER ORG (PEER ORG)...
function commitChaincodeDefinition() {
  parsePeerConnectionParameters $@
  res=$?
  verifyResult $res "Invoke transaction failed on channel '$CHANNEL_NAME' due to uneven number of peer and org parameters "

  # while 'peer chaincode' command can get the orderer endpoint from the
  # peer (if join was successful), let's supply it directly as we know
  # it using the "-o" option
  set -x
  peer lifecycle chaincode commit --signature-policy $SIGNATURE_POLICY -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile $ORDERER_CA --channelID $CHANNEL_NAME --name fabcar $PEER_CONN_PARMS --version ${VERSION} --sequence ${VERSION} --init-required >&log.txt
  res=$?
  set +x
  cat log.txt
  verifyResult $res "Chaincode definition commit failed on peer0.org${ORG} on channel '$CHANNEL_NAME' failed"
  echo "===================== Chaincode definition committed on channel '$CHANNEL_NAME' ===================== "
  echo
}

# queryCommitted ORG
function queryCommitted() {
  ORG=$1
  setGlobals $ORG
  EXPECTED_RESULT="Version: ${VERSION}, Sequence: ${VERSION}, Endorsement Plugin: escc, Validation Plugin: vscc"
  echo "===================== Querying chaincode definition on peer0.org${ORG} on channel '$CHANNEL_NAME'... ===================== "
  local rc=1
  local COUNTER=1
  # continue to poll
  # we either get a successful response, or reach MAX RETRY
  while [ $rc -ne 0 -a $COUNTER -lt $MAX_RETRY ] ; do
    sleep $DELAY
    echo "Attempting to Query committed status on peer0.org${ORG}, Retry after $DELAY seconds."
    set -x
    peer lifecycle chaincode querycommitted --channelID $CHANNEL_NAME --name fabcar >&log.txt
    res=$?
    set +x
    test $res -eq 0 && VALUE=$(cat log.txt | grep -o '^Version: [0-9], Sequence: [0-9], Endorsement Plugin: escc, Validation Plugin: vscc')
    test "$VALUE" = "$EXPECTED_RESULT" && let rc=0
    COUNTER=$(expr $COUNTER + 1)
  done
  echo
  cat log.txt
  if test $rc -eq 0; then
    echo "===================== Query chaincode definition successful on peer0.org${ORG} on channel '$CHANNEL_NAME' ===================== "
    echo
  else
    echo "!!!!!!!!!!!!!!! After $MAX_RETRY attempts, Query chaincode definition result on peer0.org${ORG} is INVALID !!!!!!!!!!!!!!!!"
    echo
    exit 1
  fi
}

function chaincodeInvokeInit() {
  parsePeerConnectionParameters $@
  res=$?
  verifyResult $res "Invoke transaction failed on channel '$CHANNEL_NAME' due to uneven number of peer and org parameters "

  # while 'peer chaincode' command can get the orderer endpoint from the
  # peer (if join was successful), let's supply it directly as we know
  # it using the "-o" option
  set -x
  peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile $ORDERER_CA -C $CHANNEL_NAME -n fabcar $PEER_CONN_PARMS --isInit -c '{"function":"initLedger","Args":[]}' >&log.txt
  res=$?
  set +x
  cat log.txt
  verifyResult $res "Invoke execution on $PEERS failed "
  echo "===================== Invoke transaction successful on $PEERS on channel '$CHANNEL_NAME' ===================== "
  echo
}

function chaincodeQuery() {
  ORG=$1
  setGlobals $ORG
  echo "===================== Querying on peer0.org${ORG} on channel '$CHANNEL_NAME'... ===================== "
  local rc=1
  local COUNTER=1
  # continue to poll
  # we either get a successful response, or reach MAX RETRY
  while [ $rc -ne 0 -a $COUNTER -lt $MAX_RETRY ] ; do
    sleep $DELAY
    echo "Attempting to Query peer0.org${ORG}, Retry after $DELAY seconds."
    set -x
    peer chaincode query -C $CHANNEL_NAME -n fabcar -c '{"Args":["queryAllCars"]}' >&log.txt
    res=$?
    set +x
    let rc=$res
    COUNTER=$(expr $COUNTER + 1)
  done
  echo
  cat log.txt
  if test $rc -eq 0; then
    echo "===================== Query successful on peer0.org${ORG} on channel '$CHANNEL_NAME' ===================== "
    echo
  else
    echo "!!!!!!!!!!!!!!! After $MAX_RETRY attempts, Query result on peer0.org${ORG} is INVALID !!!!!!!!!!!!!!!!"
    echo
    exit 1
  fi
}

function join_by { local d=$1; shift; local f=$1; shift; printf %s "$f" "${@/#/$d}"; }

function main() {
  ORGMEMBERARR=""
  for i in "${!PeerAddress[@]}"; do
    ORGMEMBERARR="${ORGMEMBERARR} Org$((i+1))MSP.member"
  done
  SIGNATURE_POLICY=$(join_by "','" ${ORGMEMBERARR})
  SIGNATURE_POLICY="AND('${SIGNATURE_POLICY}')"
  # SIGNATURE_POLICY="AND('Org1MSP.member','Org2MSP.member','Org3MSP.member','Org4MSP.member','Org5MSP.member')"

  ## at first we package the chaincode
  packageChaincode 1

  ## Install chaincode on peer0.org1 and peer0.org2
  echo "Installing chaincode on peers..."
  for i in "${!PeerAddress[@]}"; do
    installChaincode $((i+1))
  done

  ## query whether the chaincode is installed
  queryInstalled 1

  ## approve the definition for org
  for i in "${!PeerAddress[@]}"; do
    approveForMyOrg $((i+1))
    checkCommitReadiness $((i+1)) "\"Org$((i+1))MSP\": true"
  done

  ## now that we know for sure both orgs have approved, commit the definition
  COMMIT_IDS=""
  for i in "${!PeerAddress[@]}"; do
    COMMIT_IDS="${COMMIT_IDS} $((i+1))"
  done
  commitChaincodeDefinition ${COMMIT_IDS}

  ## query on both orgs to see that the definition committed successfully
  for i in "${!PeerAddress[@]}"; do
    queryCommitted $((i+1))
  done

  ## Invoke the chaincode
  chaincodeInvokeInit ${COMMIT_IDS}

  sleep 10

  # Query chaincode on peer0.org1
  # echo "Querying chaincode on peer0.org1..."
  # chaincodeQuery 1

  exit 0
}

main

