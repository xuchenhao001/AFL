#!/bin/bash

CHANNEL_NAME="$1"
DELAY="$2"
MAX_RETRY="$3"
VERBOSE="$4"
: ${CHANNEL_NAME:="mychannel"}
: ${DELAY:="3"}
: ${MAX_RETRY:="5"}
: ${VERBOSE:="false"}

# import utils
. scripts/envVar.sh

if [ ! -d "channel-artifacts" ]; then
  mkdir channel-artifacts
fi

function createChannelTx() {
  set -x
  echo $FABRIC_CFG_PATH
  configtxgen -profile TwoOrgsChannel -outputCreateChannelTx ./channel-artifacts/${CHANNEL_NAME}.tx -channelID $CHANNEL_NAME
  res=$?
  set +x
  if [ $res -ne 0 ]; then
    echo "Failed to generate channel configuration transaction..."
    exit 1
  fi
  echo

}

function createAncorPeerTx() {
  ORGMSP=$1

  echo "#######    Generating anchor peer update transaction for ${ORGMSP}  ##########"
  set -x
  configtxgen -profile TwoOrgsChannel -outputAnchorPeersUpdate ./channel-artifacts/${ORGMSP}anchors.tx -channelID $CHANNEL_NAME -asOrg ${ORGMSP}
  res=$?
  set +x
  if [ $res -ne 0 ]; then
    echo "Failed to generate anchor peer update transaction for ${ORGMSP}..."
    exit 1
  fi
  echo
}

function createChannel() {
  setGlobals 1
  # Poll in case the raft leader is not set yet
  local rc=1
  local COUNTER=1
  while [ $rc -ne 0 -a $COUNTER -lt $MAX_RETRY ] ; do
    sleep $DELAY
    set -x
    peer channel create -o localhost:7050 -c $CHANNEL_NAME --ordererTLSHostnameOverride orderer.example.com -f ./channel-artifacts/${CHANNEL_NAME}.tx --outputBlock ./channel-artifacts/${CHANNEL_NAME}.block --tls --cafile $ORDERER_CA >&log.txt
    res=$?
    set +x
    let rc=$res
    COUNTER=$(expr $COUNTER + 1)
  done
  cat log.txt
  verifyResult $res "Channel creation failed"
  echo
  echo "===================== Channel '$CHANNEL_NAME' created ===================== "
  echo
}

# queryCommitted ORG
function joinChannel() {
  ORG=$1
  setGlobals $ORG
  local rc=1
  local COUNTER=1
  ## Sometimes Join takes time, hence retry
  while [ $rc -ne 0 -a $COUNTER -lt $MAX_RETRY ] ; do
    sleep $DELAY
    set -x
    peer channel join -b ./channel-artifacts/$CHANNEL_NAME.block >&log.txt
    res=$?
    set +x
    let rc=$res
    COUNTER=$(expr $COUNTER + 1)
  done
  cat log.txt
  echo
  verifyResult $res "After $MAX_RETRY attempts, peer0.org${ORG} has failed to join channel '$CHANNEL_NAME' "
}

function updateAnchorPeers() {
  ORG=$1
  setGlobals $ORG
  local rc=1
  local COUNTER=1
  ## Sometimes Join takes time, hence retry
  while [ $rc -ne 0 -a $COUNTER -lt $MAX_RETRY ] ; do
    sleep $DELAY
    set -x
    peer channel update -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com -c $CHANNEL_NAME -f ./channel-artifacts/${CORE_PEER_LOCALMSPID}anchors.tx --tls --cafile $ORDERER_CA >&log.txt
    res=$?
    set +x
    let rc=$res
    COUNTER=$(expr $COUNTER + 1)
  done
  cat log.txt
  verifyResult $res "Anchor peer update failed"
  echo "===================== Anchor peers updated for org '$CORE_PEER_LOCALMSPID' on channel '$CHANNEL_NAME' ===================== "
  sleep $DELAY
  echo
}

function verifyResult() {
  if [ $1 -ne 0 ]; then
    echo "!!!!!!!!!!!!!!! "$2" !!!!!!!!!!!!!!!!"
    echo
    exit 1
  fi
}


function main() {
  ## Create channeltx
  echo "### Generating channel create transaction '${CHANNEL_NAME}.tx' ###"
  createChannelTx

  ## Create anchorpeertx
  echo "### Generating anchor peer update transactions ###"
  for i in "${!PeerAddress[@]}"; do
    createAncorPeerTx Org$((i+1))MSP
  done

  ## Create channel
  echo "Creating channel "$CHANNEL_NAME
  createChannel

  ## Join all the peers to the channel
  echo "Join Org peers to the channel..."
  for i in "${!PeerAddress[@]}"; do
    joinChannel $((i+1))
  done

  ## Set the anchor peers for each org in the channel
  echo "Updating anchor peers for orgs..."
  for i in "${!PeerAddress[@]}"; do
    updateAnchorPeers $((i+1))
  done

  echo
  echo "========= Channel successfully joined =========== "
  echo

  exit 0
}

main

