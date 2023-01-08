#!/bin/bash

# release certs; clean logs; clean certs;
# prepending $PWD/../bin to PATH to ensure we are picking up the correct binaries
# this may be commented out to resolve installed version of tools if desired
export PATH=${PWD}/bin:$PATH
export FABRIC_CFG_PATH=${PWD}/network-cache
export VERBOSE=false

source ./network.config
source ./.env
source scriptUtils.sh

function releaseCerts() {
  tar -zcf networkCache.tar.gz network-cache/
  for i in "${!PeerAddress[@]}"; do
    addrIN=(${PeerAddress[i]//:/ })
    # check ssh connection first
    status=$(ssh -o BatchMode=yes -o ConnectTimeout=5 ${HostUser}@${addrIN[0]} echo ok 2>&1)
    if [[ $status != "ok" ]]; then
      echo "Please add your public key to other hosts with user \"${HostUser}\" before release certs through command \"ssh-copy-id\"!"
      exit 1
    fi
    scp networkCache.tar.gz ${HostUser}@${addrIN[0]}:~/AFL/fabric-network/networkCache.tar.gz
    ssh ${HostUser}@${addrIN[0]} "cd ~/AFL/fabric-network/ && tar -zxf networkCache.tar.gz"
  done
}

function createOrgs() {

  if [ -d "network-cache/peerOrganizations" ]; then
    rm -Rf network-cache/peerOrganizations && rm -Rf network-cache/ordererOrganizations
  fi

  # Create crypto material using cryptogen
  which cryptogen
  if [ "$?" -ne 0 ]; then
    fatalln "cryptogen tool not found. exiting"
  fi
  infoln "Generate certificates using cryptogen tool"

  infoln "Create Orgs Identities"

  set -x
  cryptogen generate --config=./network-cache/crypto-config.yaml --output="network-cache"
  res=$?
  { set +x; } 2>/dev/null
  if [ $res -ne 0 ]; then
    fatalln "Failed to generate certificates..."
  fi

  for i in "${!PeerAddress[@]}"; do
    prepareID org$((i+1)) Org$((i+1))MSP
  done

  echo
  echo "Generate CCP files for Orgs"
  ccpGenerate
  for i in "${!PeerAddress[@]}"; do
    prepareCCP org$((i+1))
  done
}

function one_line_pem {
    echo "`awk 'NF {sub(/\\n/, ""); printf "%s\\\\\\\n",$0;}' $1`"
}

function json_ccp {
    local PP=$(one_line_pem $5)
    local CP=$(one_line_pem $6)
    sed -e "s/\${ORG}/$1/" \
        -e "s/\${P0IPADDR}/$2/" \
        -e "s/\${P0PORT}/$3/" \
        -e "s/\${CAPORT}/$4/" \
        -e "s#\${PEERPEM}#$PP#" \
        -e "s#\${CAPEM}#$CP#" \
        organizations/ccp-template.json
}

function ccpGenerate() {
  for i in "${!PeerAddress[@]}"; do
    addrIN=(${PeerAddress[i]//:/ })
    echo "$(json_ccp $((i+1)) ${addrIN[0]} ${addrIN[1]} $((addrIN[1]+3)) network-cache/peerOrganizations/org$((i+1)).example.com/tlsca/tlsca.org$((i+1)).example.com-cert.pem network-cache/peerOrganizations/org$((i+1)).example.com/ca/ca.org$((i+1)).example.com-cert.pem)" > network-cache/peerOrganizations/org$((i+1)).example.com/connection-org$((i+1)).json
  done
}

# USING_ORG=org1
function prepareID() {
  USING_ORG=$1
  MSPID=$2
  CERT=$(sed ':a;N;$!ba;s/\n/\\\\n/g' ./network-cache/peerOrganizations/${USING_ORG}.example.com/users/User1@${USING_ORG}.example.com/msp/signcerts/User1@${USING_ORG}.example.com-cert.pem)
  CERT=$(echo $CERT | sed 's/\//\\\//g')
  PRIK=$(sed ':a;N;$!ba;s/\n/\\\\r\\\\n/g' ./network-cache/peerOrganizations/${USING_ORG}.example.com/users/User1@${USING_ORG}.example.com/msp/keystore/priv_sk)
  PRIK=$(echo $PRIK | sed 's/\//\\\//g')
  sed -e "s/CERT/${CERT}/g" -e "s/PRIK/${PRIK}/g" -e "s/MSPID/${MSPID}/g" ./organizations/wallet-template.json > ${USING_ORG}.id
  mkdir -p ../blockchain-server/routes/rest/wallet/
  mv ${USING_ORG}.id ../blockchain-server/routes/rest/wallet/
}

function prepareCCP() {
  USING_ORG=$1
  cp ./network-cache/peerOrganizations/${USING_ORG}.example.com/connection-${USING_ORG}.json ../blockchain-server/routes/rest/wallet/connection-${USING_ORG}.json
}

# Generate orderer system channel genesis block.
function createConsortium() {

  which configtxgen
  if [ "$?" -ne 0 ]; then
    echo "configtxgen tool not found. exiting"
    exit 1
  fi

  echo "#########  Generating Orderer Genesis block ##############"

  # Note: For some unknown reason (at least for now) the block file can't be
  # named orderer.genesis.block or the orderer will fail to launch!
  set -x
  configtxgen -profile TwoOrgsOrdererGenesis -channelID system-channel -outputBlock ./system-genesis-block/genesis.block
  res=$?
  set +x
  if [ $res -ne 0 ]; then
    echo "Failed to generate orderer genesis block..."
    exit 1
  fi
}

# Do some basic sanity checking to make sure that the appropriate versions of fabric
# binaries/images are available. In the future, additional checking for the presence
# of go or other items could be added.
function checkPrereqs() {
  ## Check if your have cloned the peer binaries and configuration files.
  peer version > /dev/null 2>&1

  if [[ $? -ne 0 ]]; then
    echo "ERROR! Peer binary and configuration files not found.."
    echo
    echo "Follow the instructions in the Fabric docs to install the Fabric Binaries:"
    echo "https://hyperledger-fabric.readthedocs.io/en/latest/install.html"
    exit 1
  fi
  # use the fabric tools container to see if the samples and binaries match your
  # docker images
  LOCAL_VERSION=$(peer version | sed -ne 's/ Version: //p')
  DOCKER_IMAGE_VERSION=$(docker run --rm hyperledger/fabric-tools:$IMAGE_TAG peer version | sed -ne 's/ Version: //p' | head -1)

  echo "LOCAL_VERSION=$LOCAL_VERSION"
  echo "DOCKER_IMAGE_VERSION=$DOCKER_IMAGE_VERSION"

  if [ "$LOCAL_VERSION" != "$DOCKER_IMAGE_VERSION" ]; then
    echo "=================== WARNING ==================="
    echo "  Local fabric binaries and docker images are  "
    echo "  out of  sync. This may cause problems.       "
    echo "==============================================="
  fi

  for UNSUPPORTED_VERSION in $BLACKLISTED_VERSIONS; do
    echo "$LOCAL_VERSION" | grep -q $UNSUPPORTED_VERSION
    if [ $? -eq 0 ]; then
      echo "ERROR! Local Fabric binary version of $LOCAL_VERSION does not match the versions supported by the test network."
      exit 1
    fi

    echo "$DOCKER_IMAGE_VERSION" | grep -q $UNSUPPORTED_VERSION
    if [ $? -eq 0 ]; then
      echo "ERROR! Fabric Docker image version of $DOCKER_IMAGE_VERSION does not match the versions supported by the test network."
      exit 1
    fi
  done
}

#################################
# prepare docker-compose-org*.yaml
#################################

function generatePeerDockerCompose() {
  for i in "${!PeerAddress[@]}"; do
    addrIN=(${PeerAddress[i]//:/ })
    echo "$(prepareDockerCompose $((i+1)) ${addrIN[0]} ${addrIN[1]} $((addrIN[1]+1)))" > network-cache/docker-compose-org$((i+1)).yaml
  done
}

function prepareDockerCompose() {
  sed -e "s/\${ORGNUM}/$1/g" \
      -e "s/\${ADDR}/$2/g" \
      -e "s/\${PORT}/$3/g" \
      -e "s/\${CCPORT}/$4/g" \
      docker/peer-template.yaml
}


#################################
# prepare orderer.yaml
#################################

function generateOrdererDockerCompose() {
  cp docker/orderer-template.yaml network-cache/orderer.yaml
}

#################################
# prepare core.yaml
#################################

function prepareCoreConfig() {
  cp configtx/core.yaml network-cache/core.yaml
}

##########################
# prepare configtx.yaml
##########################
function generateConfigTX() {
  ORGS_DETAIL=""
  for i in "${!PeerAddress[@]}"; do
      addrIN=(${PeerAddress[i]//:/ })
      ORGS_DETAIL=$(echo -e "$ORGS_DETAIL\n$(parseOrgDetail $((i+1)) ${addrIN[0]} ${addrIN[1]})")
  done
  ORGS_DETAIL=$(echo "$ORGS_DETAIL" | awk '{printf "%s\\n", $0}' | sed 's/\\n//')
  ORGS_DETAIL=$(echo "${ORGS_DETAIL//\"/\\\"}")
  ORGS_DETAIL=$(echo "${ORGS_DETAIL//\&/\\\&}")
  ORGS_DETAIL=$(echo "$ORGS_DETAIL" | sed 's/\//\\\//g')

  ORGSLIST1=""
  for i in "${!PeerAddress[@]}"; do
    ORGSLIST1=$(echo "$ORGSLIST1 \n                    - *Org$((i+1))")
  done
  ORGSLIST1=$(echo "$ORGSLIST1" | sed 's/\\n //')

  ORGSLIST2=""
  for i in "${!PeerAddress[@]}"; do
    ORGSLIST2=$(echo "$ORGSLIST2 \n                - *Org$((i+1))")
  done
  ORGSLIST2=$(echo "$ORGSLIST2" | sed 's/\\n //')

  ORG1AddrPort=(${PeerAddress[0]//:/ })
  ORDERERADDR="${ORG1AddrPort[0]}:$((${ORG1AddrPort[1]}-1))"
  
  sed -e "s/ORDERERADDR/${ORDERERADDR}/g" \
      -e "s/ORGSDETAIL/$ORGS_DETAIL/g" \
      -e "s/ORGSLIST1/$ORGSLIST1/g" \
      -e "s/ORGSLIST2/$ORGSLIST2/g" \
      ./configtx/configtx-template.yaml > network-cache/configtx.yaml
}

function parseOrgDetail() {
    myOrgNum=$1
    myOrgAddr=$2
    myOrgPort=$3
    ORG_DETAIL_TEMP=$(cat <<EOF
    - &OrgORGNUM
        Name: OrgORGNUMMSP
        ID: OrgORGNUMMSP
        MSPDir: ./peerOrganizations/orgORGNUM.example.com/msp
        Policies:
            Readers:
                Type: Signature
                Rule: "OR('OrgORGNUMMSP.admin', 'OrgORGNUMMSP.peer', 'OrgORGNUMMSP.client')"
            Writers:
                Type: Signature
                Rule: "OR('OrgORGNUMMSP.admin', 'OrgORGNUMMSP.client')"
            Admins:
                Type: Signature
                Rule: "OR('OrgORGNUMMSP.admin')"
            Endorsement:
                Type: Signature
                Rule: "OR('OrgORGNUMMSP.peer')"
        AnchorPeers:
            - Host: ORGADDR
              Port: ORGPORT
EOF
)

    ORG_DETAIL=${ORG_DETAIL_TEMP//ORGNUM/$myOrgNum}
    ORG_DETAIL=${ORG_DETAIL//ORGADDR/$myOrgAddr}
    ORG_DETAIL=${ORG_DETAIL//ORGPORT/$myOrgPort}
    echo -e "$ORG_DETAIL"
}

##########################
# prepare crypto-config.yaml
##########################

function generateCryptoConfig() {
    PEERS_DETAIL=""
    for i in "${!PeerAddress[@]}"; do
        addrIN=(${PeerAddress[i]//:/ })
        PEERS_DETAIL=$(echo -e "$PEERS_DETAIL\n$(parseCryptoPeerDetail $((i+1)) ${addrIN[0]})")
    done
    PEERS_DETAIL=$(echo "$PEERS_DETAIL" | awk '{printf "%s\\n", $0}' | sed 's/\\n//')
    PEERS_DETAIL=$(echo "${PEERS_DETAIL//\"/\\\"}")
    PEERS_DETAIL=$(echo "${PEERS_DETAIL//\&/\\\&}")
    PEERS_DETAIL=$(echo "$PEERS_DETAIL" | sed 's/\//\\\//g')

    ordererIN=(${PeerAddress[0]//:/ })
    sed -e "s/ORDERERADDR/${ordererIN[0]}/g" \
        -e "s/PEERSDETAIL/$PEERS_DETAIL/g" \
        ./organizations/crypto-config-template.yaml > network-cache/crypto-config.yaml
}

function parseCryptoPeerDetail() {
    myOrgNum=$1
    myOrgAddr=$2
    PEER_DETAIL_TEMP=$(cat <<EOF
  - Name: OrgORGNUM
    Domain: orgORGNUM.example.com
    EnableNodeOUs: true
    Template:
      Count: 1
      SANS:
        - localhost
        - PEERADDR
    Users:
      Count: 1
EOF
)

    PEER_DETAIL=${PEER_DETAIL_TEMP//ORGNUM/$myOrgNum}
    PEER_DETAIL=${PEER_DETAIL//PEERADDR/$myOrgAddr}
    echo -e "$PEER_DETAIL"
}

##########################
# prepare certs
##########################

function generateCerts() {
  checkPrereqs
  createOrgs
  createConsortium
}

##########################
# main function
##########################

function main() {
  generateCryptoConfig
  generateConfigTX
  generateCerts
  generatePeerDockerCompose
  generateOrdererDockerCompose
  prepareCoreConfig
  releaseCerts
}

main


