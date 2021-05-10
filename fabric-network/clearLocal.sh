#!/bin/bash

# NEED TO RUN WITH ROOT
function clearContainers() {
  CONTAINER_IDS=$(docker ps -a | awk '($2 ~ /dev-peer.*/) {print $1}')
  if [ -z "$CONTAINER_IDS" -o "$CONTAINER_IDS" == " " ]; then
    echo "---- No containers available for deletion ----"
  else
    docker rm -f $CONTAINER_IDS
  fi
}

function removeUnwantedImages() {
  DOCKER_IMAGE_IDS=$(docker images | awk '($1 ~ /dev-peer.*/) {print $3}')
  if [ -z "$DOCKER_IMAGE_IDS" -o "$DOCKER_IMAGE_IDS" == " " ]; then
    echo "---- No images available for deletion ----"
  else
    docker rmi -f $DOCKER_IMAGE_IDS
  fi
}

function clear() {
	clearContainers
	removeUnwantedImages
	set -x
	rm -rf system-genesis-block/*.block networkCache.tar.gz network-cache/* 
	rm -rf channel-artifacts log.txt fabcar.tar.gz fabcar

	# clean wallet
	rm -f ../blockchain-server/routes/rest/wallet/*
}

clear
