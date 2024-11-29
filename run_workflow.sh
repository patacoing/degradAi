#!/bin/bash

JOB=${1:-"ml-pipeline"}
SECRET_FILE="act.secrets"
DOCKER_IMAGE="ghcr.io/catthehacker/ubuntu:act-latest"

#Check if act is installed
if ! command -v act 2>&1 >/dev/null
then
  echo "act is not installed. Please install it before running this script."
  exit 1
fi

docker pull $DOCKER_IMAGE

act -P ubuntu-latest=$DOCKER_IMAGE push -j $JOB --secret-file=$SECRET_FILE