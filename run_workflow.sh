#!/bin/bash

SUBSCRIPTION_ID=$1

if [ -z "$SUBSCRIPTION_ID" ]
then
  echo "Usage: $0 <subscription_id> [job_name]"
  exit 1
fi

JOB=${2:-"ml-pipeline"}
SECRET_FILE="act.secrets"
DOCKER_IMAGE="ghcr.io/catthehacker/ubuntu:act-latest"

#Check if act is installed
if ! command -v act 2>&1 >/dev/null
then
  echo "act is not installed. Please install it before running this script."
  exit 1
fi

#Set up the credentials
source setup_credentials.sh $SUBSCRIPTION_ID

docker pull $DOCKER_IMAGE

act -P ubuntu-latest=$DOCKER_IMAGE push -j $JOB --secret-file=$SECRET_FILE