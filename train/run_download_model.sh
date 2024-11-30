#!/bin/bash

MODEL_VERSION=$1
AZURE_RESOURCE_GROUP_NAME=$2
AZURE_ML_WORKSPACE_NAME=$3
AZURE_LOCATION=$4
AZURE_SUBSCRIPTION_ID=$5
DOWNLOAD_PATH=$6
MODEL_NAME="degradai"


# Retrieve the latest model version if not provided
if [ "$MODEL_VERSION" = "latest" ]
then
  MODEL_VERSION=$(az ml model list --workspace-name "$AZURE_ML_WORKSPACE_NAME" --resource-group "$AZURE_RESOURCE_GROUP_NAME" --name "$MODEL_NAME" --query "max([*].version).to_number(@)")
fi

echo "Retrieving the model version $MODEL_VERSION"

poetry run python azureml_run_download_model.py \
    --subscription_id "$AZURE_SUBSCRIPTION_ID" \
    --resource_group_name "$AZURE_RESOURCE_GROUP_NAME" \
    --workspace_name "$AZURE_ML_WORKSPACE_NAME" \
    --location "$AZURE_LOCATION" \
    --version "$MODEL_VERSION" \
    --download_path "$DOWNLOAD_PATH"

cd $DOWNLOAD_PATH
mkdir -p model/
cp $MODEL_NAME/*.keras model/
rm -rf $MODEL_NAME