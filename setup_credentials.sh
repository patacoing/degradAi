#!/bin/bash

SUBSCRIPTION_ID=$1

# get credentials
credentials=$(az ad sp create-for-rbac --name "mlapp-romain" --role contributor --scopes "/subscriptions/46113ffc-4a75-4334-ad25-40ea09122058" --sdk-auth)

# Format credentials
formatted_credentials=$(echo $credentials | jq -c @json | tr -d "\\" | sed -r 's/^.{1}//' | sed 's/.$//')

gh secret set AZURE_CREDENTIALS --body "$formatted_credentials" --env "degradai"
echo "AZURE_CREDENTIALS=$formatted_credentials" > act.secrets