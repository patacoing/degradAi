#!/bin/bash


function usage {
  echo "Usage: $0 subscription_id [options]"
  echo "Options:"
  echo "  -g: Set the secret in the GitHub repository"
  echo "  -h: Display this help message"
  exit 1
}

SUBSCRIPTION_ID=$1
shift

SECRET_FILE="act.secrets"
GH_SECRET=false
GH_ENV="degradai"
OPTSTRING="gh"

while getopts ${OPTSTRING} opt; do
  case ${opt} in
    g)
      echo "Setting secret in GitHub repository"
      GH_SECRET=true
      ;;
    h)
      usage
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      usage
      ;;
  esac
done

if [ -z "$SUBSCRIPTION_ID" ]; then
  usage
fi

# get credentials
credentials=$(az ad sp create-for-rbac --name "mlapp-romain" --role contributor --scopes "/subscriptions/$SUBSCRIPTION_ID" --sdk-auth)

# Format credentials to remove escape characters and remove first and last character
formatted_credentials=$(echo $credentials | jq -c @json | tr -d "\\" | sed -r 's/^.{1}//' | sed 's/.$//')

if [ "$GH_SECRET" = true ]; then
  gh secret set AZURE_CREDENTIALS --body "$formatted_credentials" --env "$GH_ENV"
  gh secret set AZURE_SUBSCRIPTION_ID --body "$SUBSCRIPTION_ID" --env "$GH_ENV"
fi

echo "AZURE_CREDENTIALS=$formatted_credentials" > $SECRET_FILE
echo "AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID" >> $SECRET_FILE
echo "Secrets saved in $SECRET_FILE"