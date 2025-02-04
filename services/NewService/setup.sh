#!/bin/bash

# Create necessary directories
mkdir -p app/data
mkdir -p app/model

python -c """
import boto3
import json
import os

def get_kaggle_api_key(secret_name):
  client = boto3.client(‘secretsmanager’)
  response = client.get_secret_value(SecretId=secret_name)
  secret = json.loads(response[‘SecretString’])

  os.environ[“KAGGLE_USERNAME”] = secret[“username”]
  os.environ[“KAGGLE_KEY”] = secret[“key”]
get_kaggle_api_key("mkuce-kaggleAPI")
"""

# Download the dataset using Kaggle API
echo "Downloading house price dataset..."
kaggle datasets download alyelbadry/house-pricing-dataset -p app/data --unzip

echo "Setup complete. Dataset is available in app/data/"

# echo cloning down textcat_goemotions...
# spacy project clone tutorials/textcat_goemotions
# echo Downloading spacy project assets...
# cd textcat_goemotions
# spacy project assets
# cd ..