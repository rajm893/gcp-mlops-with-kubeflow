#!/bin/bash 
PROJECT_ID="gcp-project-raj33342"
REGION="europe-west1"
REPOSITORY="kfp-mlops"
IMAGE='demo'
IMAGE_TAG='demo_model:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
