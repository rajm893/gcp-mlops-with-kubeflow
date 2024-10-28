
#!/bin/bash 
PROJECT_ID="gcp-project-raj33342"
REGION="europe-west1"
REPOSITORY="kfp-mlops"
IMAGE_TAG='demo_model:latest'


# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
 
 # Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG