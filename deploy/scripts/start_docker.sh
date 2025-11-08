#!/bin/bash

# Login to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 277707129180.dkr.ecr.us-east-1.amazonaws.com

# Pull the image
docker pull 277707129180.dkr.ecr.us-east-1.amazonaws.com/emotion-det:v1

# Stop container if running
docker stop emotion-det || true

# Remove old container if exists
docker rm emotion-det || true

# Run new container
docker run -d -p 80:5000 -e DAGSHUB_PAT=b505b69837f827ef5da9faccb4a1043ffd54c5d3 --name emotion-det \
  277707129180.dkr.ecr.us-east-1.amazonaws.com/emotion-det:v1
