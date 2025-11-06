#!/bin/bash

# Ensure Docker is running
sudo systemctl start docker

# Check if the container is already running or exists
if [ "$(sudo docker ps -aq -f name=emotion-det)" ]; then
    echo "Stopping and removing existing container 'emotion-det'..."
    sudo docker stop emotion-det
    sudo docker rm emotion-det
fi

# Run the Docker container
echo "Starting new container 'emotion-det'..."
sudo docker run -d -p 80:5000 \
    -e DAGSHUB_PAT=b505b69837f827ef5da9faccb4a1043ffd54c5d3 \
    --name emotion-det \
    277707129180.dkr.ecr.us-east-1.amazonaws.com/emotion:latest
