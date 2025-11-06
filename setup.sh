#!/bin/bash
# ==================================================
# Script: setup_docker_aws.sh
# Purpose: Install Docker, AWS CLI, and configure permissions on Ubuntu
# Author: Soham (because you’re the dev with style 😎)
# ==================================================

set -e  # Exit immediately if any command fails

echo "🚀 Updating package lists..."
sudo apt-get update -y

echo "🐳 Installing Docker..."
sudo apt-get install -y docker.io

echo "🔧 Starting and enabling Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

echo "👤 Adding current user to Docker group..."
sudo usermod -aG docker ubuntu

echo "📦 Installing unzip and curl..."
sudo apt-get install -y unzip curl

echo "☁️ Downloading AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

echo "🗜️ Unzipping AWS CLI package..."
unzip -o awscliv2.zip

echo "⚙️ Installing AWS CLI..."
sudo ./aws/install

echo "🧹 Cleaning up temporary files..."
rm -rf awscliv2.zip aws/

echo "✅ Installation complete!"
echo "ℹ️ You may need to log out and back in for Docker permissions to take effect."
echo "💡 To verify installations, run:"
echo "   docker --version"
echo "   aws --version"
