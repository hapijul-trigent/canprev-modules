#!/bin/bash


echo "Updating package index..."
sudo apt-get install libgl1-mesa-glx


# Upgrade pip to the latest version
pip install --upgrade pip
pip install -r requirements.txt