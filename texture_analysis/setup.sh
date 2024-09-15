#!/bin/bash


echo "Updating package index..."
sudo apt-get install libgl1-mesa-glx


# Upgrade pip to the latest version
pip install --upgrade pip
git clone https://github.com/ultralytics/yolov5
pip install ultralytics streamlit scikit-image
pip install --no-cache-dir -r requirements.txt
