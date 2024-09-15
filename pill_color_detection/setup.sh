#!/bin/bash


echo "Updating package index..."
sudo apt update
git clone https://github.com/ultralytics/yolov5
pip install -q ultralytics streamlit