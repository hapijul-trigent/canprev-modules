import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt
from supervision import Detections, BoundingBoxAnnotator

# Ensure you have Streamlit installed:
# pip install streamlit

@st.cache_resource(show_spinner=False)
def load_neckband_model(model_path='neckbandModelv8.pt'):
    """Load and return the YOLOv8 model."""
    return YOLO(model_path)


def detect_neckband(image, model):
    """Perform object detection and return annotated image."""
    result = model(image)[0]
    annotated_image = result.plot()
    detections = Detections.from_ultralytics(result)

    return annotated_image

