import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import matplotlib.pyplot as plt
from supervision import Detections, BoxAnnotator, LabelAnnotator, ColorPalette


@st.cache_resource(show_spinner=False)
def load_seal_model(model_path='neckbandModelv8.pt'):
    """Load and return the YOLOv8 model."""
    return YOLO(model_path)


def detect_seal(image, model):
    """Perform object detection and return annotated image with highest confidence box only."""
    result = model(image)[0]
    detections = Detections.from_ultralytics(result)
   
    if detections.xyxy.any():
        detections = detections[np.array([True if detections.confidence.max() == confidence else False for confidence in detections.confidence])]
        annotated_image = image.copy()
        
        # Annotate the highest confidence box
        annotator = BoxAnnotator(color=ColorPalette.ROBOFLOW, thickness=5)
        label_annotator = LabelAnnotator(text_scale=2, color=ColorPalette.ROBOFLOW, text_thickness=7)
        labels = [
            f"{class_name} {confidence*100:.2f}%"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]
        annotated_image = label_annotator.annotate(annotated_image, detections=detections, labels=labels)
        annotated_image = annotator.annotate(annotated_image, detections=detections)
        return annotated_image
    else:
        return image
