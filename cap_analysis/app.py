import streamlit as st
import cv2
import numpy as np
from PIL import Image
from .tools import load_neckband_model, detect_neckband 

# Load
neckband_model = load_neckband_model(model_path='cap_analysis/model_caps.pt')

def main():

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        
        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            st.header("Uploaded Image")
            image_pil = Image.fromarray(image)
            st.image(image=image, caption='...', use_column_width=True)
        
        with col2:
            # Perform neckband detection
            result_image = detect_neckband(image, model=neckband_model)
            
            # Convert images to PIL format for displaying
            result_image_pil = Image.fromarray(result_image)
            st.header("Detection")
            st.image(result_image_pil, caption='...', use_column_width=True)

if __name__ == "__main__":
    main()
