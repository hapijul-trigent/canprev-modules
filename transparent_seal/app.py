import streamlit as st
import cv2
import numpy as np
from PIL import Image
from .tools import load_transparent_seal_model, detect_transparent_seal

# Load
transparent_seal_model = load_transparent_seal_model(model_path='transparent_seal/model_transparent_seal.pt')

def main():

    uploaded_file = st.file_uploader("Choose an image...", type="jpg", key='Transparent Seal')

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
            
            result_image = detect_transparent_seal(image, model=transparent_seal_model)
            result_image_pil = Image.fromarray(result_image)

            st.header("Analysis")
            st.image(result_image_pil, caption='...', use_column_width=True)

if __name__ == "__main__":
    main()
