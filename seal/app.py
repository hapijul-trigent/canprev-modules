import streamlit as st
import cv2
import numpy as np
from PIL import Image
from .tools import load_seal_model, detect_seal

# Load
seal_model = load_seal_model(model_path='seal/model_seal.pt')

def main():

    uploaded_file = st.file_uploader("Choose an image...", type="jpg", key='Seal')

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
            
            result_image = detect_seal(image, model=seal_model)
            result_image_pil = Image.fromarray(result_image)

            st.header("Analysis")
            st.image(result_image_pil, caption='...', use_column_width=True)

if __name__ == "__main__":
    main()
