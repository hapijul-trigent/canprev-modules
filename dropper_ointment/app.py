import streamlit as st
import cv2
import numpy as np
from PIL import Image
from .tools import load_dropper_ointment_model, detect_dropper_ointment

# Load
dropper_ointment_model = load_dropper_ointment_model(model_path='dropper_ointment/model_bottle_dropper_ointment.pt')

def main():

    uploaded_file = st.file_uploader("Choose an image...", type="jpg", key='dropper_ointment_model')

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
            
            result_image = detect_dropper_ointment(image, model=dropper_ointment_model)
            result_image_pil = Image.fromarray(result_image)

            st.header("Analysis")
            st.image(result_image_pil, caption='...', use_column_width=True)

if __name__ == "__main__":
    main()
