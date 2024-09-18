import streamlit as st
import cv2
import numpy as np
from PIL import Image
import cap_analysis.app
import texture_analysis.app
import pill_color_detection.app


# Function for cap detection
def cap_detection(image):
    st.write("Cap detection functionality goes here.")
    # You can use OpenCV to detect objects in the image
    # For now, we are just showing the image
    st.image(image, caption="Uploaded Image for Cap Detection", use_column_width=True)

# Function for texture analysis
def texture_analysis_f(image):
    st.write("Texture analysis functionality goes here.")
    # Sample method: Convert to grayscale and calculate texture features
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    st.image(gray_image, caption="Grayscale Image for Texture Analysis", use_column_width=True)
    # Add actual texture analysis code (e.g., Haralick texture features) here

# Function for pill analysis
def pill_analysis(image):
    st.write("Pill analysis functionality goes here.")
    # Sample: Display the image for pill analysis
    st.image(image, caption="Uploaded Image for Pill Analysis", use_column_width=True)
    # Add actual pill analysis (e.g., shape, color, imprint analysis) here


st.set_page_config(
    page_title="Canprev AI",  # Title of the web page
    page_icon=":camera:",                # Favicon, use emoji or a file path
    layout="centered",                       # Layout options: "wide" or "centered"
    initial_sidebar_state="expanded"     # Initial state of the sidebar: "auto", "expanded", or "collapsed"
)

# Main app function
def main():
    st.image('CanPrev_4D-logo.png', width=200)
    st.title("Canprev AI")
    
    # Tabs for different types of analysis
    tab1, tab2, tab3 = st.tabs(["Cap Detection", "Texture Analysis", "Pill Analysis"])
    
    # Cap Detection tab
    with tab1:
        st.header("Cap Detection")
        cap_analysis.app.main()
    # Texture Analysis tab
    with tab2:
        st.header("Texture Analysis")
        texture_analysis.app.main()
    # Pill Analysis tab
    with tab3:
        st.header("Pill Analysis")
        pill_color_detection.app.main()

if __name__ == "__main__":
    main()
