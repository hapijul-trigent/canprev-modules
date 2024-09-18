import streamlit as st
import cv2
import numpy as np
from PIL import Image
import cap_analysis.app
import texture_analysis.app
import pill_color_detection.app
import seal.app


favicon = Image.open("CanPrev_4D-logo.png")
st.set_page_config(
    page_title="Canprev AI",  # Title of the web page
    page_icon=favicon,                # Favicon, use emoji or a file path
    layout="centered",                       # Layout options: "wide" or "centered"
    initial_sidebar_state="expanded"     # Initial state of the sidebar: "auto", "expanded", or "collapsed"
)
st.markdown("""
    <style>
       /* change the select box properties */
        div[data-baseweb="select"]>div {
        background-color:#fff;
        border-color:rgb(194, 189, 189);
        width: 100%;
    }

    /* change the tag font properties */
        span[data-baseweb="tag"]>span {
        color: black;
        font-size: 17px;
    }
    span.st-ae{
        background-color:  #FCF1C9 ;
    }
    
    .e1q9reml2 {
        color: #F4FAF3;
    }
    
    .st-fw p{
        padding: 0.3rem 0.4rem;
        border-radius: 5px;
        background-color: #6699cc;
        color: white;
    }
 
    </style>
    """, unsafe_allow_html=True)
# Main app function
def main():
    
    st.image('CanPrev_4D-logo.png', width=200)
    st.title("Canprev AI")
    
    # Tabs for different types of analysis
    tab1, tab2, tab3, tab4 = st.tabs(["Cap Detection", "Texture Analysis", "Pill Analysis", 'Open Bottle Seal Analysis'])
    
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
    
    with tab4:
        st.header('Open Bottle Seal Analysis')
        seal.app.main()

if __name__ == "__main__":
    main()