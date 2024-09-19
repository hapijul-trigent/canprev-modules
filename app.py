import streamlit as st
import cv2
import numpy as np
from PIL import Image
import cap_analysis.app
import texture_analysis.app
import pill_color_detection.app
import seal.app
import transparent_seal.app
import dropper_ointment.app
import cotton_tab
import dropper_ointment_tab
import powder_liquid_lumps_tab
import bottle_shoulder_tab

favicon = Image.open("CanPrev_4D-logo.png")
st.set_page_config(
    page_title="Canprev AI",  # Title of the web page
    page_icon=favicon,                # Favicon, use emoji or a file path
    layout="centered",                       # Layout options: "wide" or "centered"
    initial_sidebar_state="expanded"     # Initial state of the sidebar: "auto", "expanded", or "collapsed"
)
st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 3px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 40px;
        white-space: pre-wrap;
		background-color: #13276F;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding: 10px 2px 10px 2px;
        color: white;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
        color: #13276F;
        border: 2px solid #13276F;
        border-bottom: none;
	}

</style>""", unsafe_allow_html=True)
# Main app function
def main():
    
    st.image('CanPrev_4D-logo.png', width=200)
    st.title("Canprev AI")
    
    # Tabs for different types of analysis
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "Cap Detection", "Texture Analysis", "Pill Analysis", 
            'Open Bottle Seal', "Transparent Seal", "Dropper Ointment",
            "Cotton Detection", 'Powder-Liquid-Lumps', "Bottle Shoulder"
        ]
    )
    
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


    with tab5:
        st.header("Package Box Transparent Seal")
        transparent_seal.app.main()
    
    with tab6:
        st.header("Dropper Ointment")
        dropper_ointment_tab.panel()

    with tab7:
        st.header("Cotton Detection")
        cotton_tab.panel()
    
    with tab8:
        st.header("Powder-Liquid-Lumps")
        powder_liquid_lumps_tab.panel()

    with tab9:
        st.header("Bottle Shoulder")
        bottle_shoulder_tab.panel()

if __name__ == "__main__":
    main()