import streamlit as st
from PIL import Image
import numpy as np
from .tools import DinoV2Model, ImageProcessor, FaissIndexer, EmbeddingOrchestrator
import os
from .powder_liquid import get_rois, load_model
import warnings

warnings.filterwarnings(action='ignore')


def get_or_create_session_state_variable(key, default_value=None):
    """
    Retrieves the value of a variable from Streamlit's session state.
    If the variable doesn't exist, it creates it with the provided default value.

    Args:
        key (str): The key of the variable in session state.
        default_value (Any): The default value to assign if the variable doesn't exist.

    Returns:
        Any: The value of the session state variable.
    """
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]


get_or_create_session_state_variable(key='index_generated', default_value=False)

# Initialize components
roi_model = load_model()
dinoV2Model = DinoV2Model()
dinoV2Model.load_model()
image_processor = ImageProcessor(get_roi=get_rois, roi_model=roi_model)
faiss_indexer = FaissIndexer(embedding_dim=384)
# voyager_indexer = tools.VoyagerIndexer(embedding_dim=384)

# Load the FAISS index
index_file_path = 'texture_analysis/faiss_index.index'
voyager_index_path = 'voyager_index.voy'
image_paths = [f'texture_analysis/images/{image}' for image in os.listdir('texture_analysis/images')]

# Create Embedding Orchestrator
orchestrator = EmbeddingOrchestrator(dinoV2Model, image_processor, faiss_indexer)
# orchestrator_voyager = tools.EmbeddingOrchestrator(model, image_processor, voyager_indexer)
if not os.path.exists(index_file_path):
    orchestrator.process_images_and_create_index(image_paths, index_file_path=index_file_path)
# orchestrator.process_images_and_create_index(image_paths, voyager_index_path)
faiss_indexer.load_index(index_file_path)
# voyager_indexer.load_index(voyager_index_path)

def main():
    # Streamlit App
    st.write("Upload an image, and the app will find similar images based on texture.")

    # Image uploader
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        
        col1, col2 = st.columns([1, 1])

        with col1:
            image = Image.open(uploaded_image)
            st.subheader('Uploaded Image')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            image.save('texture_analysis/temp/temp.jpg')

        

        query_embedding = orchestrator.get_query_embedding(image_path='texture_analysis/temp/temp.jpg')

        # Perform the FAISS search
        k = 4 
        distances, indices = orchestrator.perform_search(query_embedding, k)


        with col2:
            st.write("Similar textures:")
            for i in range(0, 4, 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(indices[0]):
                        image_index = indices[0][i + j]
                        distance = distances[0][i + j]
                        image_index_path = image_paths[image_index]
                        similar_image_path = os.path.join("images/", f"{image_index_path}")
                        print(image_index)
                        similar_image_path = image_paths[image_index]

                        if distance >=0.2:
                            continue

                        if os.path.exists(similar_image_path):
                            similar_image = Image.open(similar_image_path)
                            cols[j].image(similar_image, caption=f"Distance: {distance:.4f}", use_column_width=True)
                        else:
                            cols[j].write(f"Image {similar_image_path} not found.")
