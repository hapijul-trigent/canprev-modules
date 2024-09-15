import streamlit as st
from PIL import Image
import numpy as np
import tools  # Import the module containing the logic for image processing and FAISS search
import os

# Initialize components
model = tools.DinoV2Model()
model.load_model()
image_processor = tools.ImageProcessor()
faiss_indexer = tools.FaissIndexer(embedding_dim=384)

# Load the FAISS index
index_file_path = 'faiss_index.index'
images_paths = [
    'FFP016XD.jpg',
    'EMP003HN.jpg',
    'FFP015XD.jpg',
    'EMF001HN.jpg',
    'EMD001HN.jpg',
    'GUP015XD.jpg',
    'GUP013XD.jpg',
    'ENS012CX.jpg',
    'FFP014XD.jpg',
    'ENS013CX.jpg',
    'GUP014XD.jpg'
]
image_paths = [f'images/{image}' for image in os.listdir('images')]
print(image_paths)

# Create Embedding Orchestrator
orchestrator = tools.EmbeddingOrchestrator(model, image_processor, faiss_indexer)
orchestrator.process_images_and_create_index(image_paths, 'faiss_index.index')
faiss_indexer.load_index(index_file_path)

# Streamlit App
st.title("Image Similarity Search")
st.write("Upload an image, and the app will find similar images based on texture.")

# Image uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_image)
        st.subheader('Uploaded Image')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image.save('temp/temp.jpg')

    

    query_embedding = orchestrator.get_query_embedding(image_path='temp/temp.jpg')

    # Perform the FAISS search
    k = 4 
    distances, indices = orchestrator.perform_search(query_embedding, k)


    with col2:
        st.write("Similar textures:")

        # Use a grid layout (2x2) for similar images
        for i in range(0, 4, 2):  # Loop through 2 images at a time
            cols = st.columns(2)  # Create two columns in each row
            for j in range(2):  # Display two images per row
                if i + j < len(indices[0]):
                    image_index = indices[0][i + j]
                    distance = distances[0][i + j]
                    image_index_path = images_paths[image_index]
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
