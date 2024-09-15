import torch
from torchvision import transforms
from PIL import Image
import faiss
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoImageProcessor, AutoModel
import voyager
from typing import Union
import streamlit as st

class BaseModel(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_embedding(self, image_tensor):
        pass


class DinoV2Model(BaseModel):
    
    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device)

    def get_embedding(self, image_tensor):
        with torch.no_grad():
            embedding = self.model(**image_tensor).last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding


class ImageProcessor:
    def __init__(self, get_roi, roi_model):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.get_roi = get_roi
        self.roi_model = roi_model
        
    def load_and_preprocess_image(self, image_path):
        
        image_np = Image.open(image_path).convert('RGB')
        image_np = np.array(image_np)
        
        rois, _ = self.get_roi(image_np, self.roi_model)
        if len(rois)==0:
            return self.processor(images=image_np, return_tensors="pt").to(self.device)
        
        return self.processor(images=rois[0], return_tensors="pt").to(self.device)



class FaissIndexer:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        self.index.add(embeddings)

    def save_index(self, index_file_path):
        faiss.write_index(self.index, index_file_path)

    def load_index(self, index_file_path):
        self.index = faiss.read_index(index_file_path)

    def search(self, query_embedding, k=1):
        D, I = self.index.search(query_embedding, k)
        return D, I


class VoyagerIndexer:
    def __init__(self, embedding_dim, M=12, ef_construction=200, max_elements=1000, space=voyager.Space.Cosine, storage_type=voyager.StorageDataType.Float32):
        self.embedding_dim = embedding_dim
        self.space = space
        self.storage_type = storage_type
        self.index = voyager.Index(space, embedding_dim, M, ef_construction, max_elements, storage_type)

    def add_embeddings(self, embeddings):
        for i, embedding in enumerate(embeddings):
            self.index.add(i, embedding)

    def build(self):
        self.index.build()

    def save_index(self, index_file_path):
        self.index.save(index_file_path)

    def load_index(self, index_file_path):
        # Recreate the index with the necessary parameters
        self.index = voyager.Index(self.space, self.embedding_dim, M=12, ef_construction=200, max_elements=10000, storage_data_type=self.storage_type)
        self.index.load(index_file_path)

    def search(self, query_embedding, k=1):
        return self.index.search(query_embedding, k)





class EmbeddingOrchestrator:
    def __init__(self, model: BaseModel, image_processor: ImageProcessor, indexer: Union[FaissIndexer, VoyagerIndexer]):
        self.model = model
        self.image_processor = image_processor
        self.indexer = indexer

    def process_images_and_create_index(self, image_paths, index_file_path):
        embeddings = []
        for image_path in image_paths:
            
            image_tensor = self.image_processor.load_and_preprocess_image(image_path)
            
            # Extract & Normalize Embeddings
            embedding = self.model.get_embedding(image_tensor)
            embedding = np.float32(embedding)
            if isinstance(self.indexer, FaissIndexer):
                faiss.normalize_L2(embedding)
            embeddings.append(embedding)

        
        all_embeddings_np = np.vstack(embeddings)
        
        # Add embeddings to FAISS index and save
        self.indexer.add_embeddings(all_embeddings_np)
        self.indexer.save_index(index_file_path)

        return all_embeddings_np
    
    def get_query_embedding(self, image_path):
      
        image_tensor = self.image_processor.load_and_preprocess_image(image_path=image_path)
        embedding = self.model.get_embedding(image_tensor)
        embedding = np.float32(embedding)
        if isinstance(self.indexer, FaissIndexer):
            faiss.normalize_L2(embedding)
      
        return embedding

    def perform_search(self, query_embedding, k=1):
        return self.indexer.search(query_embedding, k)


if __name__ == "__main__":
    
    image_paths = ['/content/ABP008XD.jpg', '/content/CCR004.jpg']

    
    image_processor = ImageProcessor()
    model = DinoV2Model()
    model.load_model()
    faiss_indexer = FaissIndexer(embedding_dim=384)

    
    orchestrator = EmbeddingOrchestrator(model, image_processor, faiss_indexer)

    
    embeddings = orchestrator.process_images_and_create_index(image_paths, 'faiss_index.index')

    
    faiss_indexer.load_index('faiss_index.index')
    query_embedding = embeddings[1:2]
    distances, indices = orchestrator.perform_search(query_embedding, k=1)

    print("Distances:", distances)
    print("Indices:", indices)
