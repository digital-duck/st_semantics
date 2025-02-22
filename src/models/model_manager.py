# models/model_manager.py
import streamlit as st
from abc import ABC, abstractmethod
import numpy as np
import requests
from typing import List, Optional
from utils.error_handling import (
    handle_errors, ModelNotFoundError, EmbeddingError,
)
from config import (
    OLLAMA_MODELS, MODEL_INFO,
)
from transformers import (
    AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel,
)
from laserembeddings import Laser

@st.cache_resource
def get_ollama_session():
    """Create a cached session for Ollama requests to improve performance"""
    session = requests.Session()
    return session

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    @abstractmethod
    def get_embeddings(self, texts: List[str], lang: str = "en") -> np.ndarray:
        pass

class OllamaModel(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.session = get_ollama_session()
        
    @handle_errors
    def get_embeddings(self, texts: List[str], lang: str = "en") -> Optional[np.ndarray]:
        embeddings = []
        for idx, text in enumerate(texts):
            response = self.session.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            if response.status_code == 200:
                embedding = response.json().get("embedding")
                if embedding:
                    embeddings.append(embedding)
            else:
                raise EmbeddingError(f"Failed to get embedding for text: {text}")
                    
        return np.array(embeddings) if embeddings else None

class HuggingFaceModel(EmbeddingModel):
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def _lazy_load(self):
        if not self.tokenizer:
            if self.model_name == "mT5":
                # Load the mT5 tokenizer and encoder model
                self.tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
                self.model = T5EncoderModel.from_pretrained("google/mt5-small")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path)
            
    @handle_errors
    def get_embeddings(self, texts: List[str], lang: str = "en") -> Optional[np.ndarray]:
        if self.model_name == "LASER":
            laser = Laser()
            return laser.embed_sentences(texts, lang=lang)
                  
        self._lazy_load()
        embeddings = []
        for text in texts:
            # Tokenize the input word
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            # Get the encoder outputs (no need for decoder inputs here)
            outputs = self.model(**inputs)
            # Use the last hidden state as the embedding
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
        return np.vstack(embeddings)

def get_model(model_name: str) -> EmbeddingModel:
    """Factory function for creating embedding models"""
    if model_name in OLLAMA_MODELS:
        return OllamaModel(OLLAMA_MODELS[model_name]["path"])
    elif model_name in MODEL_INFO:
        return HuggingFaceModel(model_name, MODEL_INFO[model_name]["path"])
    else:
        raise ModelNotFoundError(f"Model {model_name} not found")