# models/model_manager.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

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
        with st.progress(0) as progress_bar:
            for idx, text in enumerate(texts):
                response = self.session.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                )
                if response.status_code == 200:
                    embedding = response.json().get("embedding")
                    if embedding:
                        embeddings.append(embedding)
                    progress_bar.progress((idx + 1) / len(texts))
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            
    @handle_errors
    def get_embeddings(self, texts: List[str], lang: str = "en") -> Optional[np.ndarray]:
        self._lazy_load()
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
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