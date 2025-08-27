import streamlit as st
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE, Isomap
from sklearn.decomposition import PCA, KernelPCA
from umap import UMAP
from phate import PHATE
from utils.error_handling import handle_errors

class DimensionReducer:
    def __init__(self):
        self.reducers = {
            "t-SNE": self._get_tsne,
            "Isomap": self._get_isomap,
            "UMAP": self._get_umap,
            "LLE": self._get_lle,
            "MDS": self._get_mds,
            "PCA": self._get_pca,
            "Kernel PCA": self._get_kernel_pca,
            "Spectral Embedding": self._get_spectral,
            "PHATE": self._get_phate
        }

    @handle_errors
    def reduce_dimensions(self, embeddings: np.ndarray, method: str, dimensions: int = 2) -> np.ndarray:
        """Reduce dimensions of embeddings using specified method"""
        n_samples = embeddings.shape[0]

        # Handle very small datasets
        if n_samples < 3:
            st.warning(f"Dataset too small for {method}. Using PCA instead.")
            return PCA(n_components=dimensions).fit_transform(embeddings)

        # Get appropriate reducer
        reducer = self.reducers[method](n_samples, dimensions)
        return reducer.fit_transform(embeddings)

    def _get_tsne(self, n_samples: int, dimensions: int):
        perplexity = min(30, n_samples - 1)
        return TSNE(n_components=dimensions, random_state=42, perplexity=perplexity)

    def _get_isomap(self, n_samples: int, dimensions: int):
        return Isomap(n_components=dimensions)

    def _get_umap(self, n_samples: int, dimensions: int):
        return UMAP(n_components=dimensions, random_state=42)

    def _get_lle(self, n_samples: int, dimensions: int):
        return LocallyLinearEmbedding(n_components=dimensions, random_state=42)

    def _get_mds(self, n_samples: int, dimensions: int):
        return MDS(n_components=dimensions, random_state=42)

    def _get_pca(self, n_samples: int, dimensions: int):
        return PCA(n_components=dimensions)

    def _get_kernel_pca(self, n_samples: int, dimensions: int):
        return KernelPCA(n_components=dimensions, kernel='rbf')

    def _get_spectral(self, n_samples: int, dimensions: int):
        return SpectralEmbedding(n_components=dimensions, random_state=42)

    def _get_phate(self, n_samples: int, dimensions: int):
        return PHATE(n_components=dimensions)