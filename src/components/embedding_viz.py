import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from config import (
    MODEL_INFO, METHOD_INFO, DEFAULT_MODEL, DEFAULT_METHOD,
    COLOR_MAP, 
    sample_chn_input_data, sample_enu_input_data
)
from models.model_manager import get_model
from utils.error_handling import handle_errors

from components.plotting import PlotManager

class EmbeddingVisualizer:
    def __init__(self):
        self.model_names = sorted(list(MODEL_INFO.keys()))
        self.method_names = sorted(list(METHOD_INFO.keys()))

    def render_sidebar(self) -> Tuple[str, str, str, bool, Optional[int]]:
        """Render sidebar controls and return settings"""
        with st.sidebar:
            st.header("Visualization Settings")
            st.write("Ollama model is slower")

            # Model selection
            model_name = st.radio(
                "Choose Embedding Model",
                options=self.model_names,
                index=self.model_names.index(DEFAULT_MODEL),
                help="Select a multilingual embedding model"
            )
            st.caption(f"**{model_name}**: {MODEL_INFO[model_name]['help']}")

            # Method selection
            method_name = st.radio(
                "Choose Dimensionality Reduction Method",
                options=self.method_names,
                index=self.method_names.index(DEFAULT_METHOD),
                help="Select a dimensionality reduction method"
            )
            st.caption(f"**{method_name}**: {METHOD_INFO[method_name]['help']}")

            # Dimensions
            dimensions = st.radio(
                "Choose Dimensions",
                options=["2D", "3D"],
                index=0
            )

            # Clustering
            do_clustering = st.checkbox("Enable Clustering?", value=False)
            n_clusters = None
            if do_clustering:
                n_clusters = st.slider("Number of Clusters", min_value=3, max_value=10, value=5)

            return model_name, method_name, dimensions, do_clustering, n_clusters

    @handle_errors
    def process_text(self, text: str) -> List[str]:
        """Process input text into list of words"""
        return [w.strip('"') for w in text.split() if w.strip('"')]

    def render_input_areas(self) -> Tuple[List[str], List[str], List[str]]:
        """Render text input areas and return processed words"""
        col1, col2, col_check12 = st.columns([5, 5, 1])
        
        with col1:
            chinese_text = st.text_area(
                "Chinese Words/Phrases:", 
                value=sample_chn_input_data,
                height=150)

        with col2:
            english_text = st.text_area(
                "English Words/Phrases:",
                value=sample_enu_input_data,
                height=150)
        with col_check12:
            st.write("Toggle Lang:")
            chinese_selected = st.checkbox("Chinese", value=True, key="chinese")
            english_selected = st.checkbox("English", value=True, key="english")

        chinese_words = self.process_text(chinese_text) if chinese_selected else []
        english_words = self.process_text(english_text) if english_selected else []
        
        return chinese_words, english_words, (
            [COLOR_MAP["chinese"]] * len(chinese_words) +
            [COLOR_MAP["english"]] * len(english_words)
        )
    
    @st.cache_data
    def get_embeddings(_self, words: List[str], model_name: str, lang: str) -> np.ndarray:
        """Get embeddings for words using specified model"""
        model = get_model(model_name)
        return model.get_embeddings(words, lang)

    def visualize(self):
        """Main visualization function"""
        st.title("Multilingual Word Embedding Explorer")
        
        # Get settings from sidebar
        model_name, method_name, dimensions, do_clustering, n_clusters = self.render_sidebar()
        
        # Get input words
        chinese_words, english_words, colors = self.render_input_areas()
        
        if st.button("Visualize"):
            if not (chinese_words or english_words):
                st.warning("Please enter at least one word or phrase.")
                return

            # Process embeddings
            embeddings = []
            if chinese_words:
                chinese_embeddings = self.get_embeddings(chinese_words, model_name, "zh")
                if chinese_embeddings is not None:
                    embeddings.append(chinese_embeddings)
            
            if english_words:
                english_embeddings = self.get_embeddings(english_words, model_name, "en")
                if english_embeddings is not None:
                    embeddings.append(english_embeddings)

            if not embeddings:
                st.error("Failed to generate embeddings.")
                return

            # Combine embeddings and create visualization
            combined_embeddings = np.vstack(embeddings)
            labels = chinese_words + english_words
            
            # Create visualization based on settings
            self.create_plot(
                combined_embeddings,
                labels,
                colors,
                model_name,
                method_name,
                dimensions,
                do_clustering,
                n_clusters
            )

    def create_plot(self, embeddings, labels, colors, model_name, method_name, 
                    dimensions, do_clustering, n_clusters):
        """Create and display the plot"""
        plot_title = f"[Model] {model_name}, [Method] {method_name}"
        plot_mgr = PlotManager()

        if dimensions == "2D":
            plot_mgr.plot_2d(
                embeddings=embeddings,
                labels=labels,
                colors=colors,
                title=plot_title,
                clustering=do_clustering,
                n_clusters=n_clusters if do_clustering else None
            )
        else:
            plot_mgr.plot_3d(
                embeddings=embeddings,
                labels=labels,
                colors=colors,
                title=plot_title,
                clustering=do_clustering,
                n_clusters=n_clusters if do_clustering else None
            )