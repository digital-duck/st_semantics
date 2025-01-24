import streamlit as st
import numpy as np
from components.embedding_viz import EmbeddingVisualizer
from components.dimension_reduction import DimensionReducer

from config import (
    ST_HEADER_1,
    check_login
)

# # Page config
# st.set_page_config(
#     page_title="Semantics Explorer",
#     page_icon="🔤",
#     layout="wide"
# )

def main():
    # Check login status
    check_login()
    
    st.header(ST_HEADER_1)
    
    # Initialize components
    visualizer = EmbeddingVisualizer()
    reducer = DimensionReducer()
    
    # Get settings from sidebar
    model_name, method_name, dimensions, do_clustering, n_clusters = visualizer.render_sidebar()
    
    # Get input words
    chinese_words, english_words, colors = visualizer.render_input_areas()
    
    if st.button("Visualize"):
        if not (chinese_words or english_words):
            st.warning("Please enter at least one word or phrase.")
            return

        # Process embeddings
        chinese_embeddings = None
        english_embeddings = None
        
        if chinese_words:
            chinese_embeddings = visualizer.get_embeddings(chinese_words, model_name, "zh")
            
        if english_words:
            english_embeddings = visualizer.get_embeddings(english_words, model_name, "en")

        if chinese_embeddings is None and english_embeddings is None:
            st.error("Failed to generate embeddings.")
            return

        # Combine embeddings
        all_embeddings = []
        labels = []
        
        if chinese_embeddings is not None:
            all_embeddings.append(chinese_embeddings)
            labels.extend(chinese_words)
        if english_embeddings is not None:
            all_embeddings.append(english_embeddings)
            labels.extend(english_words)
            
        combined_embeddings = np.vstack(all_embeddings)
        
        # Reduce dimensions
        dims = 3 if dimensions == "3D" else 2
        reduced_embeddings = reducer.reduce_dimensions(
            combined_embeddings, 
            method=method_name, 
            dimensions=dims
        )
        
        if reduced_embeddings is None:
            return
            
        # Create visualization
        visualizer.create_plot(
            reduced_embeddings,
            labels,
            colors,
            model_name,
            method_name,
            dimensions,
            do_clustering,
            n_clusters
        )

if __name__ == "__main__":
    main()