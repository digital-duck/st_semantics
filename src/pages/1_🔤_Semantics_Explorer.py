import streamlit as st
import numpy as np
from components.embedding_viz import EmbeddingVisualizer
from components.dimension_reduction import DimensionReducer

from config import (
    check_login
)

# Page config
st.set_page_config(
    page_title="Semantics Explorer",
    page_icon="🔤",
    layout="wide"
)

def generate_visualization(visualizer, reducer, chinese_words, english_words, colors, model_name, method_name, dimensions, do_clustering, n_clusters):
    """Generate embeddings and visualization"""
    if not (chinese_words or english_words):
        st.warning("Please enter at least one word or phrase.")
        return False

    # Process embeddings
    chinese_embeddings = None
    english_embeddings = None
    
    if chinese_words:
        chinese_embeddings = visualizer.get_embeddings(chinese_words, model_name, "zh")
        
    if english_words:
        english_embeddings = visualizer.get_embeddings(english_words, model_name, "en")

    if chinese_embeddings is None and english_embeddings is None:
        st.error("Failed to generate embeddings.")
        return False

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
        return False
        
    # Store data in session state for rotation
    st.session_state.visualization_data = {
        'reduced_embeddings': reduced_embeddings,
        'labels': labels,
        'colors': colors,
        'model_name': model_name,
        'method_name': method_name,
        'dimensions': dimensions,
        'do_clustering': do_clustering,
        'n_clusters': n_clusters
    }
    
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
    
    return True

def main():
    # Check login status
    check_login()
    
    st.subheader(f"View word/phrase embeddings in {st.session_state.get('cfg_vis_dimensions', '2D')} spaces")
    # Initialize components
    visualizer = EmbeddingVisualizer()
    reducer = DimensionReducer()

    # Get input words
    btn_actions, chinese_words, english_words, colors, chinese_selected, english_selected = visualizer.render_input_areas()
    btn_visualize, btn_rotate_90, btn_save_png = btn_actions

    # Get settings from sidebar
    model_name, method_name, dimensions, do_clustering, n_clusters = visualizer.render_sidebar()
    
    # Handle rotate button - reuse existing visualization data
    if btn_rotate_90:
        st.session_state.plot_rotation = (st.session_state.plot_rotation + 90) % 360
        # If we have existing visualization data, redraw with rotation
        if 'visualization_data' in st.session_state:
            viz_data = st.session_state.visualization_data
            visualizer.create_plot(
                viz_data['reduced_embeddings'],
                viz_data['labels'],
                viz_data['colors'],
                viz_data['model_name'],
                viz_data['method_name'],
                viz_data['dimensions'],
                viz_data['do_clustering'],
                viz_data['n_clusters']
            )
        else:
            st.warning("Please generate a visualization first by clicking 'Visualize'")
    
    # Handle save image button
    if btn_save_png:
        # Get current input name
        current_input = st.session_state.get('cfg_input_text_selected', 'untitled')
        visualizer.save_plot_image(current_input, model_name, method_name, chinese_selected, english_selected)
    
    if btn_visualize:
        generate_visualization(visualizer, reducer, chinese_words, english_words, colors, model_name, method_name, dimensions, do_clustering, n_clusters)

if __name__ == "__main__":
    main()