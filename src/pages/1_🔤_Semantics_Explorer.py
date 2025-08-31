import streamlit as st
import numpy as np
from components.embedding_viz import EmbeddingVisualizer
from components.dimension_reduction import DimensionReducer
from components.geometric_analysis import GeometricAnalyzer

from config import (
    check_login
)

# Page config
st.set_page_config(
    page_title="Semantics Explorer",
    page_icon="🔤",
    layout="wide"
)

@st.fragment 
def save_plot_image(visualizer, current_input, model_name, method_name, chinese_selected, english_selected):
    return visualizer.save_plot_image(current_input, model_name, method_name, chinese_selected, english_selected)

@st.fragment
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
    
    # Clear previous visualization data to prevent memory buildup
    if 'current_figure' in st.session_state:
        st.session_state.current_figure = None
        
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

def perform_geometric_analysis(analyzer, params):
    """Perform comprehensive geometric analysis on visualization data"""
    if 'visualization_data' not in st.session_state:
        st.error("No visualization data available for geometric analysis")
        return
    
    viz_data = st.session_state.visualization_data
    embeddings = viz_data['reduced_embeddings']
    labels = viz_data['labels']
    colors = viz_data.get('colors', [])
    
    # Store analysis results
    analysis_results = {}
    
    # Clustering Analysis
    if params.get('enable_clustering', False):
        clustering_results = analyzer.analyze_clustering(
            embeddings, 
            params['n_clusters'],
            params['density_radius'],
            labels
        )
        analysis_results['clustering'] = clustering_results
    
    # Branching Analysis
    if params.get('enable_branching', False):
        branching_results = analyzer.analyze_branching(
            embeddings,
            labels,
            params['connectivity_threshold']
        )
        analysis_results['branching'] = branching_results
    
    # Void Analysis
    if params.get('enable_void', False):
        void_results = analyzer.analyze_voids(
            embeddings,
            params['void_confidence']
        )
        analysis_results['void'] = void_results
    
    # Store results in session state
    st.session_state.geometric_analysis_results = analysis_results
    
    # Save metrics to files automatically
    try:
        # Get input name from session state
        input_name = st.session_state.get('cfg_input_text_entered', 'untitled')
        if not input_name or input_name == 'untitled':
            input_name = st.session_state.get('cfg_input_text_selected', 'sample_1')
        
        # Determine languages from visualization data
        languages = []
        
        # Check for unique language types in colors array
        unique_colors = set(colors) if colors else set()
        if 'chinese' in unique_colors:
            languages.append('chinese')
        if 'english' in unique_colors:
            languages.append('english')
        
        # Get model and method info
        model_name = viz_data.get('model_name', 'unknown-model')
        method_name = viz_data.get('method_name', 'unknown-method')
        
        # Save metrics
        save_json = params.get('save_json_files', False)
        saved_files = analyzer.save_metrics_to_files(
            analysis_results, input_name, model_name, method_name, languages, save_json
        )
        
        # Display save status
        analyzer.display_metrics_save_status(saved_files)
        
    except Exception as e:
        st.warning(f"Could not save metrics automatically: {str(e)}")
    
    # Display results
    display_geometric_analysis_results(analyzer, analysis_results, embeddings, labels)

def display_geometric_analysis_results(analyzer, results, embeddings, labels):
    """Display geometric analysis results in the UI"""
    with st.expander("🔬 Geometric Analysis Results", expanded=False):
        
        # Create tabs for different analysis types
        tabs = []
        tab_names = []
        
        if 'clustering' in results:
            tab_names.append("🔍 Clustering")
        if 'branching' in results:
            tab_names.append("🌿 Branching")
        if 'void' in results:
            tab_names.append("🕳️ Voids")
        if len(results) > 1:
            tab_names.append("📊 Summary")
        
        if tab_names:
            tabs = st.tabs(tab_names)
            
            tab_idx = 0
            
            # Clustering tab
            if 'clustering' in results:
                with tabs[tab_idx]:
                    analyzer.display_clustering_metrics(results['clustering'])
                tab_idx += 1
            
            # Branching tab
            if 'branching' in results:
                with tabs[tab_idx]:
                    analyzer.display_branching_metrics(results['branching'])
                tab_idx += 1
            
            # Void tab
            if 'void' in results:
                with tabs[tab_idx]:
                    analyzer.display_void_metrics(results['void'])
                tab_idx += 1
            
            # Summary tab (if multiple analyses were performed)
            if len(results) > 1 and tab_idx < len(tabs):
                with tabs[tab_idx]:
                    # Summary statistics first
                    st.subheader("📈 Key Metrics Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'clustering' in results and 'basic_metrics' in results['clustering']:
                            silhouette = results['clustering']['basic_metrics'].get('silhouette_score', 0)
                            st.metric("Clustering Quality", f"{silhouette:.3f}", 
                                    help="Silhouette score: measures how well-separated clusters are. Range [-1,1], >0.5 is good, >0.7 is excellent")
                    
                    with col2:
                        if 'branching' in results and 'linearity_scores' in results['branching']:
                            linearity = results['branching']['linearity_scores'].get('overall_linearity', 0)
                            st.metric("Overall Linearity", f"{linearity:.3f}",
                                    help="Principal component variance ratio: measures how linear/straight the data layout is. Higher values indicate more linear arrangement")
                    
                    with col3:
                        if 'void' in results and 'void_regions' in results['void']:
                            void_count = results['void']['void_regions'].get('num_voids', 0)
                            st.metric("Void Regions Found", void_count,
                                    help="Number of empty regions detected in the embedding space where no data points exist")
                    
                    st.markdown("---")
                    
                    # Clustering visualization only
                    st.subheader("🎯 Clustering Analysis Visualization")
                    
                    try:
                        # Create simplified clustering plot
                        clustering_fig = analyzer.create_comprehensive_analysis_plot(
                            embeddings, labels, 
                            results.get('clustering', {}),
                            results.get('branching', {}),
                            results.get('void', {})
                        )
                        st.plotly_chart(clustering_fig, use_container_width=True)
                        
                        # Save plot as PNG
                        try:
                            # Get input and model info for PNG filename
                            input_name = st.session_state.get('cfg_input_text_entered', 'untitled')
                            if not input_name or input_name == 'untitled':
                                input_name = st.session_state.get('cfg_input_text_selected', 'sample_1')
                            
                            viz_data = st.session_state.visualization_data
                            model_name = viz_data.get('model_name', 'unknown-model')
                            method_name = viz_data.get('method_name', 'unknown-method')
                            
                            png_filename = analyzer.save_summary_plot_as_png(clustering_fig, input_name, model_name, method_name)
                            if png_filename:
                                st.success(f"📸 Clustering visualization saved as: data/metrics/{png_filename}")
                            
                        except Exception as png_error:
                            st.warning(f"Could not save PNG: {str(png_error)}")
                        
                    except Exception as e:
                        st.error(f"Error creating clustering visualization: {str(e)}")

def main():
    # Check login status
    check_login()
    
    st.subheader(f"🔤 Explore Word/Phrase Embeddings in {st.session_state.get('cfg_vis_dimensions', '2D')} Spaces")
    # Initialize components
    visualizer = EmbeddingVisualizer()
    reducer = DimensionReducer()
    geometric_analyzer = GeometricAnalyzer()


    # Get settings from sidebar
    model_name, method_name, dimensions, do_clustering, n_clusters = visualizer.render_sidebar()
    
    # Add geometric analysis controls to sidebar
    with st.sidebar:
        with st.expander("🔬 Geometric Analysis", expanded=False):
            enable_geometric_analysis = st.checkbox(
                "Enable Geometric Analysis", 
                value=True,
                help="Perform clustering, branching, and void analysis"
            )
            
            if enable_geometric_analysis:
                analysis_params = geometric_analyzer.render_controls()
            else:
                analysis_params = None

    # Get input words
    btn_actions, chinese_words, english_words, colors, chinese_selected, english_selected = visualizer.render_input_areas()
    btn_visualize, btn_rotate_90, btn_save_png = btn_actions

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
        # visualizer.save_plot_image(current_input, model_name, method_name, chinese_selected, english_selected)
        file_png = save_plot_image(visualizer, current_input, model_name, method_name, chinese_selected, english_selected)
        if file_png:
            st.sidebar.success(f"Image saved as: {file_png}")
            st.image(f"data/images/{file_png}", caption=f"{file_png}", use_column_width=True)
        else:
            st.error("Failed to save image.")

    # Handle visualize button
    if btn_visualize:
        success = generate_visualization(visualizer, reducer, chinese_words, english_words, colors, model_name, method_name, dimensions, do_clustering, n_clusters)
        
        # Perform geometric analysis if enabled and visualization was successful
        if success and enable_geometric_analysis and analysis_params and 'visualization_data' in st.session_state:
            with st.spinner("🔬 Performing geometric analysis..."):
                perform_geometric_analysis(geometric_analyzer, analysis_params)



if __name__ == "__main__":
    main()