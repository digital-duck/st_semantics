import streamlit as st
import numpy as np
import os
from components.embedding_viz import EmbeddingVisualizer
from components.dimension_reduction import DimensionReducer
from components.plotting_echarts import EChartsPlotManager
from components.geometric_analysis import GeometricAnalyzer

from config import (
    check_login,
    DEFAULT_N_CLUSTERS
)

# Page config
st.set_page_config(
    page_title="Semantics Explorer - ECharts",
    page_icon="📊",
    layout="wide"
)

class EChartsEmbeddingVisualizer(EmbeddingVisualizer):
    """Enhanced embedding visualizer using Apache ECharts for plotting"""

    def __init__(self):
        super().__init__()
        self.echarts_plot_manager = EChartsPlotManager()

    def create_plot(self, reduced_embeddings, labels, colors, model_name, method_name,
                   dimensions="2D", do_clustering=False, n_clusters=DEFAULT_N_CLUSTERS, dataset_name=""):
        """Create visualization using ECharts instead of Plotly"""

        # Create plot title
        title = f"ECharts Visualization: {method_name} | {model_name}"
        if dataset_name:
            title += f" | {dataset_name}"

        # Apply rotation if set
        if hasattr(st.session_state, 'plot_rotation') and st.session_state.plot_rotation != 0:
            if dimensions == "2D":
                # Apply 2D rotation
                angle = np.radians(st.session_state.plot_rotation)
                rotation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
                reduced_embeddings = reduced_embeddings @ rotation_matrix.T

        # Create appropriate plot based on dimensions
        if dimensions == "3D":
            plot_option = self.echarts_plot_manager.plot_3d(
                reduced_embeddings, labels, colors, title,
                clustering=do_clustering, n_clusters=n_clusters,
                method_name=method_name, model_name=model_name, dataset_name=dataset_name
            )
        else:
            plot_option = self.echarts_plot_manager.plot_2d(
                reduced_embeddings, labels, colors, title,
                clustering=do_clustering, n_clusters=n_clusters,
                method_name=method_name, model_name=model_name, dataset_name=dataset_name
            )

        # Store the chart configuration for potential export
        st.session_state.current_echarts_config = plot_option

        return plot_option

@st.fragment
def save_plot_image(visualizer, current_input, model_name, method_name, chinese_selected, english_selected):
    return visualizer.save_plot_image(current_input, model_name, method_name, chinese_selected, english_selected)

@st.fragment
def generate_visualization_echarts(visualizer, reducer, chinese_words, english_words, colors, model_name, method_name, dimensions, do_clustering, n_clusters):
    """Generate embeddings and ECharts visualization"""
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

    # Get dataset name from selected input
    dataset_name = st.session_state.get('input_name_selected', 'User Input')
    if not dataset_name:
        dataset_name = st.session_state.get('cfg_input_text_selected', 'User Input')

    # Store data in session state for rotation and analysis
    st.session_state.visualization_data = {
        'reduced_embeddings': reduced_embeddings,
        'labels': labels,
        'colors': colors,
        'model_name': model_name,
        'method_name': method_name,
        'dimensions': dimensions,
        'do_clustering': do_clustering,
        'n_clusters': n_clusters,
        'dataset_name': dataset_name
    }

    # Create ECharts visualization
    visualizer.create_plot(
        reduced_embeddings,
        labels,
        colors,
        model_name,
        method_name,
        dimensions,
        do_clustering,
        n_clusters,
        dataset_name
    )

    # Auto-save the visualization
    try:
        current_input = st.session_state.get('cfg_input_text_entered', 'untitled')
        if not current_input or current_input == 'untitled':
            current_input = st.session_state.get('input_name_selected', 'sample_1')
            if not current_input:
                current_input = st.session_state.get('cfg_input_text_selected', 'sample_1')

        # Determine language selections
        chinese_text = st.session_state.get('chn_text_area', '')
        english_text = st.session_state.get('enu_text_area', '')
        chinese_selected = bool(chinese_text.strip())
        english_selected = bool(english_text.strip())

        # Auto-save the plot (consolidated to reduce message clutter)
        saved_files = []
        auto_save_settings = st.session_state.get('echarts_auto_save', {'enabled': False})
        auto_save_status = visualizer.echarts_plot_manager.get_auto_save_status()

        if 'current_echarts_config' in st.session_state:
            filename_parts = [current_input, model_name, method_name]

            # Save ECharts configuration
            saved_config = visualizer.echarts_plot_manager.save_echarts_as_png(
                st.session_state.current_echarts_config,
                filename_parts,
                dimensions
            )
            if saved_config:
                saved_files.append(f"JSON: {saved_config}")

            # Auto-save PNG for 2D visualizations (if enabled and selenium available)
            if (auto_save_settings.get('enabled', False) and
                auto_save_status['available'] and
                dimensions == "2D"):
                with st.spinner("📸 Auto-saving PNG..."):
                    saved_png = visualizer.echarts_plot_manager.save_echarts_as_png_auto(
                        st.session_state.current_echarts_config,
                        filename_parts,
                        dimensions,
                        width=auto_save_settings.get('width', 1200),
                        height=auto_save_settings.get('height', 800)
                    )
                    if saved_png:
                        if isinstance(saved_png, dict):
                            saved_files.append(f"PNG: {saved_png['filename']}")
                            # Store the filepath for later display
                            st.session_state['last_echarts_png_path'] = saved_png['filepath']
                        else:
                            saved_files.append(f"PNG: {saved_png}")
                            # Try to construct filepath for older format
                            echarts_dir = os.path.join("src", "data", "images", "echarts")
                            st.session_state['last_echarts_png_path'] = os.path.join(echarts_dir, saved_png)
        else:
            # Fallback to regular PNG save if no ECharts config
            saved_filename = visualizer.save_plot_image(current_input, model_name, method_name, chinese_selected, english_selected, dimensions)
            if saved_filename:
                saved_files.append(f"PNG: {saved_filename}")

        # # Single consolidated success message
        # if saved_files:
        #     st.success(f"📊 **Auto-saved**: {' | '.join(saved_files)}")

        #     # Display the auto-saved PNG image in main panel if available
        #     if 'last_echarts_png_path' in st.session_state:
        #         png_path = st.session_state['last_echarts_png_path']
        #         if os.path.exists(png_path):
        #             st.image(png_path, caption="📊 Auto-saved ECharts Visualization", use_column_width=True)

    except Exception as auto_save_error:
        st.warning(f"Could not auto-save visualization: {str(auto_save_error)}")

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

    # Save metrics and display results
    try:
        input_name = st.session_state.get('cfg_input_text_entered', 'untitled')
        if not input_name or input_name == 'untitled':
            input_name = st.session_state.get('cfg_input_text_selected', 'sample_1')

        languages = []
        unique_colors = set(colors) if colors else set()
        if 'chinese' in unique_colors:
            languages.append('chinese')
        if 'english' in unique_colors:
            languages.append('english')

        model_name = viz_data.get('model_name', 'unknown-model')
        method_name = viz_data.get('method_name', 'unknown-method')

        save_json = params.get('save_json_files', False)
        saved_files = analyzer.save_metrics_to_files(
            analysis_results, input_name, model_name, method_name, languages, save_json
        )

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
            tab_names.append("=s Voids")
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

            # Summary tab
            if len(results) > 1 and tab_idx < len(tabs):
                with tabs[tab_idx]:
                    st.subheader("📈 Key Metrics Summary")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if 'clustering' in results and 'basic_metrics' in results['clustering']:
                            silhouette = results['clustering']['basic_metrics'].get('silhouette_score', 0)
                            st.metric("Clustering Quality", f"{silhouette:.3f}",
                                    help="Silhouette score: measures how well-separated clusters are.")

                    with col2:
                        if 'branching' in results and 'linearity_scores' in results['branching']:
                            linearity = results['branching']['linearity_scores'].get('overall_linearity', 0)
                            st.metric("Overall Linearity", f"{linearity:.3f}",
                                    help="Principal component variance ratio: measures linearity of data layout.")

                    with col3:
                        if 'void' in results and 'void_regions' in results['void']:
                            void_count = results['void']['void_regions'].get('num_voids', 0)
                            st.metric("Void Regions Found", void_count,
                                    help="Number of empty regions detected in the embedding space.")

def main():
    # Check login status
    check_login()

    st.subheader("📊 Semantics Explorer - Apache ECharts")


    # Initialize components
    visualizer = EChartsEmbeddingVisualizer()
    reducer = DimensionReducer()
    geometric_analyzer = GeometricAnalyzer()

    # Organize sidebar in logical sections
    with st.sidebar:
        # Core model and visualization settings
        model_name, method_name, dimensions, do_clustering, n_clusters = visualizer.render_sidebar()

        # st.markdown("---")  # Separator
        # st.markdown("## 🎨 ECharts Features")

        # ECharts-specific settings
        echarts_settings = visualizer.echarts_plot_manager.render_settings_controls()


        with st.expander("⚙️ PNG Export Settings", expanded=False):

            # Check selenium availability
            auto_save_status = visualizer.echarts_plot_manager.get_auto_save_status()
            # Main auto-save toggle - prominent and defaults to True
            if auto_save_status['available']:
                enable_auto_png = st.checkbox(
                    "📊 **Auto-save PNG images (Selenium)**",
                    value=True,  # Default to True for better UX
                    help="Automatically save high-quality PNG images for 2D visualizations using headless browser screenshot"
                )
                if enable_auto_png:
                    st.caption("✅ Auto-PNG export enabled for 2D charts")
                else:
                    st.caption("ℹ️ Auto-PNG export disabled - JSON only")
            else:
                enable_auto_png = st.checkbox(
                    "📊 **Auto-save PNG images (Selenium)**",
                    value=False,
                    disabled=True,
                    help="Install selenium and webdriver-manager to enable automatic PNG export"
                )
                st.warning("⚠️ Selenium not available. Install with: `pip install selenium webdriver-manager`")



            png_width = st.number_input(
                "Image Width (px)",
                min_value=600,
                max_value=2400,
                value=1200,
                step=100,
                help="Width of auto-saved PNG images (higher = better quality)"
            )

            png_height = st.number_input(
                "Image Height (px)",
                min_value=400,
                max_value=1600,
                value=800,
                step=100,
                help="Height of auto-saved PNG images (higher = better quality)"
            )

            # Quality preset buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📱 Mobile", help="800x600"):
                    png_width, png_height = 800, 600
            with col2:
                if st.button("💻 Desktop", help="1200x800"):
                    png_width, png_height = 1200, 800
            with col3:
                if st.button("📄 Print", help="1800x1200"):
                    png_width, png_height = 1800, 1200


        # Store settings in session state
        st.session_state.echarts_auto_save = {
            'enabled': enable_auto_png,
            'width': png_width,
            'height': png_height,
            'available': auto_save_status['available']
        }

        # Geometric analysis controls
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
    btn_actions, chinese_words, target_words_dict, colors, chinese_selected, target_selected_dict = visualizer.render_input_areas()
    btn_visualize, btn_rotate_90, btn_save_png = btn_actions

    # Combine all target language words for backward compatibility
    all_target_words = []
    for words in target_words_dict.values():
        all_target_words.extend(words)

    english_words = all_target_words
    english_selected = any(target_selected_dict.values())

    # Handle rotate button
    if btn_rotate_90:
        st.session_state.plot_rotation = (st.session_state.plot_rotation + 90) % 360
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
                viz_data['n_clusters'],
                viz_data.get('dataset_name', 'User Input')
            )
        else:
            st.warning("Please generate a visualization first by clicking 'Visualize'")

    # Handle save image button - now saves ECharts config
    if btn_save_png:
        current_input = st.session_state.get('cfg_input_text_selected', 'untitled')

        # Save ECharts configuration if available
        if 'current_echarts_config' in st.session_state:
            filename_parts = [current_input, model_name, method_name]
            saved_file = visualizer.echarts_plot_manager.save_echarts_as_png(
                st.session_state.current_echarts_config,
                filename_parts,
                dimensions
            )
            if saved_file:
                st.sidebar.success(f"📊 ECharts config saved: {saved_file}")
        else:
            # Fallback to regular PNG save if no ECharts config
            file_png = save_plot_image(visualizer, current_input, model_name, method_name, chinese_selected, english_selected)
            if file_png:
                st.sidebar.success(f"Image saved as: {file_png}")
                st.image(f"data/images/{file_png}", caption=f"{file_png}", width='stretch')
            else:
                st.error("Failed to save image.")

    # Handle visualize button
    if btn_visualize:
        success = generate_visualization_echarts(visualizer, reducer, chinese_words, english_words, colors, model_name, method_name, dimensions, do_clustering, n_clusters)

        # Perform geometric analysis if enabled and visualization was successful (only for 2D)
        if success and enable_geometric_analysis and analysis_params and 'visualization_data' in st.session_state:
            if dimensions == "2D":
                with st.spinner("🔬 Performing geometric analysis..."):
                    perform_geometric_analysis(geometric_analyzer, analysis_params)
            else:
                st.info("9 Geometric analysis is only available for 2D visualizations.")

    with st.sidebar:
        st.markdown("---")  # Separator
        

        # Add info about ECharts features
        with st.expander("ℹ️ About ECharts Features", expanded=False):
            st.markdown("""
            **Apache ECharts** brings enhanced interactivity and beautiful visualizations to semantic exploration:

            - **🎯 Enhanced Interactivity**: Smooth zoom, pan, and hover interactions
            - **🎨 Rich Visual Effects**: Beautiful animations and transitions
            - **📊 Advanced Clustering**: Dynamic cluster visualization with customizable boundaries
            - **< Network Analysis**: Semantic force visualization showing word relationships
            - **📱 Responsive Design**: Optimized for different screen sizes
            - **⚡ Performance**: Optimized rendering for large datasets

            **Note**: 3D visualizations use ECharts GL for enhanced 3D rendering capabilities.
            """)

        # Add ECharts usage tips
        with st.expander("💡 ECharts Interaction Tips", expanded=False):
            st.markdown("""
            **Mouse Controls:**
            - **Left Click + Drag**: Pan around the visualization
            - **Mouse Wheel**: Zoom in/out
            - **Hover**: See detailed information for each point
            - **Click Legend**: Toggle cluster visibility (in clustering mode)

            **3D Controls (3D mode only):**
            - **Left Click + Drag**: Rotate the 3D scene
            - **Right Click + Drag**: Pan the 3D scene
            - **Mouse Wheel**: Zoom in/out

            **Settings:**
            - Adjust text size, point size, and other visual properties in the sidebar
            - Toggle animations for smoother or faster rendering
            - Enable/disable grid lines for cleaner appearance
            """)


if __name__ == "__main__":
    main()