import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.embedding_viz import EmbeddingVisualizer
from components.dimension_reduction import DimensionReducer
from components.geometric_analysis import GeometricAnalyzer
from config import (
    check_login,
    PLOT_WIDTH,
    PLOT_HEIGHT,
    COLOR_MAP,
    MODEL_INFO,
    METHOD_INFO,
    DEFAULT_MODEL,
    DEFAULT_METHOD,
    sample_chn_input_data,
    sample_enu_input_data
)
from pathlib import Path
from utils.download_helpers import handle_download_button
from components.shared.publication_settings import PublicationSettingsWidget

# Page config
st.set_page_config(
    page_title="Dual Viewer",
    page_icon="🔍",
    layout="wide"
)

DEFAULT_STEP_SIZE = 0.005

class EnhancedDualViewManager:
    """Enhanced dual-view with center/size based zoom controls"""
    
    def __init__(self):
        self.overview_config = {
            'marker_size': 6,
            'opacity': 0.8,
            'show_text': False
        }
        self.detail_config = {
            'marker_size': 16,
            'opacity': 1.0,
            'show_text': True,
            'textfont_size': 16
        }

    def center_size_to_bounds(self, center_x, center_y, width, height):
        """Convert center/size to min/max bounds"""
        return {
            'x_min': center_x - width/2,
            'x_max': center_x + width/2,
            'y_min': center_y - height/2,
            'y_max': center_y + height/2
        }

    def create_enhanced_dual_view(self, embeddings, labels, colors, title, zoom_params, model_name=None, method_name=None, dataset_name=None):
        """Create separate overview and detail figures for the enhanced dual view"""
        
        # Convert center/size to bounds
        viewport_coords = self.center_size_to_bounds(
            zoom_params['center_x'], zoom_params['center_y'],
            zoom_params['width'], zoom_params['height']
        )
        
        # Separate data by language
        chinese_mask = np.array([color == 'chinese' for color in colors])
        english_mask = np.array([color == 'english' for color in colors])
        
        # Create overview figure
        overview_fig = go.Figure()
        
        # OVERVIEW - All points, small markers
        if np.any(chinese_mask):
            overview_fig.add_trace(go.Scatter(
                x=embeddings[chinese_mask, 0],
                y=embeddings[chinese_mask, 1],
                mode='markers',
                marker=dict(size=6, color='red', opacity=0.8, line=dict(width=1, color='white')),
                text=[labels[i] for i in range(len(labels)) if chinese_mask[i]],
                hovertemplate='<b>%{text}</b><br>中文<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                name="中文",
                showlegend=False
            ))
        
        if np.any(english_mask):
            overview_fig.add_trace(go.Scatter(
                x=embeddings[english_mask, 0],
                y=embeddings[english_mask, 1],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.8, line=dict(width=1, color='white')),
                text=[labels[i] for i in range(len(labels)) if english_mask[i]],
                hovertemplate='<b>%{text}</b><br>English<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                name="English",
                showlegend=False
            ))
        
        # Add zoom box to overview
        overview_fig.add_shape(
            type="rect",
            x0=viewport_coords['x_min'], y0=viewport_coords['y_min'],
            x1=viewport_coords['x_max'], y1=viewport_coords['y_max'],
            line=dict(color="orange", width=3),
            fillcolor="rgba(255, 165, 0, 0.2)"
        )
        
        # Set overview axis ranges
        data_x_min, data_x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
        data_y_min, data_y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
        x_padding = (data_x_max - data_x_min) * 0.1
        y_padding = (data_y_max - data_y_min) * 0.1
        
        overview_fig.update_xaxes(
            range=[data_x_min - x_padding, data_x_max + x_padding],
            title_text="x",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot'
        )
        overview_fig.update_yaxes(
            range=[data_y_min - y_padding, data_y_max + y_padding],
            title_text="y",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot',
            scaleanchor="x", scaleratio=1
        )
        
        # Create overview title with standardized format
        title_parts = []
        if method_name:
            title_parts.append(f"[Method] {method_name}")
        if model_name:
            title_parts.append(f"[Model] {model_name}")
        if dataset_name:
            title_parts.append(f"[Dataset] {dataset_name}")
        overview_title = ", ".join(title_parts) if title_parts else "Overview"
        
        overview_fig.update_layout(
            title=dict(
                text=overview_title,
                font=dict(size=18, family='Arial, sans-serif'),
                x=0.5,  # Center align title
                xanchor='center'
            ),
            dragmode='pan',
            hovermode='closest',
            showlegend=False,
            height=700,  # More square aspect ratio
            width=800,   # Controlled width to reduce white space
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Create detail figure
        detail_fig = go.Figure()
        
        # DETAIL - Only points within viewport
        viewport_mask = (
            (embeddings[:, 0] >= viewport_coords['x_min']) &
            (embeddings[:, 0] <= viewport_coords['x_max']) &
            (embeddings[:, 1] >= viewport_coords['y_min']) &
            (embeddings[:, 1] <= viewport_coords['y_max'])
        )
        
        chinese_viewport = chinese_mask & viewport_mask
        english_viewport = english_mask & viewport_mask
        
        if np.any(chinese_viewport):
            detail_fig.add_trace(go.Scatter(
                x=embeddings[chinese_viewport, 0],
                y=embeddings[chinese_viewport, 1],
                mode='markers+text',
                marker=dict(size=16, color='red', opacity=1.0, line=dict(width=2, color='white')),
                text=[labels[i] for i in range(len(labels)) if chinese_viewport[i]],
                textposition="top center",
                textfont=dict(size=16, color='red', family='Arial Black'),
                hovertemplate='<b>%{text}</b><br>中文 (Detail)<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                name="中文",
                showlegend=False
            ))
        
        if np.any(english_viewport):
            detail_fig.add_trace(go.Scatter(
                x=embeddings[english_viewport, 0],
                y=embeddings[english_viewport, 1],
                mode='markers+text',
                marker=dict(size=16, color='blue', opacity=1.0, line=dict(width=2, color='white')),
                text=[labels[i] for i in range(len(labels)) if english_viewport[i]],
                textposition="top center",
                textfont=dict(size=16, color='blue', family='Arial Black'),
                hovertemplate='<b>%{text}</b><br>English (Detail)<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                name="English",
                showlegend=False
            ))
        
        # Detail: Viewport range with 10% margin
        x_range = viewport_coords['x_max'] - viewport_coords['x_min']
        y_range = viewport_coords['y_max'] - viewport_coords['y_min']
        x_margin = x_range * 0.1
        y_margin = y_range * 0.1
        
        detail_fig.update_xaxes(
            range=[viewport_coords['x_min'] - x_margin, viewport_coords['x_max'] + x_margin],
            title_text="x",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot'
        )
        detail_fig.update_yaxes(
            range=[viewport_coords['y_min'] - y_margin, viewport_coords['y_max'] + y_margin],
            title_text="y",
            showgrid=True, 
            gridwidth=1,
            gridcolor='#D0D0D0',
            griddash='dot',
            scaleanchor="x", scaleratio=1
        )
        
        # Create detail view title with standardized format
        title_parts = []
        if method_name:
            title_parts.append(f"[Method] {method_name}")
        if model_name:
            title_parts.append(f"[Model] {model_name}")
        if dataset_name:
            title_parts.append(f"[Dataset] {dataset_name}")
        detail_title = ", ".join(title_parts) if title_parts else "Detail View"
        
        detail_fig.update_layout(
            title=dict(
                text=detail_title,
                font=dict(size=18, family='Arial, sans-serif'),
                x=0.5,  # Center align title
                xanchor='center'
            ),
            dragmode='pan',
            hovermode='closest',
            showlegend=False,
            height=900,
            plot_bgcolor='white',
            font=dict(family='Arial, sans-serif'),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Count points in viewport
        points_in_viewport = viewport_mask.sum()
        
        return overview_fig, detail_fig, points_in_viewport, viewport_mask

def perform_dual_view_geometric_analysis(analyzer, params, embeddings, labels, model_name=None, method_name=None):
    """Perform comprehensive geometric analysis for dual view"""
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
    st.session_state.dual_view_geometric_analysis = analysis_results
    
    # Save metrics to files automatically
    try:
        # Get input name from session state
        input_name = st.session_state.get('cfg_input_text_entered', 'untitled')
        if not input_name or input_name == 'untitled':
            input_name = st.session_state.get('cfg_input_text_selected', 'dual_view')
        
        # Determine languages from enhanced data
        if 'enhanced_data' in st.session_state:
            enhanced_data = st.session_state.enhanced_data
            colors = enhanced_data.get('colors', [])
            languages = []
            if 'chinese' in colors:
                languages.append('chinese')
            if 'english' in colors:
                languages.append('english')
        else:
            languages = ['unknown']
        
        # Use provided model and method names, or fallback to defaults
        if model_name is None:
            model_name = 'dual-view-model'
        if method_name is None:
            method_name = 'dual-view-method'
        
        # Save metrics
        save_json = params.get('save_json_files', False)
        saved_files = analyzer.save_metrics_to_files(
            analysis_results, input_name, model_name, method_name, languages, save_json
        )
        
        # Display save status
        analyzer.display_metrics_save_status(saved_files)
        
    except Exception as e:
        st.warning(f"Could not save dual view metrics automatically: {str(e)}")

def display_dual_view_geometric_analysis(model_name=None, method_name=None):
    """Display geometric analysis results for dual view"""
    if 'dual_view_geometric_analysis' not in st.session_state:
        return
    
    results = st.session_state.dual_view_geometric_analysis
    
    if not results:
        return
    
    with st.expander("🔬 Geometric Analysis Results - Dual View", expanded=False):
        
        # Display analysis results without nested expanders
        from components.geometric_analysis import GeometricAnalyzer
        analyzer = GeometricAnalyzer()
        
        if 'clustering' in results:
            st.subheader("🔍 Clustering Analysis")
            analyzer.display_clustering_metrics(results['clustering'])
        
        if 'branching' in results:
            st.subheader("🌿 Branching Analysis")
            analyzer.display_branching_metrics(results['branching'])
        
        if 'void' in results:
            st.subheader("🕳️ Void Analysis")
            analyzer.display_void_metrics(results['void'])
        
        # Summary visualization if multiple analyses exist
        if len(results) > 1 and 'enhanced_data' in st.session_state:
            st.subheader("📊 Comprehensive Analysis Visualization")
            try:
                from components.geometric_analysis import GeometricAnalyzer
                analyzer = GeometricAnalyzer()
                
                enhanced_data = st.session_state.enhanced_data
                embeddings = enhanced_data['embeddings']
                labels = enhanced_data['labels']
                
                # Get dataset information for consistent title
                dataset_name = st.session_state.get('cfg_input_text_selected', 'User Input')
                
                comprehensive_fig = analyzer.create_comprehensive_analysis_plot(
                    embeddings, labels,
                    results.get('clustering', {}),
                    results.get('branching', {}),
                    results.get('void', {}),
                    model_name, method_name, dataset_name
                )
                st.plotly_chart(comprehensive_fig, use_container_width=True)
                
                # Auto-save the clustering chart (saves user from having to click download)
                try:
                    # Get current input name for filename
                    current_input = st.session_state.get('cfg_input_text_entered', 'untitled')
                    if not current_input or current_input == 'untitled':
                        current_input = st.session_state.get('cfg_input_text_selected', 'sample_1')
                    
                    # Get publication settings for proper formatting
                    pub_settings = st.session_state.get('dual_view_publication_settings', {})
                    textfont_size = pub_settings.get('textfont_size', 16)
                    point_size = pub_settings.get('point_size', 12)
                    export_dpi = pub_settings.get('export_dpi', 300)
                    export_format = pub_settings.get('export_format', 'PNG').lower()
                    
                    # Create auto-save filename with clustering suffix
                    clean_method = (method_name or "unknown-method").lower().replace(" ", "-").replace(",", "").replace("_", "-")
                    clean_model = (model_name or "unknown-model").lower().replace(" ", "-").replace(",", "").replace("_", "-")
                    clean_dataset = dataset_name.lower().replace(" ", "-").replace(",", "").replace("_", "-")
                    clean_input = current_input.lower().replace(" ", "-").replace(",", "").replace("_", "-")
                    
                    clustering_filename = f"{clean_method}-{clean_model}-{clean_dataset}-{clean_input}-dpi-{export_dpi}-text-{textfont_size}-point-{point_size}-clustering-auto.{export_format}"
                    
                    # Auto-save the clustering chart
                    from pathlib import Path
                    images_dir = Path("data/images")
                    images_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save with high quality
                    img_bytes = comprehensive_fig.to_image(format=export_format, width=800, height=700, scale=export_dpi/96)
                    
                    # Write to file
                    clustering_path = images_dir / clustering_filename
                    clustering_path.write_bytes(img_bytes)
                    
                    st.success(f"📸 Clustering chart auto-saved as: {clustering_filename}")
                    
                except Exception as auto_save_error:
                    st.warning(f"Could not auto-save clustering chart: {str(auto_save_error)}")
                
                # Add download button for clustering chart
                handle_download_button(comprehensive_fig, model_name, method_name, dataset_name, "clustering", "dual_view")
                    
            except Exception as e:
                st.error(f"Error creating comprehensive analysis plot: {str(e)}")

def setup_sidebar_controls():
    """Setup sidebar controls and return settings"""
    settings = {}
    
    with st.sidebar:
        st.header("🎛️ Settings")
        
        # Model and method selection
        with st.expander("Visualization Settings", expanded=False):
            settings['model_name'] = st.selectbox(
                "Embedding Model:",
                options=list(MODEL_INFO.keys()),
                index=list(MODEL_INFO.keys()).index(DEFAULT_MODEL)
            )
            
            settings['method_name'] = st.selectbox(
                "Reduction Method:",
                options=list(METHOD_INFO.keys()),
                index=list(METHOD_INFO.keys()).index(DEFAULT_METHOD)
            )
        
        # Publication Settings (using shared component)
        publication_settings = PublicationSettingsWidget.render_publication_settings("dual_view")
        
        # Store settings in session state for backward compatibility
        st.session_state.dual_view_publication_settings = publication_settings
        settings['publication_settings'] = publication_settings
        
        return settings


def handle_text_input():
    """Handle text input UI and return processed text data"""
    with st.sidebar:
        # Text input areas - same as Semantics Explorer
        with st.expander("Enter Text Data (Word/Phrase):", expanded=False):
            input_dir = Path("data/input")
            
            # File loading section
            col_input_select, col_load_txt = st.columns([3, 1])
            with col_input_select:
                available_inputs = []
                if input_dir.exists():
                    input_names = set()
                    for file_path in input_dir.glob("*.txt"):
                        name_part = file_path.stem.replace("-chn", "").replace("-enu", "")
                        input_names.add(name_part)
                    available_inputs = sorted(list(input_names)) if input_names else ["sample_1"]
                else:
                    available_inputs = ["sample_1"]
                
                input_name_selected = st.selectbox(
                    "Select Input",
                    options=[""] + available_inputs,
                    index=0,
                    key="cfg_input_text_selected"
                )
            with col_load_txt:
                btn_load_txt = st.button("Load Text", type="primary", 
                                        help="Load input texts", 
                                        disabled=not input_name_selected)

            # Initialize text areas with default or loaded content
            default_chinese = sample_chn_input_data
            default_english = sample_enu_input_data
            
            # Load text if button is clicked
            if btn_load_txt:
                chn_file = input_dir / f"{input_name_selected}-chn.txt"
                enu_file = input_dir / f"{input_name_selected}-enu.txt"
                
                if chn_file.exists():
                    try:
                        loaded_chinese = chn_file.read_text(encoding='utf-8').strip()
                        default_chinese = loaded_chinese
                        st.session_state.chinese_text_area = loaded_chinese
                    except Exception as e:
                        st.error(f"Error reading Chinese file: {e}")
                
                if enu_file.exists():
                    try:
                        loaded_english = enu_file.read_text(encoding='utf-8').strip()
                        default_english = loaded_english
                        st.session_state.english_text_area = loaded_english
                    except Exception as e:
                        st.error(f"Error reading English file: {e}")
                
                if not chn_file.exists() and not enu_file.exists():
                    st.warning(f"No text files found for '{input_name_selected}'")

            col1, col2 = st.columns(2)
            with col1:
                chinese_text = st.text_area(
                    "Chinese:", 
                    value=st.session_state.get('chinese_text_area', default_chinese),
                    height=200,
                    key='chinese_text_input'
                )
                chinese_selected = st.checkbox("Chinese", value=True, key="chinese")

            with col2:
                english_text = st.text_area(
                    "English:",
                    value=st.session_state.get('english_text_area', default_english),
                    height=200,
                    key='english_text_input'
                )
                english_selected = st.checkbox("English", value=True, key="english")

            # Process text into word lists
            visualizer = EmbeddingVisualizer()
            chinese_words = visualizer.process_text(chinese_text) if chinese_selected else []
            english_words = visualizer.process_text(english_text) if english_selected else []
            
            # Save functionality
            col_input_enter, col_save_txt = st.columns([3, 1])
            with col_input_enter:
                input_name_raw = st.text_input(
                    "Name Input",
                    value=input_name_selected if input_name_selected else "untitled",
                    key="cfg_input_text_entered",
                    help="Name will be automatically sanitized for filename compatibility"
                )
                # Show sanitized preview
                sanitized_preview = visualizer.sanitize_filename(input_name_raw)
                if sanitized_preview != input_name_raw.lower():
                    st.caption(f"📝 Preview: `{sanitized_preview}`")
                    
            with col_save_txt:
                btn_save_txt = st.button("Save Text", type="primary", 
                                        help="Save input texts", 
                                        disabled=(input_name_raw=="untitled"))
                
            # Handle save text
            if btn_save_txt:
                input_dir.mkdir(parents=True, exist_ok=True)
                safe_name = visualizer.sanitize_filename(input_name_raw)
                success_count = 0
                
                if chinese_selected and chinese_text.strip():
                    chn_file = input_dir / f"{safe_name}-chn.txt"
                    try:
                        chn_file.write_text(chinese_text.strip(), encoding='utf-8')
                        success_count += 1
                    except Exception as e:
                        st.error(f"Error saving Chinese text: {e}")
                
                if english_selected and english_text.strip():
                    enu_file = input_dir / f"{safe_name}-enu.txt"
                    try:
                        enu_file.write_text(english_text.strip(), encoding='utf-8')
                        success_count += 1
                    except Exception as e:
                        st.error(f"Error saving English text: {e}")
                
                if success_count > 0:
                    st.success(f"Saved {success_count} text file(s) as '{safe_name}'")
                    st.rerun()
                else:
                    st.warning("No text to save")

            return chinese_words, english_words, chinese_selected, english_selected


def setup_geometric_analysis_controls():
    """Setup geometric analysis controls and return parameters"""
    with st.sidebar:
        # Geometric Analysis Controls
        with st.expander("🔬 Geometric Analysis", expanded=False):
            enable_geometric_analysis = st.checkbox(
                "Enable Geometric Analysis", 
                value=True,
                help="Perform clustering, branching, and void analysis on dual view data"
            )
            
            if enable_geometric_analysis:
                geometric_analyzer = GeometricAnalyzer()
                analysis_params = geometric_analyzer.render_controls()
            else:
                analysis_params = None
                
            return enable_geometric_analysis, analysis_params


def setup_zoom_controls():
    """Setup zoom controls and return zoom parameters"""  
    with st.sidebar:
        # Zoom controls
        with st.expander("Zoom Controls", expanded=True):
            # Global box size parameters
            global_width = 0.8
            global_height = 0.3
            
            # Initialize default zoom parameters
            if 'zoom_params' not in st.session_state:
                st.session_state.zoom_params = {
                    'center_x': 0.0, 'center_y': 0.0,
                    'width': 0.05, 'height': 0.05,  # Default zoom box size
                    'delta_x': 0.005, 'delta_y': 0.005  # Fixed panning step
                }
            
            zoom_params = st.session_state.zoom_params

            st.write("**Box Center:**")
            col1, col2 = st.columns(2)
            with col1:
                center_x = st.number_input("Center X", value=zoom_params['center_x'], step=DEFAULT_STEP_SIZE, format="%.3f")
            with col2:
                center_y = st.number_input("Center Y", value=zoom_params['center_y'], step=DEFAULT_STEP_SIZE, format="%.3f")
            
            st.write("**Box Size:**")
            col3, col4 = st.columns(2)
            with col3:
                width = st.number_input("Width", value=zoom_params['width'], step=DEFAULT_STEP_SIZE, format="%.3f")
            with col4:
                height = st.number_input("Height", value=zoom_params['height'], step=DEFAULT_STEP_SIZE, format="%.3f")
            
            st.write("**Panning Step:**")
            col5, col6 = st.columns(2)
            with col5:
                delta_x = st.number_input("Delta X", value=zoom_params['delta_x'], step=DEFAULT_STEP_SIZE, format="%.3f")
            with col6:
                delta_y = st.number_input("Delta Y", value=zoom_params['delta_y'], step=DEFAULT_STEP_SIZE, format="%.3f")
            
            col_update, col_reset = st.columns(2)
            with col_update:
                if st.button("🔄 Update Zoom", type="primary"):
                    # Apply pan movement to center
                    new_center_x = center_x + delta_x
                    new_center_y = center_y + delta_y
                    
                    st.session_state.zoom_params = {
                        'center_x': new_center_x, 'center_y': new_center_y,
                        'width': width, 'height': height,
                        'delta_x': delta_x, 'delta_y': delta_y
                    }
                    st.rerun()
            
            with col_reset:
                if st.button("🎯 Reset Zoom"):
                    st.session_state.zoom_params = {
                        'center_x': 0.0, 'center_y': 0.0,
                        'width': 0.05, 'height': 0.05,  # Default zoom box size
                        'delta_x': 0.005, 'delta_y': 0.005  # Fixed panning step
                    }
                    st.rerun()
        
        # Usage tips
        with st.expander("Usage Tips"):
            st.markdown("""
            **How to use Enhanced Dual Viewer:**
            1. **Zoom Center**: Set the center point of your zoom region
            2. **Zoom Size**: Define width/height of the zoom box
            3. **Pan Movement**: Use Delta X/Y to move the viewport
            4. **Update View**: Click 'Update Zoom' to apply center + pan changes
            5. **Detail View**: Shows large labels with 10% margin around zoom area
            """)


def setup_action_buttons():
    """Setup action buttons and return button states"""
    with st.sidebar:
        # Generate button
        btn_vis_col, btn_pan_col, btn_save_img_col = st.columns([2, 2, 2])
        
        with btn_vis_col:
            btn_vis = st.button("Visualize", type="primary", help="Generate visualization", key="btn_visualize")
        with btn_pan_col:
            btn_pan = st.button("Panning", type="secondary", help="Apply pan movement from zoom controls", key="btn_panning")
        with btn_save_img_col:
            btn_save_detail_img = st.button("Save Image", type="secondary", help="Save Detail View to Image File", key="btn_save_detail_img")
            
        return btn_vis, btn_pan, btn_save_detail_img


def main():
    check_login()
    
    st.subheader("🔍 Semantics_Explorer - Dual View")
    
    # Initialize components
    visualizer = EmbeddingVisualizer()
    reducer = DimensionReducer()
    dual_manager = EnhancedDualViewManager()
    geometric_analyzer = GeometricAnalyzer()
    
    # Setup all sidebar components
    settings = setup_sidebar_controls()
    model_name = settings['model_name']
    method_name = settings['method_name']
    
    chinese_words, english_words, chinese_selected, english_selected = handle_text_input()
    enable_geometric_analysis, analysis_params = setup_geometric_analysis_controls()
    setup_zoom_controls()
    btn_vis, btn_pan, btn_save_detail_img = setup_action_buttons()
    
    # Handle button actions
    if btn_vis:
        if chinese_words or english_words:
            st.session_state.generate_requested = True
            st.rerun()

    # Handle save detail image
    if btn_save_detail_img:
        if 'enhanced_data' in st.session_state:
            # Get current input name from session state or use a default
            current_input_name = st.session_state.get('cfg_input_text_entered', 'dual-view')
            if not current_input_name or current_input_name == 'untitled':
                current_input_name = 'dual-view'
            
            # Recreate the detail figure for saving
            data = st.session_state.enhanced_data
            # Get dataset name
            dataset_name = st.session_state.get('cfg_input_text_selected', 'User Input')
            
            overview_fig, detail_fig, points_count, viewport_mask = dual_manager.create_enhanced_dual_view(
                data['embeddings'],
                data['labels'], 
                data['colors'],
                data['title'],
                st.session_state.zoom_params,
                model_name,
                method_name,
                dataset_name
            )
            
            # Save the detail view image
            filename = visualizer.save_detail_view_image(
                detail_fig,
                current_input_name,
                model_name,
                method_name,
                chinese_selected,
                english_selected
            )
            
            if filename:
                st.success(f"Detail view saved as: {filename}")
            else:
                st.error("Failed to save detail view image")
        else:
            st.warning("No visualization to save. Please generate a visualization first.")

    # Display existing visualization if available
    if 'enhanced_data' in st.session_state:
        data = st.session_state.enhanced_data
        
        # Create visualization
        # Get dataset name
        dataset_name = st.session_state.get('cfg_input_text_selected', 'User Input')
        
        overview_fig, detail_fig, points_count, viewport_mask = dual_manager.create_enhanced_dual_view(
            data['embeddings'],
            data['labels'],
            data['colors'],
            data['title'],
            st.session_state.zoom_params,
            model_name,
            method_name,
            dataset_name
        )
        
        # Detail view below (no container) with pan button
        st.plotly_chart(detail_fig, use_container_width=True, key="detail_view")
        
        # Add download button for detail view (always available)
        handle_download_button(detail_fig, model_name, method_name, dataset_name, "detail", "dual_view")
        
        if st.session_state.get('btn_panning', False):
            # Get current zoom params from sidebar
            current_params = st.session_state.zoom_params
            # Apply pan movement
            new_center_x = current_params['center_x'] + current_params['delta_x']
            new_center_y = current_params['center_y'] + current_params['delta_y']
            
            st.session_state.zoom_params = {
                'center_x': new_center_x, 'center_y': new_center_y,
                'width': current_params['width'], 'height': current_params['height'],
                'delta_x': current_params['delta_x'], 'delta_y': current_params['delta_y']
            }
            st.rerun()

        # Overview in expandable container
        with st.expander("📊 Overview", expanded=True):
            st.plotly_chart(overview_fig, use_container_width=True, config={'displayModeBar': False})
      
        # Stats in collapsible expander
        with st.expander("📊 Statistics", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Points", len(data['labels']))
            with col2:
                st.metric("Points in Zoom", points_count)
            with col3:
                coverage = (points_count / len(data['labels']) * 100) if len(data['labels']) > 0 else 0
                st.metric("Zoom Coverage", f"{coverage:.1f}%")
            
            # List words in zoom area
            if points_count > 0:
                st.write("**Words in Zoom Area:**")
                zoom_labels = [data['labels'][i] for i in range(len(data['labels'])) if viewport_mask[i]]
                zoom_colors = [data['colors'][i] for i in range(len(data['colors'])) if viewport_mask[i]]
                
                chinese_in_zoom = [label for i, label in enumerate(zoom_labels) if zoom_colors[i] == 'chinese']
                english_in_zoom = [label for i, label in enumerate(zoom_labels) if zoom_colors[i] == 'english']
                
                col_chn, col_eng = st.columns(2)
                with col_chn:
                    if chinese_in_zoom:
                        st.write("**Chinese:**")
                        st.write(", ".join(chinese_in_zoom))
                
                with col_eng:
                    if english_in_zoom:
                        st.write("**English:**")
                        st.write(", ".join(english_in_zoom))
        
        # Display geometric analysis results if available
        display_dual_view_geometric_analysis(model_name, method_name)
    
    # Handle generation request
    if st.session_state.get('generate_requested', False):
        st.session_state.generate_requested = False
        
        if chinese_words or english_words:
            with st.spinner("🔄 Processing embeddings..."):
                # Generate embeddings
                all_embeddings = []
                all_labels = []
                all_colors = []
                
                if chinese_words:
                    chinese_embeddings = visualizer.get_embeddings(chinese_words, model_name, "zh")
                    if chinese_embeddings is not None:
                        all_embeddings.append(chinese_embeddings)
                        all_labels.extend(chinese_words)
                        all_colors.extend(['chinese'] * len(chinese_words))

                if english_words:
                    english_embeddings = visualizer.get_embeddings(english_words, model_name, "en")
                    if english_embeddings is not None:
                        all_embeddings.append(english_embeddings)
                        all_labels.extend(english_words)
                        all_colors.extend(['english'] * len(english_words))

                if all_embeddings:
                    combined_embeddings = np.vstack(all_embeddings)
                    
                    # Reduce dimensions
                    reduced_embeddings = reducer.reduce_dimensions(
                        combined_embeddings, 
                        method=method_name, 
                        dimensions=2
                    )
                    
                    if reduced_embeddings is not None:
                        # Update zoom parameters based on data range if not already set
                        if 'zoom_params' not in st.session_state:
                            x_center = reduced_embeddings[:, 0].mean()
                            y_center = reduced_embeddings[:, 1].mean()
                            
                            st.session_state.zoom_params = {
                                'center_x': x_center, 'center_y': y_center,
                                'width': 0.05, 'height': 0.05,  # Default zoom box size
                                'delta_x': 0.005, 'delta_y': 0.005  # Fixed panning step
                            }
                        
                        # Store data
                        st.session_state.enhanced_data = {
                            'embeddings': reduced_embeddings,
                            'labels': all_labels,
                            'colors': all_colors,
                            'title': f"{model_name} + {method_name}"
                        }
                        
                        # Perform geometric analysis if enabled
                        if enable_geometric_analysis and analysis_params:
                            with st.spinner("🔬 Performing geometric analysis..."):
                                perform_dual_view_geometric_analysis(
                                    geometric_analyzer, analysis_params, 
                                    reduced_embeddings, all_labels,
                                    model_name, method_name
                                )
                        
                        st.rerun()
                    else:
                        st.error("Failed to reduce dimensions")
                else:
                    st.error("Failed to generate embeddings")
        else:
            st.warning("Please enter some text to visualize")


if __name__ == "__main__":
    main()
