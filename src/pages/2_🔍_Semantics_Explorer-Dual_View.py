import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.embedding_viz import EmbeddingVisualizer
from components.dimension_reduction import DimensionReducer
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

# Page config
st.set_page_config(
    page_title="Enhanced Dual Viewer v2",
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

    def create_enhanced_dual_view(self, embeddings, labels, colors, title, zoom_params):
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
                marker=dict(size=6, color='crimson', opacity=0.8, line=dict(width=1, color='white')),
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
                marker=dict(size=6, color='steelblue', opacity=0.8, line=dict(width=1, color='white')),
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
            title_text="X",
            gridcolor='lightgray'
        )
        overview_fig.update_yaxes(
            range=[data_y_min - y_padding, data_y_max + y_padding],
            title_text="Y",
            gridcolor='lightgray',
            scaleanchor="x", scaleratio=1
        )
        
        overview_fig.update_layout(
            title="📊 Overview",
            dragmode='pan',
            hovermode='closest',
            showlegend=False,
            height=500,
            plot_bgcolor='rgba(250, 250, 250, 0.8)'
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
                marker=dict(size=16, color='crimson', opacity=1.0, line=dict(width=2, color='white')),
                text=[labels[i] for i in range(len(labels)) if chinese_viewport[i]],
                textposition="top center",
                textfont=dict(size=16, color='darkred', family='Arial Black'),
                hovertemplate='<b>%{text}</b><br>中文 (Detail)<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                name="中文",
                showlegend=False
            ))
        
        if np.any(english_viewport):
            detail_fig.add_trace(go.Scatter(
                x=embeddings[english_viewport, 0],
                y=embeddings[english_viewport, 1],
                mode='markers+text',
                marker=dict(size=16, color='steelblue', opacity=1.0, line=dict(width=2, color='white')),
                text=[labels[i] for i in range(len(labels)) if english_viewport[i]],
                textposition="top center",
                textfont=dict(size=16, color='darkblue', family='Arial Black'),
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
            title_text="Zoom - X",
            gridcolor='lightgray'
        )
        detail_fig.update_yaxes(
            range=[viewport_coords['y_min'] - y_margin, viewport_coords['y_max'] + y_margin],
            title_text="Zoom - Y",
            gridcolor='lightgray',
            scaleanchor="x", scaleratio=1
        )
        
        detail_fig.update_layout(
            title="🔍 Detail View",
            dragmode='pan',
            hovermode='closest',
            showlegend=False,
            height=900,
            plot_bgcolor='rgba(250, 250, 250, 0.8)'
        )
        
        # Count points in viewport
        points_in_viewport = viewport_mask.sum()
        
        return overview_fig, detail_fig, points_in_viewport, viewport_mask

def main():
    check_login()
    
    st.subheader("🔍 Semantics_Explorer - Dual View")
    
    # Initialize components
    visualizer = EmbeddingVisualizer()
    reducer = DimensionReducer()
    dual_manager = EnhancedDualViewManager()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Settings")
        
        # Model and method selection
        with st.expander("Model & Method", expanded=False):
            model_name = st.selectbox(
                "Embedding Model:",
                options=list(MODEL_INFO.keys()),
                index=list(MODEL_INFO.keys()).index(DEFAULT_MODEL)
            )
            
            method_name = st.selectbox(
                "Reduction Method:",
                options=list(METHOD_INFO.keys()),
                index=list(METHOD_INFO.keys()).index(DEFAULT_METHOD)
            )
        
        # Text input areas - same as Semantics Explorer
        with st.expander("Text Input", expanded=False):
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
                chinese_words = visualizer.process_text(chinese_text) if chinese_selected else []

            with col2:
                english_text = st.text_area(
                    "English:",
                    value=st.session_state.get('english_text_area', default_english),
                    height=200,
                    key='english_text_input'
                )
                english_selected = st.checkbox("English", value=True, key="english")
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

        # Generate button
        btn_vis_col, _, btn_pan_col= st.columns([2, 2, 2])
        with btn_vis_col:
            btn_vis = st.button("Visualize", type="primary", help="Generate visualization", key="btn_visualize")
        with btn_pan_col:
            btn_pan = st.button("Panning", type="secondary", help="Apply pan movement from zoom controls", key="btn_panning")

        if btn_vis:
            if chinese_words or english_words:
                st.session_state.generate_requested = True
                st.rerun()


        # Zoom controls
        with st.expander("Zoom Controls", expanded=True):
            # Global box size parameters
            global_width = 0.8
            global_height = 0.3
            
            # Initialize default zoom parameters
            if 'zoom_params' not in st.session_state:
                st.session_state.zoom_params = {
                    'center_x': 0.0, 'center_y': 0.0,
                    'width': global_width * 0.1, 'height': global_height * 0.1,  # 10% of global
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
                        'width': global_width * 0.1, 'height': global_height * 0.1,  # 10% of global
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
        

    
    # Main content area
    if 'enhanced_data' in st.session_state:
        data = st.session_state.enhanced_data
        
        # Create visualization
        overview_fig, detail_fig, points_count, viewport_mask = dual_manager.create_enhanced_dual_view(
            data['embeddings'],
            data['labels'],
            data['colors'],
            data['title'],
            st.session_state.zoom_params
        )
        
        # Detail view below (no container) with pan button
        st.plotly_chart(detail_fig, use_container_width=True, key="detail_view")
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
                            global_width = 0.8
                            global_height = 0.3
                            x_center = reduced_embeddings[:, 0].mean()
                            y_center = reduced_embeddings[:, 1].mean()
                            
                            st.session_state.zoom_params = {
                                'center_x': x_center, 'center_y': y_center,
                                'width': global_width * 0.1, 'height': global_height * 0.1,  # 10% of global
                                'delta_x': 0.005, 'delta_y': 0.005  # Fixed panning step
                            }
                        
                        # Store data
                        st.session_state.enhanced_data = {
                            'embeddings': reduced_embeddings,
                            'labels': all_labels,
                            'colors': all_colors,
                            'title': f"{model_name} + {method_name}"
                        }
                        
                        st.rerun()
                    else:
                        st.error("Failed to reduce dimensions")
                else:
                    st.error("Failed to generate embeddings")
        else:
            st.warning("Please enter some text to visualize")

if __name__ == "__main__":
    main()