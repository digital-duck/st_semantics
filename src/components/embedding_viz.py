import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import os
import re
from pathlib import Path
import unicodedata

from config import (
    MODEL_INFO, METHOD_INFO, DEFAULT_MODEL, DEFAULT_METHOD,
    COLOR_MAP,
    sample_chn_input_data, sample_enu_input_data,
    SRC_DIR
)
from models.model_manager import get_model, get_active_models
from utils.error_handling import handle_errors
from components.shared.publication_settings import PublicationSettingsWidget

from components.plotting import PlotManager

def get_active_methods():
    """Get only active methods for UI display"""
    active_methods = {}
    for name, info in METHOD_INFO.items():
        if info.get("is_active", True):  # Default to True for backward compatibility
            active_methods[name] = info
    return active_methods

def rearrange_by_ollama(models):
    l1 = []
    l2 = []
    for i in models:
        if "(Ollama)" in i:
            l2.append(i)
        else:
            l1.append(i)
    return l1 + l2

class EmbeddingVisualizer:
    def __init__(self):
        # Use only active models and methods
        active_models = get_active_models()
        self.model_names = rearrange_by_ollama(sorted(list(active_models.keys())))

        self.active_methods = get_active_methods()
        self.method_names = sorted(list(self.active_methods.keys()))
        self.input_dir = SRC_DIR / "data/input"
        self.images_dir = SRC_DIR / "data/images"
        
        # Initialize session state for plot rotation
        if 'plot_rotation' not in st.session_state:
            st.session_state.plot_rotation = 0
        if 'current_figure' not in st.session_state:
            st.session_state.current_figure = None

    def render_sidebar(self) -> Tuple[str, str, str, bool, Optional[int]]:
        """Render sidebar controls and return settings"""
        with st.sidebar:
            
            with st.expander("Visualization Settings", expanded=False):

                # Model selection
                model_name = st.radio(
                    "Choose Embedding Model (Ollama is slower)",
                    options=self.model_names,
                    index=self.model_names.index(DEFAULT_MODEL),
                    help="Select a multilingual embedding model",
                    key="cfg_embed_model_name"
                )

                # Warning for E5-Base-v2 with Chinese text
                if model_name == "E5-Base-v2":
                    st.warning("⚠️ E5-Base-v2 may produce NaN errors with Chinese text. Consider using Sentence-BERT Multilingual or BGE-M3 for Chinese-English datasets.")
                if model_name == DEFAULT_MODEL:
                    info_msg = f"**{model_name}** (default): {MODEL_INFO[model_name]['help']}"
                else:
                    info_msg = f"**{model_name}**: {MODEL_INFO[model_name]['help']}"
                st.info(info_msg)

                # Method selection
                method_name = st.radio(
                    "Choose Dimensionality Reduction Method",
                    options=self.method_names,
                    index=self.method_names.index(DEFAULT_METHOD),
                    help="Select a dimensionality reduction method",
                    key="cfg_dim_reduc_method_name"
                )
                if method_name == DEFAULT_METHOD:
                    info_msg = f"**{method_name}** (default): {self.active_methods[method_name]['help']}"
                else:
                    info_msg = f"**{method_name}**: {self.active_methods[method_name]['help']}"
                st.info(info_msg)

                # Dimensions
                dimensions = st.radio(
                    "Choose Dimensions",
                    options=["2D", "3D"],
                    index=0,
                    help="Select 2D or 3D visualization",
                    key="cfg_vis_dimensions"
                )

                # Clustering
                do_clustering = st.checkbox(
                    "Enable Clustering?", 
                    value=False,
                    help="Toggle clustering of points in the visualization",
                    key="cfg_enable_clustering"
                )
                n_clusters = None
                if do_clustering:
                    n_clusters = st.slider("Number of Clusters", min_value=3, max_value=10, value=5)

            # Publication Settings (using shared component)
            publication_settings = PublicationSettingsWidget.render_publication_settings("main_viz")
            
            # Store settings in session state for PlotManager to access
            st.session_state.publication_settings = publication_settings

        return model_name, method_name, dimensions, do_clustering, n_clusters

    @handle_errors
    def process_text(self, text: str, dedup: bool = True) -> List[str]:
        """Process input text into list of words
        
        Args:
            text: Input text string
            dedup: Whether to remove duplicates
            
        Returns:
            List of processed words, ignoring comment lines starting with #
        """
        # Split into lines and filter out comment lines starting with #
        lines = text.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('#')]
        
        # Join back and process as before
        filtered_text = '\n'.join(filtered_lines)
        filtered_text = filtered_text.replace("\n", " ").replace(",", " ").replace(";", " ").replace("，", " ").replace("；", " ")
        results = [w.strip('"') for w in filtered_text.split() if w.strip('"')]
        return list(set(results)) if dedup else results

    def get_available_inputs(self) -> List[str]:
        """Get list of available input names from data/input directory"""
        if not self.input_dir.exists():
            return ["sample_1"]
        
        input_names = set()
        for file_path in self.input_dir.glob("*.txt"):
            name_part = file_path.stem.replace("-chn", "").replace("-enu", "")
            input_names.add(name_part)
        
        return sorted(list(input_names)) if input_names else ["sample_1"]
    
    def load_text_from_file(self, input_name: str, language: str) -> str:
        """Load text content from file"""
        file_path = self.input_dir / f"{input_name}-{language}.txt"
        if file_path.exists():
            try:
                return file_path.read_text(encoding='utf-8').strip()
            except Exception as e:
                st.error(f"Error reading file {file_path}: {e}")
        return ""
    
    def save_text_to_file(self, input_name: str, chinese_text: str, english_text: str, 
                          chinese_selected: bool, english_selected: bool):
        """Save text content to files"""
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename using the proper method
        safe_name = self.sanitize_filename(input_name)
        
        success_count = 0
        
        # Save Chinese text if provided
        if chinese_selected and chinese_text.strip():
            chn_file = self.input_dir / f"{safe_name}-chn.txt"
            try:
                chn_file.write_text(chinese_text.strip(), encoding='utf-8')
                success_count += 1
            except Exception as e:
                st.error(f"Error saving Chinese text: {e}")
        
        # Save English text if provided
        if english_selected and english_text.strip():
            enu_file = self.input_dir / f"{safe_name}-enu.txt"
            try:
                enu_file.write_text(english_text.strip(), encoding='utf-8')
                success_count += 1
            except Exception as e:
                st.error(f"Error saving English text: {e}")
        
        if success_count > 0:
            st.success(f"Saved {success_count} text file(s) as '{safe_name}'")
            st.rerun()  # Refresh to update the selectbox options
        else:
            st.warning("No text to save")

    def save_multilingual_text(self, input_name_raw: str, chinese_text: str, chinese_selected: bool, 
                              target_words_dict: dict, target_selected_dict: dict, lang_code_map: dict):
        """Save multilingual text content to files"""
        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name = self.sanitize_filename(input_name_raw)
        success_count = 0
        
        # Save Chinese text if provided
        if chinese_selected and chinese_text.strip():
            chn_file = self.input_dir / f"{safe_name}-chn.txt"
            try:
                chn_file.write_text(chinese_text.strip(), encoding='utf-8')
                success_count += 1
            except Exception as e:
                st.error(f"Error saving Chinese text: {e}")
        
        # Save target language texts
        reverse_lang_map = {v: k for k, v in lang_code_map.items()}  # Map lang codes back to names
        for lang_code, selected in target_selected_dict.items():
            if selected:
                # Get the actual text from session state
                lang_text = st.session_state.get(f'{lang_code}_text_area', '')
                if lang_text.strip():
                    lang_file = self.input_dir / f"{safe_name}-{lang_code}.txt"
                    try:
                        lang_file.write_text(lang_text.strip(), encoding='utf-8')
                        success_count += 1
                    except Exception as e:
                        lang_name = reverse_lang_map.get(lang_code, lang_code)
                        st.error(f"Error saving {lang_name} text: {e}")
        
        if success_count > 0:
            st.success(f"Saved {success_count} text file(s) as '{safe_name}'")
            st.rerun()  # Refresh to update the selectbox options
        else:
            st.warning("No text to save")

    def render_input_areas(self) -> Tuple[List[str], List[str], List[str]]:
        """Render text input areas and return processed words"""
        with st.sidebar:
            with st.expander("Enter Text Data (Word/Phrase):", expanded=True):
                
                # Language selection
                st.markdown("Source Language: Chinese (chn)")
                # Multi-target language selection
                target_language = st.selectbox(
                    "**Target Language**",
                    options=["English"], #, "French", "Spanish", "German"],
                    index=0,
                    help="Select target language for comparison with Chinese",
                    key='target_languages'
                )
                
                # Language code mapping
                lang_code_map = {
                    "English": "enu", 
                    "French": "fra", 
                    "Spanish": "spa", 
                    "German": "deu"
                }
                target_lang_code = lang_code_map[target_language]

                col_input_select, col_load_txt = st.columns([3, 1])
                with col_input_select:
                    available_inputs = self.get_available_inputs()
                    input_name_selected = st.selectbox(
                        "Select Input",
                        options=[""] + available_inputs,
                        index=0,
                        key="input_name_selected"
                    )
                with col_load_txt:
                    btn_load_txt = st.button("Load Text", type="primary", 
                                             help="Load input texts", 
                                             disabled=not input_name_selected)

                # Initialize text areas with default or loaded content
                default_texts = {
                    "chn": sample_chn_input_data,
                    "enu": sample_enu_input_data,
                    "fra": "rouge\nbleu\nvert\njaune\norange",  # Sample French
                    "spa": "rojo\nazul\nverde\namarillo\nnaranja",  # Sample Spanish  
                    "deu": "rot\nblau\ngrün\ngelb\norange"  # Sample German
                }
                
                # Load text if button is clicked
                if btn_load_txt:
                    missing_files = []
                    # Load Chinese (source)
                    loaded_chinese = self.load_text_from_file(input_name_selected, "chn")
                    st.session_state["chn_text_area"] = loaded_chinese
                    if not loaded_chinese:
                        missing_files.append(f"'{input_name_selected}-chn.txt'")

                    # Load target languages
                    lang_code = target_lang_code
                    loaded_text = self.load_text_from_file(input_name_selected, lang_code)
                    st.session_state[f"{lang_code}_text_area"] = loaded_text
                    if not loaded_text:
                        missing_files.append(f"'{input_name_selected}-{lang_code}.txt'")

                    if missing_files:
                        st.warning("No text files found: " + " ; ".join(missing_files))
                    # else:
                    #     # Force rerun to update checkbox states
                    #     st.rerun()


                # Dynamic text areas based on selected languages
                num_languages = 2  # Chinese + target languages
                cols = st.columns(num_languages)  # Max 3 columns for readability
                
                # Chinese (source) - always first
                with cols[0]:
                    chinese_text = st.text_area(
                        "Chinese (chn):",
                        value=st.session_state.get('chn_text_area', default_texts["chn"]),
                        height=200,
                        key='chn_text_area'
                    )
                    chinese_text = (chinese_text or "").strip()
                    # print(f"[DBG] chinese_text: {chinese_text}")
                    # Calculate checkbox value from session state content
                    chn_content = st.session_state.get('chn_text_area', '').strip()
                    chinese_selected = st.checkbox("Include", value=(len(chn_content) > 0), key='chn_include_checkbox') 
                    chinese_words = self.process_text(chn_content) if chn_content else []

                # Target languages
                target_words_dict = {}
                target_selected_dict = {}
                col_idx = 1


                lang_code = target_lang_code
                lang_name = target_language                
                # print(f"[DBG] ENU_text: {st.session_state[f'{lang_code}_text_import']}")
                with cols[1]:

                    lang_text = st.text_area(
                        f"{lang_name} ({lang_code}):",
                        value=st.session_state.get(f'{lang_code}_text_area', default_texts.get(lang_code, "")),
                        height=200,
                        key=f'{lang_code}_text_area'
                    )
                    lang_text = (lang_text or "").strip()
                    # print(f"[DBG] lang_text: {lang_text}")
                    # Calculate checkbox value from session state content
                    lang_content = st.session_state.get(f'{lang_code}_text_area', '').strip()
                    target_selected_dict[lang_code] = st.checkbox("Include", value=(len(lang_content) > 0), key=f'{lang_code}_include_checkbox') 
                    target_words_dict[lang_code] = self.process_text(lang_content) if lang_content else []
                    

                # User can enter a name for the input and save the texts 
                col_input_enter, col_save_txt = st.columns([3, 1])
                with col_input_enter:
                    input_name_raw = st.text_input(
                        "Name Input",
                        value=st.session_state.get("input_name_selected", "untitled"),
                        key="cfg_input_text_entered",
                        help="Name will be automatically sanitized for filename compatibility"
                    )
                    # Show sanitized preview
                    sanitized_preview = self.sanitize_filename(input_name_raw)
                    if sanitized_preview != input_name_raw.lower():
                        st.caption(f"📝 Preview: `{sanitized_preview}`")
                        
                with col_save_txt:
                    btn_save_txt = st.button("Save Text", type="primary", 
                                             help="Save input texts", 
                                             disabled=(input_name_raw=="untitled"))
                    
                # Handle save text
                if btn_save_txt:
                    self.save_multilingual_text(input_name_raw, chinese_text, chinese_selected, 
                                               target_words_dict, target_selected_dict, lang_code_map)

            col_vis, col_rotate, col_save_png = st.columns([1, 1, 1])
            with col_vis:
                btn_visualize = st.button("Visualize", type="primary")
            with col_rotate:
                btn_rotate_90 = st.button("Rotate", help="Rotate 2D plot by 90°")
            with col_save_png:
                btn_save_png = st.button("Save Image", help="Save current plot as PNG image")
            btn_actions = (btn_visualize, btn_rotate_90, btn_save_png)

            # Combine all words and create corresponding colors
            all_words = chinese_words.copy()
            all_colors = [COLOR_MAP["chinese"]] * len(chinese_words)
            
            # Map language codes to color map keys
            lang_color_map = {
                "enu": "english",
                "fra": "french", 
                "spa": "spanish",
                "deu": "german"
            }
            
            # Add target language words and colors
            for lang_code, words in target_words_dict.items():
                if words:  # Only add if there are words
                    all_words.extend(words)
                    color_key = lang_color_map.get(lang_code, "english")
                    all_colors.extend([COLOR_MAP[color_key]] * len(words))
            
            return btn_actions, chinese_words, target_words_dict, all_colors, chinese_selected, target_selected_dict
    
    @st.cache_data
    def get_embeddings(_self, words: List[str], model_name: str, lang: str) -> np.ndarray:
        """Get embeddings for words using specified model"""
        model = get_model(model_name)
        return model.get_embeddings(words, lang)

    
    def sanitize_filename(self, text: str) -> str:
        """Sanitize text for use in filename"""


        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove or replace characters that are problematic in filenames
        # Keep alphanumeric, Chinese/CJK characters, hyphens, underscores, and spaces
        sanitized = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf-]', '', text)

        # Replace multiple whitespace with single underscore
        sanitized = re.sub(r'\s+', '_', sanitized.strip())

        # Remove duplicate underscores
        sanitized = re.sub(r'_+', '_', sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')

        return sanitized if sanitized else "untitled"

    
    def save_plot_image(self, input_name: str, model_name: str, method_name: str, chinese_selected: bool, english_selected: bool):
        """Save the current plot as PNG image with language tags"""
        if st.session_state.current_figure is None:
            st.warning("No plot to save. Please generate a visualization first.")
            return ""
            
        # Ensure images directory exists
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sanitized filename
        safe_input = self.sanitize_filename(input_name)
        safe_model = self.sanitize_filename(model_name)
        safe_method = self.sanitize_filename(method_name)
        
        # Add language tags
        lang_tags = []
        if chinese_selected:
            lang_tags.append("chn")
        if english_selected:
            lang_tags.append("enu")
        
        lang_suffix = "-".join(lang_tags) if lang_tags else "none"
        
        filename = f"{safe_input}-{safe_model}-{safe_method}-{lang_suffix}.png"
        file_path = self.images_dir / filename
        
        try:
            # Save the figure as PNG
            st.session_state.current_figure.write_image(str(file_path), width=1200, height=800, scale=2)
            # st.success(f"Image saved as: {filename}")
            return filename
        except Exception as e:
            st.error(f"Error saving image: {e}")
            return ""

    def save_detail_view_image(self, detail_figure, input_name: str, model_name: str, method_name: str, chinese_selected: bool, english_selected: bool):
        """Save the detail view plot as PNG image with zoom ID"""
        if detail_figure is None:
            st.warning("No detail view to save. Please generate a visualization first.")
            return ""
            
        # Ensure images directory exists
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sanitized filename
        safe_input = self.sanitize_filename(input_name)
        safe_model = self.sanitize_filename(model_name)
        safe_method = self.sanitize_filename(method_name)
        
        # Add language tags
        lang_tags = []
        if chinese_selected:
            lang_tags.append("chn")
        if english_selected:
            lang_tags.append("enu")
        
        lang_suffix = "-".join(lang_tags) if lang_tags else "none"
        
        # Initialize zoom counter if not exists
        if 'zoom_save_counter' not in st.session_state:
            st.session_state.zoom_save_counter = 1
        
        zoom_id = st.session_state.zoom_save_counter
        filename = f"{safe_input}-{safe_model}-{safe_method}-{lang_suffix}-zoom-{zoom_id}.png"
        file_path = self.images_dir / filename
        
        try:
            # Save the detail figure as PNG with higher resolution for paper figures
            detail_figure.write_image(str(file_path), width=1600, height=1200, scale=2)
            
            # Increment counter for next save
            st.session_state.zoom_save_counter += 1
            
            return filename
        except Exception as e:
            st.error(f"Error saving detail view image: {e}")
            return ""

    def create_plot(self, embeddings, labels, colors, model_name, method_name, 
                    dimensions, do_clustering, n_clusters, dataset_name="User Input"):
        """Create and display the plot"""
        plot_title = f"[Model] {model_name}, [Method] {method_name}"
        plot_mgr = PlotManager()
        
        # Apply rotation if needed (only for 2D plots)
        if dimensions == "2D" and st.session_state.plot_rotation > 0:
            # Apply rotation transformation
            angle = np.radians(st.session_state.plot_rotation)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            embeddings = embeddings @ rotation_matrix.T

        if dimensions == "2D":
            fig = plot_mgr.plot_2d(
                embeddings=embeddings,
                labels=labels,
                colors=colors,
                title=plot_title,
                clustering=do_clustering,
                n_clusters=n_clusters if do_clustering else None,
                method_name=method_name,
                model_name=model_name,
                dataset_name=dataset_name
            )
        else:
            fig = plot_mgr.plot_3d(
                embeddings=embeddings,
                labels=labels,
                colors=colors,
                title=plot_title,
                clustering=do_clustering,
                n_clusters=n_clusters if do_clustering else None,
                method_name=method_name,
                model_name=model_name,
                dataset_name=dataset_name
            )
        
        # Store the figure in session state for saving
        st.session_state.current_figure = fig
        
    def display_saved_images(self):
        """Display all saved images in the images directory"""
        if not self.images_dir.exists():
            st.info("No images saved yet. Generate a visualization and click 'Save Image'.")
            return
            
        image_files = list(self.images_dir.glob("*.png"))
        
        if not image_files:
            st.info("No images found in the images directory.")
            return
            
        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        st.write(f"Found {len(image_files)} saved images:")
        
        # Display images in a grid
        cols = st.columns(2)  # 2 columns for images
        
        for idx, image_file in enumerate(image_files):
            col = cols[idx % 2]
            
            with col:
                # Display filename
                st.write(f"**{image_file.name}**")
                
                # Display image
                try:
                    st.image(str(image_file), caption=image_file.stem, use_container_width=True)
                    
                    # Add download button
                    with open(image_file, "rb") as file:
                        st.download_button(
                            label=f"Download {image_file.name}",
                            data=file.read(),
                            file_name=image_file.name,
                            mime="image/png",
                            key=f"download_{image_file.stem}_{idx}"
                        )
                        
                    # Add delete button
                    if st.button(f"Delete", key=f"delete_{image_file.stem}_{idx}", help=f"Delete {image_file.name}"):
                        try:
                            image_file.unlink()
                            st.success(f"Deleted {image_file.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting {image_file.name}: {e}")
                    
                    st.divider()
                    
                except Exception as e:
                    st.error(f"Error displaying {image_file.name}: {e}")