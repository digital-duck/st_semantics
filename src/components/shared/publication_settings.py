"""
Shared publication settings widget for eliminating code duplication
"""
import streamlit as st
from typing import Dict, Any


class PublicationSettingsWidget:
    """Centralized widget for publication settings across pages"""
    
    @staticmethod
    def render_publication_settings(session_key_prefix: str = "publication") -> Dict[str, Any]:
        """
        Render publication settings UI and return the settings dictionary
        
        Args:
            session_key_prefix: Prefix for session state keys to avoid conflicts
            
        Returns:
            Dictionary containing all publication settings
        """
        with st.expander("📊 Publication Settings", expanded=False):
            publication_mode = st.checkbox(
                "Publication Mode", 
                value=False, 
                help="Enable high-quality settings for publication",
                key=f"{session_key_prefix}_publication_mode"
            )
            
            col1, col2 = st.columns(2)
            
            # Left column - Text and Point sizes
            with col1:
                if publication_mode:
                    textfont_size = st.number_input(
                        "Text Size", 
                        min_value=12, max_value=24, value=16, step=1,
                        help="Font size for labels (12-24pt). Suggested: 16pt for publication, 18pt for presentations",
                        key=f"{session_key_prefix}_textfont_size_pub"
                    )
                    point_size = st.number_input(
                        "Point Size", 
                        min_value=8, max_value=20, value=12, step=1,
                        help="Size of data points (8-20pt). Suggested: 12pt for publication, 14-16pt for presentations",
                        key=f"{session_key_prefix}_point_size_pub"
                    )
                else:
                    textfont_size = st.number_input(
                        "Text Size", 
                        min_value=8, max_value=20, value=16, step=1,
                        help="Font size for labels (8-20pt). Default: 16pt for better readability",
                        key=f"{session_key_prefix}_textfont_size_std"
                    )
                    point_size = st.number_input(
                        "Point Size", 
                        min_value=2, max_value=12, value=12, step=1,
                        help="Size of data points (2-12pt). Default: 12pt for clear visibility",
                        key=f"{session_key_prefix}_point_size_std"
                    )
            
            # Right column - Plot dimensions
            with col2:
                if publication_mode:
                    plot_width = st.number_input(
                        "Width",
                        min_value=800, max_value=1600, value=1200, step=50,
                        help="Plot width in pixels (800-1600px). Square aspect ratio recommended for manifold learning",
                        key=f"{session_key_prefix}_plot_width_pub"
                    )
                    plot_height = st.number_input(
                        "Height",
                        min_value=800, max_value=1600, value=1200, step=50,
                        help="Plot height in pixels (800-1600px). Square aspect ratio preserves geometric relationships",
                        key=f"{session_key_prefix}_plot_height_pub"
                    )
                else:
                    plot_width = st.number_input(
                        "Width",
                        min_value=600, max_value=1000, value=800, step=50,
                        help="Plot width in pixels (600-1000px). Square aspect ratio recommended for manifold learning",
                        key=f"{session_key_prefix}_plot_width_std"
                    )
                    plot_height = st.number_input(
                        "Height",
                        min_value=600, max_value=1000, value=800, step=50,
                        help="Plot height in pixels (600-1000px). Square aspect ratio preserves geometric relationships",
                        key=f"{session_key_prefix}_plot_height_std"
                    )
            
            # Export options (only in publication mode)
            if publication_mode:
                st.markdown("**Export Options**")
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    export_format = st.selectbox(
                        "Format", 
                        ["PNG", "SVG", "PDF"], 
                        index=0,
                        help="Export format: PNG (raster, good for most uses), SVG (vector, scalable), PDF (vector, publication-ready)",
                        key=f"{session_key_prefix}_export_format"
                    )
                
                with col_exp2:
                    export_dpi = st.number_input(
                        "DPI", 
                        min_value=150, max_value=600, value=300, step=50,
                        help="Dots per inch (150-600). Suggested: 300 DPI for journals, 150-200 for web, 600 for high-quality prints",
                        key=f"{session_key_prefix}_export_dpi"
                    )
            else:
                export_format = "PNG"
                export_dpi = 150
            
            # Return settings dictionary
            return {
                'publication_mode': publication_mode,
                'textfont_size': textfont_size,
                'point_size': point_size,
                'plot_width': plot_width,
                'plot_height': plot_height,
                'export_format': export_format,
                'export_dpi': export_dpi
            }
    
    @staticmethod
    def get_default_settings() -> Dict[str, Any]:
        """Get default settings for use when publication settings are not available"""
        return {
            'publication_mode': False,
            'textfont_size': 16,
            'point_size': 12,
            'plot_width': 800,
            'plot_height': 800,  # Square aspect ratio for manifold learning
            'export_format': 'PNG',
            'export_dpi': 300  # 300 DPI default for publication quality
        }