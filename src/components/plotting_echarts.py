import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from config import (
    PLOT_WIDTH, PLOT_HEIGHT,
    DEFAULT_N_CLUSTERS, DEFAULT_MIN_CLUSTERS, DEFAULT_MAX_CLUSTERS,
    DEFAULT_MAX_WORDS
)
import json
import os
import base64
from datetime import datetime
import time
import tempfile
import html

# Optional imports for automatic PNG export
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class EChartsPlotManager:
    """Apache ECharts plotting manager for semantic visualizations"""

    def __init__(self):
        self.min_clusters = DEFAULT_MIN_CLUSTERS
        self.max_clusters = DEFAULT_MAX_CLUSTERS
        # ECharts styling configuration
        self.echarts_theme = {
            'background_color': '#ffffff',
            'text_color': '#333333',
            'grid_color': '#e0e0e0',
            'chinese_color': '#1f4e79',
            'english_color': '#1f4e79',
            'cluster_colors': [
                '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4',
                '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd'
            ]
        }

    def get_visualization_settings(self):
        """Get visualization settings from session state"""
        default_settings = {
            'plot_width': PLOT_WIDTH,
            'plot_height': PLOT_HEIGHT,
            'text_size': 12,
            'point_size': 8,
            'show_grid': True,
            'animation_enabled': True
        }
        return st.session_state.get('echarts_settings', default_settings)

    def create_title(self, method_name, model_name, dataset_name=""):
        """Create standardized plot title"""
        title_parts = [f"[Method] {method_name}", f"[Model] {model_name}"]
        if dataset_name:
            title_parts.append(f"[Dataset] {dataset_name}")
        return ", ".join(title_parts)

    def plot_2d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                max_words=DEFAULT_MAX_WORDS, method_name="", model_name="", dataset_name=""):
        """Create 2D scatter plot using ECharts"""
        settings = self.get_visualization_settings()

        # Create standardized title
        if method_name and model_name:
            title = self.create_title(method_name, model_name, dataset_name)

        if clustering:
            return self._plot_2d_cluster_echarts(embeddings, labels, colors, title, n_clusters, settings, method_name, model_name, dataset_name)
        else:
            return self._plot_2d_simple_echarts(embeddings, labels, colors, title, settings, method_name, model_name, dataset_name)

    def plot_3d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS,
                method_name="", model_name="", dataset_name=""):
        """Create 3D scatter plot using ECharts"""
        settings = self.get_visualization_settings()

        # Create standardized title
        if method_name and model_name:
            title = self.create_title(method_name, model_name, dataset_name)

        if clustering:
            return self._plot_3d_cluster_echarts(embeddings, labels, colors, title, n_clusters, settings, method_name, model_name, dataset_name)
        else:
            return self._plot_3d_simple_echarts(embeddings, labels, colors, title, settings, method_name, model_name, dataset_name)

    def _perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> dict:
        """Perform clustering and calculate quality metrics"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        metrics = {
            "silhouette": round(silhouette_score(embeddings, clusters), 3),
            "calinski": round(calinski_harabasz_score(embeddings, clusters), 3),
            "inertia": round(kmeans.inertia_, 3),
            "cluster_centers": kmeans.cluster_centers_,
            "cluster_labels": clusters
        }

        return metrics, kmeans

    def _display_cluster_metrics(self, metrics: dict):
        """Display clustering quality metrics"""
        cols = st.columns(3)

        with cols[0]:
            st.metric(
                "Silhouette Score",
                metrics["silhouette"],
                help="Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1], higher is better."
            )

        with cols[1]:
            st.metric(
                "Calinski-Harabasz Score",
                metrics["calinski"],
                help="Ratio of between-cluster variance to within-cluster variance. Higher is better."
            )

        with cols[2]:
            st.metric(
                "Inertia",
                metrics["inertia"],
                help="Sum of squared distances to nearest cluster center. Lower is better."
            )

    def _plot_2d_simple_echarts(self, embeddings, labels, colors, title, settings, method_name="", model_name="", dataset_name=""):
        """Create simple 2D scatter plot with ECharts"""
        # Prepare data for ECharts
        data_points = []
        for i, (x, y) in enumerate(embeddings):
            color = self.echarts_theme['chinese_color'] if colors[i] == 'red' else self.echarts_theme['english_color']
            data_points.append({
                'value': [float(x), float(y)],
                'name': labels[i],
                'itemStyle': {'color': color}
            })

        option = {
            'title': {
                'text': title,
                'left': 'center',
                'textStyle': {
                    'fontSize': 16,
                    'color': self.echarts_theme['text_color']
                }
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{b}: ({c})'
            },
            'grid': {
                'left': '10%',
                'right': '10%',
                'bottom': '15%',
                'top': '5%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'value',
                'name': 'X',
                'nameLocation': 'middle',
                'nameGap': 30,
                'splitLine': {
                    'show': settings['show_grid'],
                    'lineStyle': {'color': self.echarts_theme['grid_color']}
                }
            },
            'yAxis': {
                'type': 'value',
                'name': 'Y',
                'nameLocation': 'middle',
                'nameGap': 40,
                'splitLine': {
                    'show': settings['show_grid'],
                    'lineStyle': {'color': self.echarts_theme['grid_color']}
                }
            },
            'series': [{
                'type': 'scatter',
                'data': data_points,
                'symbolSize': settings['point_size'],
                'label': {
                    'show': True,
                    'position': 'top',
                    'fontSize': settings['text_size'],
                    'color': self.echarts_theme['text_color'],
                    'formatter': '{b}'
                },
                'emphasis': {
                    'itemStyle': {
                        'shadowBlur': 10,
                        'shadowColor': 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }],
            'animation': settings.get('animation_enabled', True),
            'backgroundColor': self.echarts_theme['background_color']
        }

        # Enhance chart with export capabilities
        filename_parts = [method_name, model_name, dataset_name] if method_name and model_name else ["simple", "2d"]
        option = self.enhance_chart_with_export(option, filename_parts, "2D")

        # Render ECharts
        st_echarts(
            options=option,
            height=f"{settings['plot_height']}px",
            width="100%",
            key="echarts_2d_simple"
        )

        return option

    def _plot_2d_cluster_echarts(self, embeddings, labels, colors, title, n_clusters, settings, method_name="", model_name="", dataset_name=""):
        """Create 2D scatter plot with clustering using ECharts"""
        # Add clustering controls
        boundary_opacity = st.slider(
            "Cluster Boundary Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Adjust the visibility of cluster boundaries."
        )

        # Perform clustering
        metrics, kmeans = self._perform_clustering(embeddings, n_clusters)
        self._display_cluster_metrics(metrics)

        # Prepare data for ECharts - group by cluster
        series_data = []
        for cluster_id in range(n_clusters):
            cluster_mask = metrics["cluster_labels"] == cluster_id
            cluster_points = []

            for i, (x, y) in enumerate(embeddings):
                if metrics["cluster_labels"][i] == cluster_id:
                    cluster_points.append({
                        'value': [float(x), float(y)],
                        'name': labels[i]
                    })

            if cluster_points:  # Only add series if it has points
                series_data.append({
                    'name': f'Cluster {cluster_id}',
                    'type': 'scatter',
                    'data': cluster_points,
                    'symbolSize': settings['point_size'],
                    'itemStyle': {
                        'color': self.echarts_theme['cluster_colors'][cluster_id % len(self.echarts_theme['cluster_colors'])]
                    },
                    'label': {
                        'show': True,
                        'position': 'top',
                        'fontSize': settings['text_size'],
                        'color': self.echarts_theme['text_color'],
                        'formatter': '{b}'
                    },
                    'emphasis': {
                        'itemStyle': {
                            'shadowBlur': 10,
                            'shadowColor': 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                })

        option = {
            'title': {
                'text': title,
                'left': 'center',
                'textStyle': {
                    'fontSize': 16,
                    'color': self.echarts_theme['text_color']
                }
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{a}: {b} ({c})'
            },
            'legend': {
                'data': [f'Cluster {i}' for i in range(n_clusters)],
                'top': 'bottom',
                'left': 'center'
            },
            'grid': {
                'left': '10%',
                'right': '10%',
                'bottom': '20%',
                'top': '5%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'value',
                'name': 'X',
                'nameLocation': 'middle',
                'nameGap': 30,
                'splitLine': {
                    'show': settings['show_grid'],
                    'lineStyle': {'color': self.echarts_theme['grid_color']}
                }
            },
            'yAxis': {
                'type': 'value',
                'name': 'Y',
                'nameLocation': 'middle',
                'nameGap': 40,
                'splitLine': {
                    'show': settings['show_grid'],
                    'lineStyle': {'color': self.echarts_theme['grid_color']}
                }
            },
            'series': series_data,
            'animation': settings.get('animation_enabled', True),
            'backgroundColor': self.echarts_theme['background_color']
        }

        # Enhance chart with export capabilities
        filename_parts = [method_name, model_name, dataset_name, "cluster"] if method_name and model_name else ["cluster", "2d"]
        option = self.enhance_chart_with_export(option, filename_parts, "2D")

        # Render ECharts
        st_echarts(
            options=option,
            height=f"{settings['plot_height']}px",
            width="100%",
            key="echarts_2d_cluster"
        )

        return option

    def _plot_3d_simple_echarts(self, embeddings, labels, colors, title, settings, method_name="", model_name="", dataset_name=""):
        """Create simple 3D scatter plot with ECharts"""
        # Prepare data for ECharts 3D
        data_points = []
        for i, (x, y, z) in enumerate(embeddings):
            color = self.echarts_theme['chinese_color'] if colors[i] == 'red' else self.echarts_theme['english_color']
            data_points.append({
                'value': [float(x), float(y), float(z)],
                'name': labels[i],
                'itemStyle': {'color': color}
            })

        option = {
            'title': {
                'text': title,
                'left': 'center',
                'textStyle': {
                    'fontSize': 16,
                    'color': self.echarts_theme['text_color']
                }
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{b}: ({c})'
            },
            'grid3D': {
                'boxWidth': 200,
                'boxHeight': 200,
                'boxDepth': 200,
                'viewControl': {
                    'projection': 'perspective',
                    'autoRotate': False,
                    'distance': 300
                }
            },
            'xAxis3D': {
                'type': 'value',
                'name': 'Dimension 1'
            },
            'yAxis3D': {
                'type': 'value',
                'name': 'Dimension 2'
            },
            'zAxis3D': {
                'type': 'value',
                'name': 'Dimension 3'
            },
            'series': [{
                'type': 'scatter3D',
                'data': data_points,
                'symbolSize': settings['point_size'],
                'label': {
                    'show': True,
                    'fontSize': settings['text_size'],
                    'color': self.echarts_theme['text_color'],
                    'formatter': '{b}'
                },
                'emphasis': {
                    'itemStyle': {
                        'shadowBlur': 10,
                        'shadowColor': 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }],
            'animation': settings.get('animation_enabled', True),
            'backgroundColor': self.echarts_theme['background_color']
        }

        # Enhance chart with export capabilities
        filename_parts = [method_name, model_name, dataset_name] if method_name and model_name else ["simple", "3d"]
        option = self.enhance_chart_with_export(option, filename_parts, "3D")

        # Render ECharts
        st_echarts(
            options=option,
            height=f"{settings['plot_height']}px",
            width="100%",
            key="echarts_3d_simple"
        )

        return option

    def _plot_3d_cluster_echarts(self, embeddings, labels, colors, title, n_clusters, settings, method_name="", model_name="", dataset_name=""):
        """Create 3D scatter plot with clustering using ECharts"""
        # Perform clustering
        metrics, kmeans = self._perform_clustering(embeddings, n_clusters)
        self._display_cluster_metrics(metrics)

        # Prepare data for ECharts 3D - group by cluster
        series_data = []
        for cluster_id in range(n_clusters):
            cluster_points = []

            for i, (x, y, z) in enumerate(embeddings):
                if metrics["cluster_labels"][i] == cluster_id:
                    cluster_points.append({
                        'value': [float(x), float(y), float(z)],
                        'name': labels[i]
                    })

            if cluster_points:  # Only add series if it has points
                series_data.append({
                    'name': f'Cluster {cluster_id}',
                    'type': 'scatter3D',
                    'data': cluster_points,
                    'symbolSize': settings['point_size'],
                    'itemStyle': {
                        'color': self.echarts_theme['cluster_colors'][cluster_id % len(self.echarts_theme['cluster_colors'])]
                    },
                    'label': {
                        'show': True,
                        'fontSize': settings['text_size'],
                        'color': self.echarts_theme['text_color'],
                        'formatter': '{b}'
                    },
                    'emphasis': {
                        'itemStyle': {
                            'shadowBlur': 10,
                            'shadowColor': 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                })

        option = {
            'title': {
                'text': title,
                'left': 'center',
                'textStyle': {
                    'fontSize': 16,
                    'color': self.echarts_theme['text_color']
                }
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{a}: {b} ({c})'
            },
            'legend': {
                'data': [f'Cluster {i}' for i in range(n_clusters)],
                'top': 'bottom',
                'left': 'center'
            },
            'grid3D': {
                'boxWidth': 200,
                'boxHeight': 200,
                'boxDepth': 200,
                'viewControl': {
                    'projection': 'perspective',
                    'autoRotate': False,
                    'distance': 300
                }
            },
            'xAxis3D': {
                'type': 'value',
                'name': 'Dimension 1'
            },
            'yAxis3D': {
                'type': 'value',
                'name': 'Dimension 2'
            },
            'zAxis3D': {
                'type': 'value',
                'name': 'Dimension 3'
            },
            'series': series_data,
            'animation': settings.get('animation_enabled', True),
            'backgroundColor': self.echarts_theme['background_color']
        }

        # Enhance chart with export capabilities
        filename_parts = [method_name, model_name, dataset_name, "cluster"] if method_name and model_name else ["cluster", "3d"]
        option = self.enhance_chart_with_export(option, filename_parts, "3D")

        # Render ECharts
        st_echarts(
            options=option,
            height=f"{settings['plot_height']}px",
            width="100%",
            key="echarts_3d_cluster"
        )

        return option


    def render_settings_controls(self):
        """Render ECharts-specific settings controls in sidebar"""
        with st.sidebar.expander("📊 ECharts Settings", expanded=False):
            settings = self.get_visualization_settings()

            # Visual settings
            settings['text_size'] = st.slider(
                "Text Size",
                min_value=8,
                max_value=20,
                value=settings.get('text_size', 12),
                help="Size of text labels"
            )

            settings['point_size'] = st.slider(
                "Point Size",
                min_value=4,
                max_value=20,
                value=settings.get('point_size', 8),
                help="Size of data points"
            )

            settings['show_grid'] = st.checkbox(
                "Show Grid",
                value=settings.get('show_grid', True),
                help="Display grid lines"
            )

            settings['animation_enabled'] = st.checkbox(
                "Enable Animations",
                value=settings.get('animation_enabled', True),
                help="Enable smooth animations and transitions"
            )

            # Chart dimensions
            col1, col2 = st.columns(2)
            with col1:
                settings['plot_width'] = st.number_input(
                    "Width",
                    min_value=400,
                    max_value=1200,
                    value=settings.get('plot_width', PLOT_WIDTH),
                    step=50
                )

            with col2:
                settings['plot_height'] = st.number_input(
                    "Height",
                    min_value=400,
                    max_value=1200,
                    value=settings.get('plot_height', PLOT_HEIGHT),
                    step=50
                )

            # Save settings to session state
            st.session_state.echarts_settings = settings

            return settings

    def save_echarts_as_png(self, chart_config, filename_parts, dimensions="2D"):
        """Save ECharts configuration as PNG image to echarts folder"""
        try:
            # Create filename without timestamp (keep only latest)
            filename_base = "-".join([
                part.lower().replace(" ", "-").replace("_", "-")
                for part in filename_parts if part
            ])

            # Add echarts prefix and dimension suffix
            if dimensions == "3D":
                filename = f"echarts-{filename_base}-3d.png"
            else:
                filename = f"echarts-{filename_base}.png"

            # Create echarts images directory if it doesn't exist
            echarts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images", "echarts")
            os.makedirs(echarts_dir, exist_ok=True)

            filepath = os.path.join(echarts_dir, filename)

            # Save chart configuration as JSON for potential future use (keep only latest)
            config_filename = filename.replace('.png', '.json')
            config_filepath = os.path.join(echarts_dir, config_filename)

            # Remove existing files if they exist (keep only latest)
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(config_filepath):
                os.remove(config_filepath)

            with open(config_filepath, 'w', encoding='utf-8') as f:
                json.dump(chart_config, f, indent=2, ensure_ascii=False)

            # Display combined success message with instructions
            st.info(f"📊 **ECharts visualization configuration saved**\n\n📁 **Config saved as**: `data/images/echarts/{config_filename}`\n\n🖼️ **To save as PNG**: Use your browser's screenshot tool or right-click → 'Save image as...' on the chart")

            # Add download button for the JSON configuration
            with open(config_filepath, 'r', encoding='utf-8') as f:
                config_data = f.read()

            st.download_button(
                label="📥 Download ECharts Config (JSON)",
                data=config_data,
                file_name=config_filename,
                mime="application/json",
                help="Download the ECharts configuration file"
            )

            return config_filename

        except Exception as e:
            st.error(f"Failed to save ECharts configuration: {str(e)}")
            return None

    def create_export_instructions(self):
        """Display instructions for exporting ECharts as PNG"""
        with st.expander("📸 How to Save ECharts as PNG", expanded=False):
            st.markdown("""
            **Method 1: Browser Screenshot (Recommended)**
            1. Right-click on the chart
            2. Select "Save image as..." or "Copy image"
            3. Save to your desired location

            **Method 2: Browser Developer Tools**
            1. Right-click on the chart → "Inspect Element"
            2. Find the `<canvas>` element in the DOM
            3. Right-click on the canvas → "Save image as..."

            **Method 3: Browser Extensions**
            - Use screenshot extensions like "Full Page Screen Capture"
            - Many browsers have built-in screenshot tools (Ctrl+Shift+S in some browsers)

            **Tips for High-Quality Images:**
            - Zoom in your browser before taking screenshot for higher resolution
            - Use browser's full-screen mode (F11) for cleaner captures
            - Adjust ECharts settings (text size, point size) for better visibility

            **Note**: ECharts configurations are automatically saved as JSON files
            in `data/images/echarts/` for reproducibility.
            """)

    def add_export_controls(self, chart_config, filename_parts, dimensions="2D"):
        """Add export controls and save functionality"""
        col1, col2 = st.columns([3, 1])

        with col1:
            # Show export instructions
            self.create_export_instructions()

        with col2:
            # Save configuration button
            if st.button("💾 Save Config", help="Save ECharts configuration as JSON"):
                saved_file = self.save_echarts_as_png(chart_config, filename_parts, dimensions)
                if saved_file:
                    st.success(f"✅ Saved: {saved_file}")

    def enhance_chart_with_export(self, option, filename_parts, dimensions="2D"):
        """Enhance chart configuration with export capabilities"""
        # Add toolbox for built-in export functionality
        if 'toolbox' not in option:
            option['toolbox'] = {
                'show': True,
                'orient': 'vertical',
                'left': 'right',
                'top': 'center',
                'feature': {
                    'saveAsImage': {
                        'show': True,
                        'title': 'Save as PNG',
                        'type': 'png',
                        'backgroundColor': '#ffffff',
                        'pixelRatio': 2,
                        'excludeComponents': ['toolbox']
                    },
                    'restore': {
                        'show': True,
                        'title': 'Reset'
                    },
                    'dataZoom': {
                        'show': True,
                        'title': {
                            'zoom': 'Zoom',
                            'back': 'Reset Zoom'
                        }
                    }
                }
            }

        # Add export controls after the chart
        self.add_export_controls(option, filename_parts, dimensions)

        return option

    def save_echarts_as_png_auto(self, chart_config, filename_parts, dimensions="2D", width=800, height=600):
        """Automatically save ECharts as PNG using headless browser (for 2D only)"""
        if dimensions == "3D":
            st.warning("⚠️ Automatic PNG export is only available for 2D visualizations (3D depends on viewing angle)")
            return None

        if not SELENIUM_AVAILABLE:
            st.warning("⚠️ Automatic PNG export requires selenium. Install with: pip install selenium webdriver-manager")
            return None

        try:
            # Create filename without timestamp (keep only latest)
            filename_base = "-".join([
                part.lower().replace(" ", "-").replace("_", "-")
                for part in filename_parts if part
            ])
            filename = f"echarts-{filename_base}.png"

            # Create echarts images directory if it doesn't exist
            echarts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images", "echarts")
            os.makedirs(echarts_dir, exist_ok=True)
            filepath = os.path.join(echarts_dir, filename)

            # Remove existing file if it exists (keep only latest)
            if os.path.exists(filepath):
                os.remove(filepath)

            # Create HTML file with ECharts - add extra padding to prevent cropping
            html_content = self._create_echarts_html(chart_config, width, height + 120)  # Add 120px padding

            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html_content)
                temp_html_path = f.name

            try:
                # Setup Chrome options for headless mode with better settings
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument(f"--window-size={width + 160},{height + 220}")  # Extra margin for cropping prevention
                chrome_options.add_argument("--hide-scrollbars")
                chrome_options.add_argument("--force-device-scale-factor=1")

                # Setup Chrome driver
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)

                try:
                    # Load the HTML file
                    driver.get(f"file://{temp_html_path}")

                    # Wait for chart container to be present
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "echarts-container"))
                    )

                    # Wait for ECharts library to load and chart to be initialized
                    WebDriverWait(driver, 10).until(
                        lambda driver: driver.execute_script("return typeof echarts !== 'undefined'")
                    )

                    # Wait for render status indicator to show chart is complete
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "#render-status.render-complete"))
                    )

                    # Wait for canvas element to be created (ECharts creates canvas for rendering)
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "#echarts-container canvas"))
                    )

                    # Additional wait to ensure chart animation is complete
                    time.sleep(2)

                    # Verify chart has content by checking if canvas has been drawn to
                    canvas_has_content = driver.execute_script("""
                        var canvas = document.querySelector('#echarts-container canvas');
                        if (!canvas) return false;
                        var ctx = canvas.getContext('2d');
                        var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                        var data = imageData.data;
                        // Check if canvas has non-transparent pixels
                        for (var i = 3; i < data.length; i += 4) {
                            if (data[i] > 0) return true;  // Found non-transparent pixel
                        }
                        return false;
                    """)

                    if not canvas_has_content:
                        # Try one more resize and wait
                        driver.execute_script("window.chart && window.chart.resize();")
                        time.sleep(1)

                    # Take screenshot of chart element
                    chart_element = driver.find_element(By.ID, "echarts-container")
                    chart_element.screenshot(filepath)

                    # Return both filename and full filepath for display
                    return {'filename': filename, 'filepath': filepath}

                finally:
                    driver.quit()

            finally:
                # Clean up temporary file
                os.unlink(temp_html_path)

        except Exception as e:
            st.error(f"Failed to auto-save PNG: {str(e)}")
            return None

    def _create_echarts_html(self, chart_config, width=800, height=600):
        """Create standalone HTML page for ECharts rendering"""
        # Escape the JSON for safe embedding in HTML
        config_json = json.dumps(chart_config, ensure_ascii=False)

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ECharts Auto Export</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 30px 30px 80px 30px;
            background-color: white;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
            overflow: hidden;
            min-height: {height + 160}px;
        }}
        #echarts-container {{
            width: {width}px;
            height: {height}px;
            margin: 5px auto 60px auto;
            padding: 0;
            box-sizing: border-box;
            border: none;
        }}
        .render-indicator {{
            display: none;
        }}
        .render-complete {{
            display: block;
        }}
    </style>
</head>
<body>
    <div id="echarts-container"></div>
    <div id="render-status" class="render-indicator">Chart rendered</div>
    <script>
        // Wait for ECharts library to be fully loaded
        window.addEventListener('load', function() {{
            var chartContainer = document.getElementById('echarts-container');
            var renderStatus = document.getElementById('render-status');

            // Initialize chart and make it globally accessible
            window.chart = echarts.init(chartContainer, null, {{
                renderer: 'canvas',
                width: {width},
                height: {height}
            }});

            var option = {config_json};

            // Set option and wait for render completion
            window.chart.setOption(option, true);

            // Force immediate resize
            window.chart.resize();

            // Listen for render completion
            window.chart.on('rendered', function() {{
                renderStatus.className = 'render-complete';
                console.log('Chart rendered successfully');
            }});

            // Listen for animation finish (more reliable than 'rendered' for complex charts)
            window.chart.on('finished', function() {{
                renderStatus.className = 'render-complete';
                console.log('Chart animation finished');
            }});

            // Additional safety measures with global chart reference
            setTimeout(function() {{
                window.chart.resize();
                renderStatus.className = 'render-complete';
            }}, 500);

            setTimeout(function() {{
                window.chart.resize();
                renderStatus.className = 'render-complete';
            }}, 1500);

            // Final safety timeout to ensure rendering is complete
            setTimeout(function() {{
                renderStatus.className = 'render-complete';
                console.log('Final render confirmation');
            }}, 2000);
        }});
    </script>
</body>
</html>"""
        return html_template

    def get_auto_save_status(self):
        """Check if automatic PNG saving is available"""
        if SELENIUM_AVAILABLE:
            return {
                'available': True,
                'message': '✅ Automatic PNG export available'
            }
        else:
            return {
                'available': False,
                'message': '❌ Install selenium for automatic PNG export: pip install selenium webdriver-manager'
            }