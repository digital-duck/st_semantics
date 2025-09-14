import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from config import (
       PLOT_WIDTH, PLOT_HEIGHT, 
       DEFAULT_N_CLUSTERS, DEFAULT_MIN_CLUSTERS, DEFAULT_MAX_CLUSTERS,
       DEFAULT_MAX_WORDS
)
import io
import base64

class PlotManager:
    def __init__(self):
        self.min_clusters = DEFAULT_MIN_CLUSTERS
        self.max_clusters = DEFAULT_MAX_CLUSTERS
        # Publication-quality settings
        self.publication_settings = {
            'textfont_size': 16,
            'point_size': 12,
            'width': 1200,
            'height': 900,
            'dpi': 300,
            'grid_color': '#D0D0D0',
            'grid_width': 1,
            'background_color': 'white',
            'font_family': 'Arial, sans-serif',
            'grid_dash': 'dot'
        }

    def create_title(self, method_name, model_name, dataset_name=""):
        """Create standardized plot title"""
        title_parts = [f"[Method] {method_name}", f"[Model] {model_name}"]
        if dataset_name:
            title_parts.append(f"[Dataset] {dataset_name}")
        return ", ".join(title_parts)
    
    def get_visualization_settings(self):
        """Get visualization settings from sidebar"""
        with st.sidebar:
            st.subheader("📊 Visualization Settings")
            
            publication_mode = st.checkbox("Publication Mode", value=False, 
                                         help="Enable high-quality settings for publication")
            
            col1, col2 = st.columns(2)
            with col1:
                if publication_mode:
                    textfont_size = st.slider("Text Size", 12, 24, 16, help="Font size for labels")
                    point_size = st.slider("Point Size", 8, 20, 12, help="Size of data points")
                else:
                    textfont_size = st.slider("Text Size", 8, 20, 12, help="Font size for labels")
                    point_size = st.slider("Point Size", 2, 12, 4, help="Size of data points")
            
            with col2:
                if publication_mode:
                    plot_width = st.slider("Width", 800, 1600, 1200, step=100)
                    plot_height = st.slider("Height", 600, 1200, 900, step=100)
                else:
                    plot_width = st.slider("Width", 600, 1200, PLOT_WIDTH, step=100)
                    plot_height = st.slider("Height", 400, 1000, PLOT_HEIGHT, step=100)
            
            # Export options
            if publication_mode:
                st.markdown("**Export Options**")
                export_format = st.selectbox("Format", ["PNG", "SVG", "PDF"], index=0)
                export_dpi = st.slider("DPI", 150, 600, 300, step=50, 
                                     help="Dots per inch for high-resolution export")
            else:
                export_format = "PNG"
                export_dpi = 150
        
        return {
            'publication_mode': publication_mode,
            'textfont_size': textfont_size,
            'point_size': point_size,
            'plot_width': plot_width,
            'plot_height': plot_height,
            'export_format': export_format,
            'export_dpi': export_dpi
        }
    
    def export_figure(self, fig, filename, settings):
        """Export figure in high quality format"""
        if settings['publication_mode']:
            if settings['export_format'] == 'PNG':
                img_bytes = fig.to_image(format="png", width=settings['plot_width'], 
                                       height=settings['plot_height'], scale=settings['export_dpi']/96)
            elif settings['export_format'] == 'SVG':
                img_bytes = fig.to_image(format="svg", width=settings['plot_width'], 
                                       height=settings['plot_height'])
            elif settings['export_format'] == 'PDF':
                img_bytes = fig.to_image(format="pdf", width=settings['plot_width'], 
                                       height=settings['plot_height'])
            
            st.download_button(
                label=f"📥 Download {settings['export_format']} ({settings['export_dpi']} DPI)",
                data=img_bytes,
                file_name=f"{filename}.{settings['export_format'].lower()}",
                mime=f"image/{settings['export_format'].lower()}"
            )
    
    def plot_2d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS, 
                semantic_forces=False, max_words=DEFAULT_MAX_WORDS, method_name="", model_name="", dataset_name=""):
        # Get visualization settings
        settings = self.get_visualization_settings()
        
        # Create standardized title
        if method_name and model_name:
            title = self.create_title(method_name, model_name, dataset_name)
        
        if semantic_forces:
            fig = self._plot_semantic_forces(embeddings, labels, title, max_words, settings)
        elif clustering:
            fig = self._plot_2d_cluster(embeddings, labels, colors, title, n_clusters, settings)
        else:
            fig = self._plot_2d_simple(embeddings, labels, colors, title, settings)
        
        # Add export functionality
        if settings['publication_mode']:
            filename = f"{method_name}_{model_name}_{dataset_name}".replace(" ", "_").replace(",", "")
            self.export_figure(fig, filename, settings)
        
        return fig

    def plot_3d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS, 
                method_name="", model_name="", dataset_name=""):
        # Get visualization settings
        settings = self.get_visualization_settings()
        
        # Create standardized title
        if method_name and model_name:
            title = self.create_title(method_name, model_name, dataset_name)
        
        if clustering:
            fig = self._plot_3d_cluster(embeddings, labels, colors, title, n_clusters, settings)
        else:
            fig = self._plot_3d_simple(embeddings, labels, colors, title, settings)
        
        # Add export functionality
        if settings['publication_mode']:
            filename = f"{method_name}_{model_name}_{dataset_name}_3D".replace(" ", "_").replace(",", "")
            self.export_figure(fig, filename, settings)
        
        return fig

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

    def _plot_2d_cluster(self, embeddings, labels, colors, title, n_clusters, settings):
        # Add dynamic controls for clustering
        boundary_threshold = st.slider(
            "Cluster Boundary Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Adjust the softness of cluster boundaries. Lower values create larger boundaries."
        )

        # Perform clustering and get metrics
        metrics, kmeans = self._perform_clustering(embeddings, n_clusters)
        
        # Display metrics
        self._display_cluster_metrics(metrics)

        # Create mesh grid for boundary visualization
        x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
        y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Create figure with boundaries
        fig = go.Figure()

        # Add contour plot for cluster boundaries
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1),
            y=np.arange(y_min, y_max, 0.1),
            z=Z,
            showscale=False,
            opacity=0.3 * boundary_threshold,
            line=dict(width=0),
            colorscale='Viridis'
        ))

        # Add scatter plot
        df = pd.DataFrame({
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "label": labels,
            "cluster": metrics["cluster_labels"]
        })

        fig.add_trace(go.Scatter(
            x=df["x"],
            y=df["y"],
            mode='markers+text',
            text=df["label"],
            textposition="top center",
            textfont=dict(
                size=settings['textfont_size'],
                family=self.publication_settings['font_family']
            ),
            marker=dict(
                size=settings['point_size'],
                color=df["cluster"],
                colorscale='Viridis',
                showscale=True,
                line=dict(width=1, color='white') if settings['publication_mode'] else dict()
            )
        ))

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family'])
            ),
            showlegend=True,
            xaxis=dict(
                showgrid=True, 
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash']
            ),
            yaxis=dict(
                showgrid=True, 
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash']
            ),
            dragmode='pan',
            hovermode='closest',
            width=settings['plot_width'],
            height=settings['plot_height'],
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family'])
        )

        st.plotly_chart(fig, use_container_width=True)
        return fig

    def _plot_2d_simple(self, embeddings, labels, colors, title, settings):
        df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1], 
                          "label": labels, "color": colors})
        
        fig = px.scatter(df, x="x", y="y", text="label", color="color", title=title,
                        color_discrete_map={"red": "red", "blue": "blue"})
        
        fig.update_traces(
            textposition='top center',
            hoverinfo='text',
            textfont_size=textfont_size,
            # Add marker properties to control point size
            marker=dict(
                size=point_size, # Use the point_size parameter here
                opacity=0.8 # Optional: adjust opacity if needed
            )
        )
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family'])
            ),
            showlegend=False,
            xaxis=dict(
                showgrid=True, 
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash']
            ),
            yaxis=dict(
                showgrid=True, 
                gridwidth=self.publication_settings['grid_width'],
                gridcolor=self.publication_settings['grid_color'],
                griddash=self.publication_settings['grid_dash']
            ),
            dragmode='pan',
            hovermode='closest',
            width=settings['plot_width'],
            height=settings['plot_height'],
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family'])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return fig


    def _plot_3d_simple(self, embeddings, labels, colors, title, settings):
        df = pd.DataFrame({
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "z": embeddings[:, 2],
            "label": labels,
            "color": colors
        })
        
        fig = px.scatter_3d(
            df,
            x="x", y="y", z="z",
            text="label",
            color="color",
            title=title,
            color_discrete_map={"red": "red", "blue": "blue"}
        )
        
        fig.update_traces(
            textposition='top center',
            hoverinfo='text',
            textfont=dict(
                size=settings['textfont_size'],
                family=self.publication_settings['font_family']
            ),
            marker=dict(
                size=settings['point_size'],
                opacity=0.8,
                line=dict(width=1, color='white') if settings['publication_mode'] else dict()
            )
        )
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family'])
            ),
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    griddash=self.publication_settings['grid_dash']
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    griddash=self.publication_settings['grid_dash']
                ),
                zaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    griddash=self.publication_settings['grid_dash']
                )
            ),
            dragmode='pan',
            hovermode='closest',
            width=settings['plot_width'],
            height=settings['plot_height'],
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family'])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return fig

    def _plot_3d_cluster(self, embeddings, labels, colors, title, n_clusters, settings):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        df = pd.DataFrame({
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "z": embeddings[:, 2],
            "label": labels,
            "color": colors,
            "cluster": clusters
        })
        
        fig = px.scatter_3d(
            df,
            x="x", y="y", z="z",
            text="label",
            color="cluster",
            title=title,
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_traces(
            textposition='top center',
            hoverinfo='text',
            textfont=dict(
                size=settings['textfont_size'],
                family=self.publication_settings['font_family']
            ),
            marker=dict(
                size=settings['point_size'],
                opacity=0.8,
                line=dict(width=1, color='white') if settings['publication_mode'] else dict()
            )
        )
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18 if settings['publication_mode'] else 14, family=self.publication_settings['font_family'])
            ),
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    griddash=self.publication_settings['grid_dash']
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    griddash=self.publication_settings['grid_dash']
                ),
                zaxis=dict(
                    showgrid=True, 
                    gridwidth=self.publication_settings['grid_width'],
                    gridcolor=self.publication_settings['grid_color'],
                    griddash=self.publication_settings['grid_dash']
                )
            ),
            dragmode='pan',
            hovermode='closest',
            width=settings['plot_width'],
            height=settings['plot_height'],
            plot_bgcolor=self.publication_settings['background_color'],
            font=dict(family=self.publication_settings['font_family'])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return fig