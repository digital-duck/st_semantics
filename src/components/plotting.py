import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from config import PLOT_WIDTH, PLOT_HEIGHT, DEFAULT_N_CLUSTERS, DEFAULT_MAX_WORDS

class PlotManager:
    def __init__(self):
        self.min_clusters = 2
        self.max_clusters = 10

    def plot_2d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS, 
                semantic_forces=False, max_words=DEFAULT_MAX_WORDS):
        if semantic_forces:
            return self._plot_semantic_forces(embeddings, labels, title, max_words)
        elif clustering:
            return self._plot_2d_cluster(embeddings, labels, colors, title, n_clusters)
        else:
            return self._plot_2d_simple(embeddings, labels, colors, title)

    def plot_3d(self, embeddings, labels, colors, title, clustering=False, n_clusters=DEFAULT_N_CLUSTERS):
        if clustering:
            return self._plot_3d_cluster(embeddings, labels, colors, title, n_clusters)
        else:
            return self._plot_3d_simple(embeddings, labels, colors, title)

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

    def _plot_2d_cluster(self, embeddings, labels, colors, title, n_clusters):
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
            marker=dict(
                size=10,
                color=df["cluster"],
                colorscale='Viridis',
                showscale=True
            )
        ))

        fig.update_layout(
            title=title,
            showlegend=True,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            dragmode='pan',
            hovermode='closest',
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_2d_simple(self, embeddings, labels, colors, title):
        df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1], 
                          "label": labels, "color": colors})
        
        fig = px.scatter(df, x="x", y="y", text="label", color="color", title=title,
                        color_discrete_map={"red": "red", "blue": "blue"})
        
        fig.update_traces(
            textposition='top center',
            hoverinfo='text',
            textfont_size=10
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            dragmode='pan',
            hovermode='closest',
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def _plot_semantic_forces(self, embeddings, labels, title, max_words=DEFAULT_MAX_WORDS):
        """Visualize semantic forces between words/phrases using arrows"""
        if len(labels) > max_words:
            st.warning(f"Only showing semantic forces for the first {max_words} words/phrases.")
            embeddings = embeddings[:max_words]
            labels = labels[:max_words]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=10, color="blue")
        ))

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                fig.add_annotation(
                    x=embeddings[j, 0],
                    y=embeddings[j, 1],
                    ax=embeddings[i, 0],
                    ay=embeddings[i, 1],
                    axref="x",
                    ayref="y",
                    xref="x",
                    yref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red"
                )

        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            xaxis_scaleanchor="y",
            xaxis_scaleratio=1,
        )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _plot_3d_simple(embeddings, labels, colors, title, textfont_size=10, point_size=5):
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
            textfont_size=textfont_size,
            # Add marker properties to control point size
            marker=dict(
                size=point_size, # Use the point_size parameter here
                opacity=0.8 # Optional: adjust opacity if needed
            )
        )
        
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            ),
            dragmode='pan',
            hovermode='closest',
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
        )
        
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _plot_3d_cluster(embeddings, labels, colors, title, n_clusters):
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
            textfont_size=10
        )
        
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
            ),
            dragmode='pan',
            hovermode='closest',
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
        )
        
        st.plotly_chart(fig, use_container_width=True)