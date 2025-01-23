import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans

from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5Model, T5EncoderModel
from laserembeddings import Laser

from sklearn.manifold import LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE, Isomap
from sklearn.decomposition import PCA, KernelPCA
from umap import UMAP
from phate import PHATE

import os
from dotenv import load_dotenv
from pathlib import Path
import re

# Load environment variables from .env file
load_dotenv()

# Create "images" subfolder if it doesn't exist
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

file_path_chn = Path("chn.txt")
file_path_enu = Path("enu.txt")
if file_path_chn.exists():
    sample_chn_input_data = open(file_path_chn).read()
else:
    sample_chn_input_data = """你好 男子汉 \n爱 "女子" \n天气\n书\n猫"""

if file_path_enu.exists():
    sample_enu_input_data = open(file_path_enu).read()
else:
    sample_enu_input_data = """Hello\nLove\nWeather\nBook "Woman" "Gentle Man" \nCat"""


# Default settings
DEFAULT_N_CLUSTERS = 5
DEFAULT_MAX_WORDS = 15
DEFAULT_MODEL = "Sentence-BERT Multilingual" # "LaBSE" # "LASER"
DEFAULT_METHOD = "PHATE" # "Isomap"
ST_APP_NAME = "Multilingual Embedding Explorer"
ST_HEADER_1 = "View word embeddings in 2-3D spaces"

COLOR_MAP = {
    "chinese": "red",
    "english": "blue",
}
PLOT_WIDTH, PLOT_HEIGHT = 800, 800

# Model information (name, Hugging Face path, and help text)
MODEL_INFO = {
    "XLM-R": {
        "path": "xlm-roberta-base",
        "help": "A robust multilingual model trained on 100+ languages. Great for cross-lingual tasks like text classification and NER."
    },
    "LASER": {
        "path": None,  # LASER uses its own library
        "help": "Language-agnostic sentence embeddings for 100+ languages. Efficient for multilingual tasks like sentence similarity."
    },
    "mBERT": {
        "path": "bert-base-multilingual-cased",
        "help": "Multilingual BERT trained on 104 languages. Widely used for cross-lingual transfer learning."
    },
    "LaBSE": {
        "path": "sentence-transformers/LaBSE",
        "help": "Language-agnostic BERT sentence embeddings for 109 languages. Excellent for sentence similarity and paraphrase detection."
    },
    "DistilBERT Multilingual": {
        "path": "distilbert-base-multilingual-cased",
        "help": "A lightweight version of mBERT. Faster and more efficient, suitable for real-time applications."
    },
    "XLM": {
        "path": "xlm-mlm-100-1280",
        "help": "Cross-lingual language model trained using masked and translation language modeling. Good for translation tasks."
    },
    "InfoXLM": {
        "path": "microsoft/infoxlm-base",
        "help": "An extension of XLM-R with improved cross-lingual transferability. Great for low-resource languages."
    },
    "mT5": {
        "path": "google/mt5-small",
        "help": "Multilingual T5 model trained on 101 languages. Versatile for text generation and embedding tasks."
    },
    "Sentence-BERT Multilingual": {
        "path": "sentence-transformers/distiluse-base-multilingual-cased-v1",
        "help": "Multilingual Sentence-BERT optimized for semantic similarity tasks like clustering and retrieval."
    },
    ### ERROR: Failed building wheel for flash-attn
    # "ModernBERT": {
    #     "path": "answerdotai/ModernBERT-base",
    #     "help": "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference."
    # },

    # "Erlangshen": {
    #     "path": "IDEA-CCNL/Erlangshen-Roberta-110M",
    #     "help": "A Chinese-focused multilingual model optimized for Chinese-English tasks like translation and sentiment analysis."
    # }
}

# Dimensionality reduction method with help text
METHOD_INFO = {
    "t-SNE": {
        "help": "t-Distributed Stochastic Neighbor Embedding. Preserves local structure and is great for visualizing clusters."
    },
    "Isomap": {
        "help": "Isometric Mapping. Preserves geodesic distances and is ideal for capturing manifold structures."
    },
    "UMAP": {
        "help": "Uniform Manifold Approximation and Projection. Fast and preserves both local and global structure."
    },
    "LLE": {
        "help": "Locally Linear Embedding. Preserves local relationships by representing points as linear combinations of neighbors."
    },
    "MDS": {
        "help": "Multidimensional Scaling. Preserves pairwise distances between points, suitable for global structure visualization."
    },
    "PCA": {
        "help": "Principal Component Analysis. A linear method that projects data onto directions of maximum variance."
    },
    "Kernel PCA": {
        "help": "Kernel PCA. A nonlinear extension of PCA that uses kernel functions to capture complex structures."
    },
    "Spectral Embedding": {
        "help": "Spectral Embedding. Based on graph Laplacian, effective for capturing underlying data structure."
    },
    "PHATE": {
        "help": "Potential of Heat-diffusion for Affinity-based Transition Embedding. Great for visualizing complex, high-dimensional data."
    },
    # "Autoencoders": {
    #     "help": "Neural network-based approach for learning compressed representations of data."
    # },
    # "LDA": {
    #     "help": "Linear Discriminant Analysis. A supervised method that maximizes class separability."
    # }
}



# Set page layout
st.set_page_config(
    layout="wide",
    page_title=ST_APP_NAME,
    page_icon="🚩",
)

# Sidebar configuration
def do_sidebar():
    with st.sidebar:
        st.header(ST_APP_NAME)
        st.subheader("Settings")

        # Model selection with help text
        model_names = sorted(list(MODEL_INFO.keys()))
        model_name = st.radio(
            "Choose Embedding Model",
            options=model_names,
            index=model_names.index(DEFAULT_MODEL),
            key="cfg_model_choice",
            help="Select a multilingual embedding model. Hover over each option for more details."
        )
        # Display help text for the selected model
        st.caption(f"**{model_name}**: {MODEL_INFO[model_name]['help']}")

        method_names = sorted(list(METHOD_INFO.keys()))
        method_name = st.radio(
            "Choose Dimensionality Reduction Method",
            options=method_names,
            index=method_names.index(DEFAULT_METHOD),
            key="cfg_reduction_method",
            help="Select a dimensionality reduction method. Hover over each option for more details."
        )
        # Display help text for the selected method
        st.caption(f"**{method_name}**: {METHOD_INFO[method_name]['help']}")

        # Dimensions (2D or 3D)
        dimensions = st.radio(
            "Choose Dimensions",
            options=["2D", "3D"],
            index=0,
            key="cfg_dimensions"
        )

        show_semantic_forces = False  # turn it off

        if dimensions == "2D":
            # Semantic force visualization
            # show_semantic_forces = st.checkbox("Show Semantic Forces?", value=False, key="cfg_semantic_forces")


            if show_semantic_forces:
                max_words = st.slider("Max # Words", min_value=10, max_value=20, value=DEFAULT_MAX_WORDS, key="cfg_max_words")


        if not show_semantic_forces:
            # clustering
            do_clustering = st.checkbox("Clustering?", value=False, key="cfg_clustering")
            if do_clustering:
                n_clusters = st.slider("# Clusters", min_value=3, max_value=10, value=DEFAULT_N_CLUSTERS, key="cfg_n_clusters")


# Simulate login handling (for demonstration purposes)
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Check if API key is already set in environment variables
    if os.getenv("HF_API_KEY"):
        st.session_state.logged_in = True
        return

    if not st.session_state.logged_in:
        st.sidebar.title("Login")
        api_key = st.sidebar.text_input("Enter Hugging Face API Key", type="password")
        if st.sidebar.button("Login"):
            if api_key:  # Simulate API key validation
                os.environ["HF_API_KEY"] = api_key
                st.session_state.logged_in = True
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Please enter a valid API key.")
        st.stop()  # Stop execution if not logged in

# Helper function to extract embeddings
@st.cache_data
def get_mt5_embeddings(words):
    try:
        # Load the mT5 tokenizer and encoder model
        tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        model = T5EncoderModel.from_pretrained("google/mt5-small")
        
        embeddings = []
        for word in words:
            # Tokenize the input word
            inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
            
            # Get the encoder outputs (no need for decoder inputs here)
            outputs = model(**inputs)
            
            # Use the last hidden state as the embedding
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
        
        return np.vstack(embeddings)
    except Exception as e:
        st.error(f"Error extracting mT5 embeddings: {e}")
        return None
    
@st.cache_data
def get_embeddings(words, model_name, lang="en"):
    try:
        if model_name == "LASER":
            laser = Laser()
            return laser.embed_sentences(words, lang=lang)
        else:
            model_path = MODEL_INFO[model_name]["path"]
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
                
            embeddings = []
            for word in words:
                inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
            return np.vstack(embeddings)
    except Exception as e:
        st.error(f"Error extracting embeddings for {model_name}: {e}")
        return None

# Helper function for dimensionality reduction
@st.cache_data
def reduce_dimensions(embeddings, method="t-SNE", dimensions=2):
    try:
        n_samples = embeddings.shape[0]

        # Handle very small datasets
        if n_samples < 3:
            st.warning(f"Dataset is too small for {method}. Switching to PCA.")
            reducer = PCA(n_components=dimensions)
            return reducer.fit_transform(embeddings)

        if method == "t-SNE":
            perplexity = min(30, n_samples - 1)  # Ensure perplexity < n_samples
            reducer = TSNE(n_components=dimensions, random_state=42, perplexity=perplexity)
        elif method == "Isomap":
            reducer = Isomap(n_components=dimensions)
        elif method == "UMAP":
            reducer = UMAP(n_components=dimensions, random_state=42)
        elif method == "LLE":
            reducer = LocallyLinearEmbedding(n_components=dimensions, random_state=42)
        elif method == "MDS":
            reducer = MDS(n_components=dimensions, random_state=42)
        elif method == "PCA":
            reducer = PCA(n_components=dimensions)
        elif method == "Kernel PCA":
            reducer = KernelPCA(n_components=dimensions, kernel='rbf')
        elif method == "Spectral Embedding":
            reducer = SpectralEmbedding(n_components=dimensions, random_state=42)
        elif method == "PHATE":
            reducer = PHATE(n_components=dimensions)
        else:
            raise ValueError("Unsupported method.")
        return reducer.fit_transform(embeddings)
    except Exception as e:
        st.error(f"Error during dimensionality reduction: {e}")
        return None
    
# Parse input data
def parse_input_data(input_text):
    input_text = input_text.replace('+', ' ').replace('=', ' ')
    pattern = re.compile(r"""(?:[^"\s,;\n]+|"[^"]*")""", re.VERBOSE)
    matches = pattern.findall(input_text)
    parsed_words = [match.strip('"') for match in matches if match.strip()]
    return parsed_words

# Plot embeddings in 2D
def plot_embeddings_2d(
        embeddings, 
        labels, 
        colors, 
        title, 
        plot_width=PLOT_WIDTH, 
        plot_height=PLOT_HEIGHT):
    # Create a DataFrame for Plotly
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1], "label": labels, "color": colors})

    # Create an interactive scatter plot
    fig = px.scatter(df, x="x", y="y", text="label", color="color", title=title,
                     color_discrete_map={"red": "red", "blue": "blue"})  # Explicit color mapping

    # Update traces for better label placement and hover info
    fig.update_traces(
        textposition='top center',  # Position labels above points
        hoverinfo='text',           # Show labels on hover
        textfont_size=10            # Adjust label font size
    )

    # Update layout for a square plot
    fig.update_layout(
        showlegend=False,  # Hide the legend
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Add gridlines
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Add gridlines
        dragmode='pan',  # Enable panning by default
        hovermode='closest',  # Show hover info for the closest point
        width=plot_width,  # Set a fixed width
        height=plot_height,  # Set the same height as width
        xaxis_scaleanchor="y",  # Lock x-axis scale to y-axis
        xaxis_scaleratio=1,     # Ensure 1:1 aspect ratio
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_embeddings_2d_cluster(
        embeddings, 
        labels, 
        colors, 
        title, 
        n_clusters=DEFAULT_N_CLUSTERS, 
        plot_width=PLOT_WIDTH, 
        plot_height=PLOT_HEIGHT):
    """
    Visualize 2D embeddings with clustering.

    Parameters:
        embeddings (np.array): The reduced 2D embeddings.
        labels (list): The labels (words/phrases) for each embedding.
        colors (list): The colors for each embedding (e.g., language or category).
        title (str): The title of the plot.
        n_clusters (int): The number of clusters to create (default: 5).
    """
    # Perform K-Means clustering on the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        "x": embeddings[:, 0],  # X-axis coordinates
        "y": embeddings[:, 1],  # Y-axis coordinates
        "label": labels,        # Word/phrase labels
        "color": colors,        # Original colors (e.g., language or category)
        "cluster": clusters     # Cluster labels
    })

    # Create an interactive scatter plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="label",
        color="cluster",  # Use cluster labels for coloring
        title=title,
        color_continuous_scale=px.colors.sequential.Viridis  # Use a color scale for clusters
    )

    # Update traces for better label placement and hover info
    fig.update_traces(
        textposition='top center',  # Position labels above points
        hoverinfo='text',           # Show labels on hover
        textfont_size=10            # Adjust label font size
    )

    # Update layout for a square plot
    fig.update_layout(
        showlegend=True,  # Show legend for clusters
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Add gridlines
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Add gridlines
        dragmode='pan',  # Enable panning by default
        hovermode='closest',  # Show hover info for the closest point
        width=plot_width,  # Set a fixed width
        height=plot_height,  # Set the same height as width
        xaxis_scaleanchor="y",  # Lock x-axis scale to y-axis
        xaxis_scaleratio=1,     # Ensure 1:1 aspect ratio
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Plot embeddings in 3D
def plot_embeddings_3d_cluster(
        embeddings, 
        labels, 
        colors, 
        title, 
        n_clusters=DEFAULT_N_CLUSTERS,
        plot_width=PLOT_WIDTH, 
        plot_height=PLOT_HEIGHT):
    """
    Visualize 3D embeddings with clustering.

    Parameters:
        embeddings (np.array): The reduced 3D embeddings.
        labels (list): The labels (words/phrases) for each embedding.
        colors (list): The colors for each embedding (e.g., language or category).
        title (str): The title of the plot.
        n_clusters (int): The number of clusters to create (default: 5).
    """
    # Perform K-Means clustering on the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        "x": embeddings[:, 0],  # X-axis coordinates
        "y": embeddings[:, 1],  # Y-axis coordinates
        "z": embeddings[:, 2],  # Z-axis coordinates
        "label": labels,        # Word/phrase labels
        "color": colors,        # Original colors (e.g., language or category)
        "cluster": clusters     # Cluster labels
    })

    # Create an interactive 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        text="label",
        color="cluster",  # Use cluster labels for coloring
        title=title,
        color_continuous_scale=px.colors.sequential.Viridis  # Use a color scale for clusters
    )

    # Update traces for better label placement and hover info
    fig.update_traces(
        textposition='top center',  # Position labels above points
        hoverinfo='text',           # Show labels on hover
        textfont_size=10            # Adjust label font size
    )

    # Update layout for a 3D plot
    fig.update_layout(
        showlegend=True,  # Show legend for clusters
        scene=dict(
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Add gridlines
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Add gridlines
            zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')   # Add gridlines
        ),
        dragmode='pan',  # Enable panning by default
        hovermode='closest',  # Show hover info for the closest point
        width=plot_width,  # Set a fixed width
        height=plot_height,  # Set the same height as width
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_embeddings_3d(
        embeddings, 
        labels, 
        colors, 
        title, 
        plot_width=PLOT_WIDTH, 
        plot_height=PLOT_HEIGHT):
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1], "z": embeddings[:, 2], "label": labels, "color": colors})
    fig = px.scatter_3d(df, x="x", y="y", z="z", text="label", color="color", title=title,
                        color_discrete_map={"red": "red", "blue": "blue"})
    fig.update_traces(textposition='top center', hoverinfo='text', textfont_size=10)
    fig.update_layout(
        showlegend=False, 
        dragmode='pan', 
        hovermode='closest',
        width=plot_width,  # Set a fixed width
        height=plot_height,  # Set the same height as width
    )
    st.plotly_chart(fig, use_container_width=True)



def plot_semantic_forces(embeddings, labels, title, max_words=DEFAULT_MAX_WORDS):
    """
    Visualize semantic forces between words/phrases using arrows.

    Parameters:
        embeddings (np.array): The reduced 2D embeddings.
        labels (list): The labels (words/phrases) for each embedding.
        title (str): The title of the plot.
        max_words (int): Maximum number of words/phrases to consider (default: 20).
    """
    # Limit the number of words/phrases to consider
    if len(labels) > max_words:
        st.warning(f"Only showing semantic forces for the first {max_words} words/phrases.")
        embeddings = embeddings[:max_words]
        labels = labels[:max_words]

    # Create a scatter plot
    fig = go.Figure()

    # Add points for words/phrases
    fig.add_trace(go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=10, color="blue")
    ))

    # Add arrows to represent semantic forces
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # Calculate the vector between two points
            dx = embeddings[j, 0] - embeddings[i, 0]
            dy = embeddings[j, 1] - embeddings[i, 1]
            # Add an arrow
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

    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        width=800,
        height=800,
        xaxis_scaleanchor="y",
        xaxis_scaleratio=1,
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Main app function
def main():
    st.subheader(ST_HEADER_1)

    # Input dataset
    c1, c_1, c2, c_2 = st.columns([5, 1, 5, 1])
    with c1:
        chinese_words = st.text_area(
            "Enter Chinese words/phrases:",
            value=sample_chn_input_data,
            height=150,
        )
    with c_1:
        chinese_selected = st.checkbox("In?", value=True, key="chinese_selected")
    with c2:
        english_words = st.text_area(
            "Enter English words/phrases:",
            value=sample_enu_input_data,
            height=150,
        )
    with c_2:
        english_selected = st.checkbox("In?", value=True, key="english_selected")

    # Parse input data
    chinese_words = parse_input_data(chinese_words)
    english_words = parse_input_data(english_words)

    # Add a button to trigger processing
    if st.button("Visualize"):
        # Validate input
        if (len(chinese_words) + len(english_words)) < 1:
            st.warning("Please enter at least one word or phrase.")
            st.stop()

        # Extract embeddings
        model_choice = st.session_state.get("cfg_model_choice", DEFAULT_MODEL)

        chinese_embeddings = None
        if chinese_selected:
            if model_choice == "mT5":
                chinese_embeddings = get_mt5_embeddings(chinese_words)
            else:
                chinese_embeddings = get_embeddings(chinese_words, model_choice, lang="zh")

        english_embeddings = None
        if english_selected:
            if model_choice == "mT5":
                english_embeddings = get_mt5_embeddings(english_words)        
            else:
                english_embeddings = get_embeddings(english_words, model_choice, lang="en")

        # Check if embeddings were successfully extracted
        if chinese_embeddings is None and english_embeddings is None:
            st.error("Failed to extract embeddings for both Chinese and English words.")
            st.stop()

        # Combine embeddings
        embeddings = []
        labels = []
        colors = []
        if chinese_embeddings is not None:
            embeddings.append(chinese_embeddings)
            labels.extend(chinese_words)
        if english_embeddings is not None:
            embeddings.append(english_embeddings)
            labels.extend(english_words)
        embeddings = np.vstack(embeddings)

        # Dimensionality reduction
        reduction_method = st.session_state.get("cfg_reduction_method", DEFAULT_METHOD)
        dimensions = 3 if st.session_state.get("cfg_dimensions", "2D") == "3D" else 2
        reduced_embeddings = reduce_dimensions(embeddings, method=reduction_method, dimensions=dimensions)

        # Check if dimensionality reduction was successful
        if reduced_embeddings is None:
            st.stop()

        # Plotting
        if chinese_selected:
            colors += [COLOR_MAP["chinese"]] * len(chinese_words) 
        if english_selected:
            colors += [COLOR_MAP["english"]] * len(english_words)

        plot_title = f"[Model] {model_choice}, [Method] {reduction_method}"
        do_clustering = st.session_state.get("cfg_clustering", False)
        n_clusters = st.session_state.get("cfg_n_clusters", DEFAULT_N_CLUSTERS)

        show_semantic_forces = st.session_state.get("cfg_semantic_forces", False)

        if dimensions == 2:
            if show_semantic_forces:
                max_words = st.session_state.get("cfg_max_words", DEFAULT_MAX_WORDS)
                plot_semantic_forces(reduced_embeddings, labels, plot_title, max_words=max_words)
            elif do_clustering:
                plot_embeddings_2d_cluster(reduced_embeddings, labels, colors, plot_title, n_clusters=n_clusters)
            else:
                plot_embeddings_2d(reduced_embeddings, labels, colors, plot_title)
        else:
            if do_clustering:
                plot_embeddings_3d_cluster(reduced_embeddings, labels, colors, plot_title, n_clusters=n_clusters)
            else:
                plot_embeddings_3d(reduced_embeddings, labels, colors, plot_title)



# Run the app
if __name__ == "__main__":
    check_login()  # Check login status
    do_sidebar()
    main()