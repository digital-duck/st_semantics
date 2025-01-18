import streamlit as st
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from laserembeddings import Laser
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA  # Add PCA import
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Set page layout
st.set_page_config(layout="wide")

# Set a font that supports Chinese characters
# mpl.rcParams['font.family'] = 'SimHei'  # no available
mpl.rcParams['font.family'] = 'Noto Sans CJK SC'  


# Load environment variables from .env file
load_dotenv()

# Create "images" subfolder if it doesn't exist
Path("images").mkdir(exist_ok=True)

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

# Sidebar configuration
def do_sidebar():
    with st.sidebar:
        st.subheader("Settings")
        st.radio("Choose Embedding Model", options=["XLM-R", "LASER"], index=0, key="cfg_model_choice")
        st.radio("Choose Dimensionality Reduction Method", options=["t-SNE", "Isomap"], index=0, key="cfg_reduction_method")

# Helper function to extract XLM-R embeddings
@st.cache_data
def get_xlmr_embeddings(words):
    try:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModel.from_pretrained("xlm-roberta-base")
        embeddings = []
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
        return np.vstack(embeddings)
    except Exception as e:
        st.error(f"Error extracting XLM-R embeddings: {e}")
        return None

# Helper function to extract LASER embeddings
@st.cache_data
def get_laser_embeddings(words, lang="zh"):
    try:
        laser = Laser()
        return laser.embed_sentences(words, lang=lang)
    except Exception as e:
        st.error(f"Error extracting LASER embeddings: {e}")
        return None

# Helper function for dimensionality reduction
@st.cache_data
def reduce_dimensions(embeddings, method="t-SNE"):
    try:
        n_samples = embeddings.shape[0]

        # Handle very small datasets
        if n_samples < 3:
            st.warning(f"Dataset is too small for {method}. Switching to PCA.")
            reducer = PCA(n_components=2)
            return reducer.fit_transform(embeddings)

        if method == "t-SNE":
            # Adjust perplexity based on the number of samples
            perplexity = min(30, n_samples - 1)  # Ensure perplexity < n_samples
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        elif method == "Isomap":
            reducer = Isomap(n_components=2)
        else:
            raise ValueError("Unsupported method.")
        return reducer.fit_transform(embeddings)
    except Exception as e:
        st.error(f"Error during dimensionality reduction: {e}")
        return None

# Helper function for plotting embeddings
@st.cache_data
def plot_embeddings_mpl(embeddings, labels, colors, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (x, y) in enumerate(embeddings):
        ax.scatter(x, y, color=colors[i], label=labels[i])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title)
    st.pyplot(fig)

    # Save plot
    if st.button("Save Plot"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        file_png = f"images/viz_embedding_{ts}.png"
        fig.savefig(file_png)
        st.success(f"Plot saved as '{file_png}'.")


@st.cache_data
def plot_embeddings(embeddings, labels, colors, title):
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        "x": embeddings[:, 0],
        "y": embeddings[:, 1],
        "label": labels,
        "color": colors
    })

    # Create an interactive scatter plot
    fig = px.scatter(df, x="x", y="y", text="label", color="color", title=title)
    fig.update_traces(
        textposition='top center',  # Position labels above points
        hoverinfo='text',           # Show labels on hover
        textfont_size=10            # Adjust label font size
    )
    fig.update_layout(showlegend=False)  # Hide the legend

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    # Return the figure for saving
    return fig

    # Save plot
    if st.button("Save Plot"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        file_png = f"images/viz_embedding_{ts}.png"
        fig.write_image(file_png)  # Save as PNG
        st.success(f"Plot saved as '{file_png}'.")

# Main app function
def main():
    st.header("Multilingual Embedding Visualization")
    st.write("Visualize Chinese characters/phrases and (optionally) their English translations in 2D space.")

    # Input dataset
    st.subheader("Input Data")
    c1, _, c2 = st.columns([3, 1, 3])
    with c1:
        chinese_words = st.text_area("Enter Chinese words/phrases (one per line)", "你好\n爱\n天气\n书\n猫")
    with c2:
        english_words = st.text_area("Enter corresponding English translations (one per line)", "Hello\nLove\nWeather\nBook\nCat")

    chinese_words = chinese_words.strip().split("\n")
    english_words = english_words.strip().split("\n")



    # Add a button to trigger processing
    if st.button("Generate Visualization"):
        # Validate input
        if (len(chinese_words) + len(english_words)) < 1:
            st.warning("Please enter at least one word or phrase.")
            st.stop()

        # Extract embeddings
        model_choice = st.session_state.get("cfg_model_choice", "XLM-R")
        if model_choice == "XLM-R":
            chinese_embeddings = get_xlmr_embeddings(chinese_words)
            english_embeddings = get_xlmr_embeddings(english_words)
        else:
            chinese_embeddings = get_laser_embeddings(chinese_words, lang="zh")
            english_embeddings = get_laser_embeddings(english_words, lang="en")

        # Check if embeddings were successfully extracted
        if chinese_embeddings is None and english_embeddings is None:
            st.error("Failed to extract embeddings for both Chinese and English words.")
            st.stop()

        # Combine embeddings (skip None values)
        embeddings = []
        labels = []
        if chinese_embeddings is not None:
            embeddings.append(chinese_embeddings)
            labels.extend(chinese_words)
        if english_embeddings is not None:
            embeddings.append(english_embeddings)
            labels.extend(english_words)
        embeddings = np.vstack(embeddings)

        # Dimensionality reduction
        reduction_method = st.session_state.get("cfg_reduction_method", "t-SNE")
        reduced_embeddings = reduce_dimensions(embeddings, method=reduction_method)

        # Check if dimensionality reduction was successful
        if reduced_embeddings is None:
            st.stop()

        # Plotting
        st.header("2D Scatter Plot")
        colors = ["blue"] * len(chinese_words) + ["red"] * len(english_words)
        plot_title = f"{model_choice} Embeddings ({reduction_method})"
        fig = plot_embeddings(reduced_embeddings, labels, colors, plot_title)

        # Save plot button
        if st.button("Save Plot"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
            file_png = f"images/viz_embedding_{ts}.png"
            fig.write_image(file_png)  # Save as PNG
            st.success(f"Plot saved as '{file_png}'.")
            
# Run the app
if __name__ == "__main__":
    check_login()  # Check login status
    do_sidebar()
    main()