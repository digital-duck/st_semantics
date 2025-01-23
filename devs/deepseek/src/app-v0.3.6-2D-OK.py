import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
from laserembeddings import Laser
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import os
from dotenv import load_dotenv
from pathlib import Path
import re

# Load environment variables from .env file
load_dotenv()

# Create "images" subfolder if it doesn't exist
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

DEFAULT_MODEL = "LASER"
DEFAULT_METHOD = "Isomap"
DEFAULT_METHODS = ["Isomap", "t-SNE",]
MODEL_METHOD_MAP = {
    "XLM-R" : DEFAULT_METHODS, 
    "LASER" : DEFAULT_METHODS, 
}

ST_APP_NAME = "Multilingual Embedding Explorer"
ST_HEADER_1 = "View word embeddings in 2-3D spaces"

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
        model_names = sorted(list(MODEL_METHOD_MAP.keys()))
        model_name = st.radio("Choose Embedding Model", 
                options=model_names, 
                index=model_names.index(DEFAULT_MODEL), 
                key="cfg_model_choice")
        method_names = MODEL_METHOD_MAP.get(model_name, DEFAULT_METHODS)
        method_name = st.radio("Choose Dimensionality Reduction Method", 
                options=method_names, 
                index=method_names.index(DEFAULT_METHOD), 
                key="cfg_reduction_method")

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

def parse_input_data(input_text):
    """
    Parse input text into a list of words/phrases, respecting quoted phrases.
    Splits on spaces, commas, newlines, and semicolons.
    """
    input_text = input_text.replace('+', ' ').replace('=', ' ')

    # Use regex to split on delimiters but respect quoted phrases
    pattern = re.compile(r"""
        (?:[^"\s,;\n]+|"[^"]*")  # Match either non-delimiter sequences or quoted phrases
    """, re.VERBOSE)
    
    # Find all matches
    matches = pattern.findall(input_text)
    
    # Remove quotes from quoted phrases
    parsed_words = [match.strip('"') for match in matches if match.strip()]
    
    return parsed_words

def plot_embeddings_2d(embeddings, labels, colors, title):
    """
    Create an interactive 2D scatter plot
    """
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        "x": embeddings[:, 0],
        "y": embeddings[:, 1],
        "label": labels,
        "color": colors
    })

    fig = px.scatter(df, x="x", y="y", text="label", color="color", title=title,
                     color_discrete_map={"red": "red", "blue": "blue"})  # Explicit color mapping
    fig.update_traces(
        textposition='top center',  # Position labels above points
        hoverinfo='text',           # Show labels on hover
        textfont_size=10            # Adjust label font size
    )
    fig.update_layout(
        showlegend=False,  # Hide the legend
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Add gridlines
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),  # Add gridlines
        dragmode='pan',  # Enable panning by default
        hovermode='closest'  # Show hover info for the closest point
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Main app function
def main():
    st.subheader(ST_HEADER_1)

    # Input dataset
    c1, _, c2 = st.columns([3, 1, 3])
    with c1:
        chinese_words = st.text_area(
                "Enter Chinese words/phrases:", 
                value="""你好 男子汉 \n爱 "女子" \n天气\n书\n猫""",
                height=150,
            )
    with c2:
        english_words = st.text_area(
                "Enter English words/phrases:", 
                value="""Hello\nLove\nWeather\nBook "Woman" "Gentle Man" \nCat""",
                height=150,
            )

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
        # st.subheader("2D Scatter Plot")
        colors = ["red"] * len(chinese_words) + ["blue"] * len(english_words)
        plot_title = f"Model: {model_choice}, Method: {reduction_method}"
        plot_embeddings_2d(reduced_embeddings, labels, colors, plot_title)

# Run the app
if __name__ == "__main__":
    check_login()  # Check login status
    do_sidebar()
    main()