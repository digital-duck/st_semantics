# config.py
import streamlit as st 
import os

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# App Constants
ST_APP_NAME = "Semantics Explorer"

ST_ICON = "🧭"

# Default Settings
DEFAULT_N_CLUSTERS = 5
DEFAULT_MIN_CLUSTERS = 2
DEFAULT_MAX_CLUSTERS = 10
DEFAULT_MAX_WORDS = 15 

DEFAULT_MODEL = f"Sentence-BERT Multilingual"
DEFAULT_METHOD = f"PHATE"
DEFAULT_DIMENSIONS = "2D"
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 800  # Square aspect ratio for manifold learning visualizations


# Plot Settings
PLOT_CONFIG = {
    "width": DEFAULT_WIDTH,
    "height": DEFAULT_HEIGHT,
    "color_map": {
        "chinese": "red",
        "english": "blue",
        "french": "green", 
        "spanish": "orange",
        "german": "purple"
    }
}

COLOR_MAP = PLOT_CONFIG["color_map"]
PLOT_WIDTH, PLOT_HEIGHT = PLOT_CONFIG["width"], PLOT_CONFIG["height"]

# Sample Data
SAMPLE_DATA = {
    "chinese": """你好
天气
晴朗
""",
    "english": """Hello
Weather
Sunny
"""
}

# File Paths
FILE_PATHS = {
    "images_dir": Path("images"),
    "chinese_file": Path("data/data-1-chn.txt"),
    "english_file": Path("data/data-1-enu.txt")
}

file_path_chn = FILE_PATHS["chinese_file"]
file_path_enu = FILE_PATHS["english_file"]
if file_path_chn.exists():
    sample_chn_input_data = open(file_path_chn).read()
else:
    sample_chn_input_data = SAMPLE_DATA["chinese"]

if file_path_enu.exists():
    sample_enu_input_data = open(file_path_enu).read()
else:
    sample_enu_input_data = SAMPLE_DATA["english"]


# Cache Settings
CACHE_CONFIG = {
    "ttl": 3600,  # Time to live for cached data (in seconds)
    "max_entries": 100
}

# Add Ollama model information to MODEL_INFO
OLLAMA_MODELS = {
    "BGE-M3 (Ollama)": {
        "path": "bge-m3",
        "help": "BGE-M3 is a new model from BAAI distinguished for its versatility in Multi-Functionality, Multi-Linguality, and Multi-Granularity."
    },
    # "Paraphrase-Multilingual (Ollama)": {
    #     "path": "paraphrase-multilingual",
    #     "help": "Sentence-transformers model (multilingual) that can be used for tasks like clustering or semantic search."
    # },
    "Snowflake-Arctic-Embed2 (Ollama)": {
        "path": "snowflake-arctic-embed2",
        "help": "Snowflake Arctic model through Ollama offering efficient embedding generation with strong multilingual capabilities, especially for Chinese-English pairs."
    },
    # "Snowflake-Arctic-Embed (Ollama)": {
    #     "path": "snowflake-arctic-embed",
    #     "help": "Original Snowflake Arctic model through Ollama for multilingual embeddings."
    # },
    "Nomic (Ollama)": {
        "path": "nomic-embed-text",
        "help": "Nomic's embedding model optimized for semantic text embeddings."
    },
    # "Mistral (Ollama)": {
    #     "path": "mistral",
    #     "help": "Mistral model through Ollama offering efficient embedding generation with good multilingual capabilities."
    # },
    # "Neural-Chat (Ollama)": {
    #     "path": "neural-chat",
    #     "help": "Neural Chat model through Ollama, optimized for conversational and semantic understanding tasks."
    # },
}

# Model information (name, Hugging Face path, and help text)
# Updated with 2025 MTEB leaderboard top performers for Chinese character analysis
MODEL_INFO = {
    # === TOP 2025 MTEB PERFORMERS (NEW ADDITIONS) ===
    "GTE-Multilingual-Base": {
        "path": "Alibaba-NLP/gte-multilingual-base",
        "help": "Leading multilingual embedding model on MTEB 2025. 305M params, 70+ languages, 8192 tokens, 10x faster inference. Optimal for cross-lingual Chinese character analysis."
    },
    "BGE-Multilingual-Gemma2": {
        "path": "BAAI/bge-multilingual-gemma2",
        "help": "SOTA on C-MTEB Chinese benchmark. Based on Gemma-2-9B, specialized for Chinese-Japanese-Korean-English. Excellent cross-lingual capabilities."
    },
    "Jina-Embeddings-v2-ZH": {
        "path": "jinaai/jina-embeddings-v2-base-zh",
        "help": "Chinese-English bilingual specialist. 570M params, 8192 tokens, JinaBERT architecture. No Chinese-English bias, perfect for mixed input."
    },
    "BGE-Base-ZH-v1.5": {
        "path": "BAAI/bge-base-zh-v1.5",
        "help": "Chinese-optimized BGE model with high C-MTEB performance. Specialized for character-level Chinese semantics and traditional character analysis."
    },
    
    # === EXISTING PROVEN MODELS (KEPT FROM ORIGINAL) ===
    "XLM-R": {
        "path": "xlm-roberta-base",
        "help": "A robust multilingual model trained on 100+ languages. Great for cross-lingual tasks like text classification and NER."
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
    f"Sentence-BERT Multilingual": {
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "help": "Proven baseline from cross-lingual research. MPNet architecture, 50+ languages. Excellent for semantic similarity tasks."
    },
    
    # === ADDITIONAL CHINESE-FOCUSED MODELS ===
    "Universal-Sentence-Encoder-Multilingual": {
        "path": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "help": "Google's multilingual sentence encoder. Strong performance across languages with efficient inference."
    },
    
    # === EXPERIMENTAL/RESEARCH MODELS ===
    # "ModernBERT": {
    #     "path": "answerdotai/ModernBERT-base",
    #     "help": "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference."
    # },
    # "Erlangshen": {
    #     "path": "IDEA-CCNL/Erlangshen-Roberta-110M", 
    #     "help": "A Chinese-focused multilingual model optimized for Chinese-English tasks like translation and sentiment analysis."
    # }
}


# Update MODEL_INFO dictionary
MODEL_INFO.update(OLLAMA_MODELS)


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
    f"PHATE": {
        "help": "Potential of Heat-diffusion for Affinity-based Transition Embedding. Great for visualizing complex, high-dimensional data."
    },
    # "Autoencoders": {
    #     "help": "Neural network-based approach for learning compressed representations of data."
    # },
    # "LDA": {
    #     "help": "Linear Discriminant Analysis. A supervised method that maximizes class separability."
    # }
}


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
