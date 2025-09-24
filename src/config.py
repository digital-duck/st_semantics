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
        "help": "BGE-M3 is a new model from BAAI distinguished for its versatility in Multi-Functionality, Multi-Linguality, and Multi-Granularity.",
        "is_active": True
    },
    # "Paraphrase-Multilingual (Ollama)": {
    #     "path": "paraphrase-multilingual",
    #     "help": "Sentence-transformers model (multilingual) that can be used for tasks like clustering or semantic search."
    # },
    "Snowflake-Arctic-Embed2 (Ollama)": {
        "path": "snowflake-arctic-embed2",
        "help": "Snowflake Arctic model through Ollama offering efficient embedding generation with strong multilingual capabilities, especially for Chinese-English pairs.",
        "is_active": True
    },
    # "Snowflake-Arctic-Embed (Ollama)": {
    #     "path": "snowflake-arctic-embed",
    #     "help": "Original Snowflake Arctic model through Ollama for multilingual embeddings."
    # },
    "EmbeddingGemma (Ollama)": {
        "path": "embeddinggemma",
        "help": "Google's EmbeddingGemma - 300M parameter state-of-the-art embedding model built from Gemma 3 with T5Gemma initialization. Trained on 100+ languages using Gemini model technology. Optimized for search, retrieval, classification, clustering, and semantic similarity tasks.",
        "is_active": True
    },
    "Nomic-Embed-Text (Ollama)": {
        "path": "nomic-embed-text",
        "help": "Nomic's v1.5 embedding model through Ollama. First truly open embedding model with multimodal capabilities and strong accuracy (86.2%). Optimized for semantic text embeddings.",
        "is_active": True
    },
    "Qwen3-Embedding-0.6B (Ollama)": {
        "path": "qwen3-embedding:0.6b",
        "help": "Qwen3 0.6B embedding model through Ollama - shows excellent metrics in EmbeddingGemma benchmarks. Part of MTEB #1 multilingual series. Updated 27 minutes ago! Perfect for geosemantic comparison with EmbeddingGemma.",
        "is_active": True
    },
    "Qwen3-Embedding-4B (Ollama)": {
        "path": "qwen3-embedding:4b",
        "help": "Qwen3 4B embedding model through Ollama - larger variant with enhanced capabilities. Part of MTEB #1 multilingual series (8B variant scores 70.58). Excellent for studying parameter scaling effects on semantic geometry.",
        "is_active": True
    },
    # "Mistral (Ollama)": {
    #     "path": "mistral",
    #     "help": "Mistral model through Ollama offering efficient embedding generation with good multilingual capabilities.",
    #     "is_active": False
    # },
    # "Neural-Chat (Ollama)": {
    #     "path": "neural-chat",
    #     "help": "Neural Chat model through Ollama, optimized for conversational and semantic understanding tasks.",
    #     "is_active": False
    # },
}

# Model information (name, Hugging Face path, and help text)
# Updated with 2025 MTEB leaderboard top performers for geosemantic analysis
MODEL_INFO = {
    # === 2025 MTEB LEADERS: TIER 1 PERFORMANCE ===
    "Stella-400M": {
        "path": "dunzhang/stella_en_400M_v5",
        "help": "Current MTEB retrieval leaderboard leader for commercial use. 400M parameters, state-of-the-art architecture optimization by Dun Zhang. Top choice for geosemantic geometric analysis.",
        "is_active": False,
        "note": "too big and slow"
    },
    "Stella-1.5B": {
        "path": "dunzhang/stella_en_1.5B_v5",
        "help": "Larger variant of Stella model (1.5B parameters). Minimal accuracy gain over 400M version but useful for studying parameter scaling effects on semantic geometry.",
        "is_active": False,
        "error": """Error in get_embeddings: The repository dunzhang/stella_en_400M_v5 contains custom code which must be executed to correctly load the model. You can inspect the repository content at https://hf.co/dunzhang/stella_en_400M_v5 . You can inspect the repository content at https://hf.co/dunzhang/stella_en_400M_v5. Please pass the argument trust_remote_code=True to allow custom code to be run.
        """,
    },
    "Jina-Embeddings-v3": {
        "path": "jinaai/jina-embeddings-v3",
        "help": "Best multilingual model (89 languages), 2nd on MTEB English leaderboard. 570M params, XLM-RoBERTa + task-specific LoRA adapters. Perfect for cross-lingual geosemantic studies.",
        "is_active": False,
        "error": """Error in get_embeddings: jinaai/xlm-roberta-flash-implementation You can inspect the repository content at https://hf.co/jinaai/jina-embeddings-v3. Please pass the argument trust_remote_code=True to allow custom code to be run.
        """
    },
    "BGE-M3": {
        "path": "BAAI/bge-m3",
        "help": "Highest retrieval accuracy (72%) in comparative studies. Multi-functionality, multi-linguality, multi-granularity. Excellent baseline for geometric structure analysis.",
        "is_active": False,
        "error": """Error in get_embeddings: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors. See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
"""
    },

    # === 2025 MTEB LEADERS: TIER 2 ARCHITECTURAL INNOVATION ===
    "Nomic-Embed-Text-v2": {
        "path": "nomic-ai/nomic-embed-text-v2",
        "help": "First MoE (Mixture-of-Experts) architecture for embeddings. Strong accuracy (86.2%) with novel approach. Critical for studying MoE vs standard transformer geometry.",
        "is_active": False,
        "error": """Error in get_embeddings: nomic-ai/nomic-embed-text-v2 is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models' If this is a private repository, make sure to pass a token having permission to this repo either by logging in with hf auth login or by passing token=<your_token>
        """,
    },
    "GTE-Multilingual-Base": {
        "path": "Alibaba-NLP/gte-multilingual-base",
        "help": "MTEB 2025 leader. 305M params, 70+ languages, 10x faster inference. Encoder-only transformer optimized for efficiency while preserving semantic structure.",
        "is_active": False,
        "error": """Error in get_embeddings: Alibaba-NLP/new-impl You can inspect the repository content at https://hf.co/Alibaba-NLP/gte-multilingual-base. Please pass the argument trust_remote_code=True to allow custom code to be run.
        """,
    },
    "E5-Base-v2": {
        "path": "intfloat/e5-base-v2",
        "help": "Balanced accuracy-speed trade-off (83-85% accuracy, 79-82ms latency). Strong performer without prefix prompts, ideal for studying geometric consistency.",
        "is_active": True,
        "note": "very good"
    },
    "Qwen3-Embedding-0.6B": {
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "help": "Qwen3 embedding model showing excellent metrics in EmbeddingGemma benchmark comparisons. 600M parameters, part of MTEB #1 multilingual series (8B variant scores 70.58). Supports 100+ languages with strong cross-lingual capabilities.",
        "is_active": True
    },
    "Qwen3-Embedding-4B": {
        "path": "Qwen/Qwen3-Embedding-4B",
        "help": "Qwen3 4B embedding model - larger variant with enhanced performance. Part of the MTEB #1 multilingual series with state-of-the-art cross-lingual capabilities. Ideal for studying parameter scaling effects on semantic geometry preservation.",
        "is_active": True
    },

    # === LEGACY TOP PERFORMERS (ESTABLISHED BASELINES) ===
    "BGE-Multilingual-Gemma2": {
        "path": "BAAI/bge-multilingual-gemma2",
        "help": "SOTA on C-MTEB Chinese benchmark. Based on Gemma-2-9B, specialized for Chinese-Japanese-Korean-English. Excellent cross-lingual capabilities.",
        "is_active": False,
        "error": """too big""",
    },
    "Jina-Embeddings-v2-ZH": {
        "path": "jinaai/jina-embeddings-v2-base-zh",
        "help": "Chinese-English bilingual specialist. 570M params, 8192 tokens, JinaBERT architecture. No Chinese-English bias, perfect for mixed input.",
        "is_active": False,
        "error": """Error in get_embeddings: You are using sdpa as attention type. However, non-absolute positional embeddings can not work with them. Please load the model with attn_implementation="eager".
        """,
    },
    "BGE-Base-ZH-v1.5": {
        "path": "BAAI/bge-base-zh-v1.5",
        "help": "Chinese-optimized BGE model with high C-MTEB performance. Specialized for character-level Chinese semantics and traditional character analysis.",
        "is_active": False,
        "error": """Error in get_embeddings: 
Due to a serious vulnerability issue in torch.load, even with weights_only=True, 
we now require users to upgrade torch to at least v2.6 in order to use the function. 
This version restriction does not apply when loading files with safetensors. 
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
"""
    },
    
    # === FOUNDATIONAL MODELS (ESTABLISHED RESEARCH BASELINES) ===
    "XLM-R": {
        "path": "xlm-roberta-base",
        "help": "A robust multilingual model trained on 100+ languages. Great for cross-lingual tasks like text classification and NER.",
        "is_active": True
    },
    "mBERT": {
        "path": "bert-base-multilingual-cased",
        "help": "Multilingual BERT trained on 104 languages. Widely used for cross-lingual transfer learning.",
        "is_active": True
    },
    "LaBSE": {
        "path": "sentence-transformers/LaBSE",
        "help": "Language-agnostic BERT sentence embeddings for 109 languages. Excellent for sentence similarity and paraphrase detection.",
        "is_active": True
    },
    "DistilBERT Multilingual": {
        "path": "distilbert-base-multilingual-cased",
        "help": "A lightweight version of mBERT. Faster and more efficient, suitable for real-time applications.",
        "is_active": True
    },
    "XLM": {
        "path": "xlm-mlm-100-1280",
        "help": "Cross-lingual language model trained using masked and translation language modeling. Good for translation tasks.",
        "is_active": False,
        "error": """Error in get_embeddings: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors. See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
        """
    },
    "InfoXLM": {
        "path": "microsoft/infoxlm-base",
        "help": "An extension of XLM-R with improved cross-lingual transferability. Great for low-resource languages.",
        "is_active": False,
        "error": """Error in get_embeddings: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors. See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
        """,
    },
    "Sentence-BERT Multilingual": {
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "help": "Proven baseline from cross-lingual research. MPNet architecture, 50+ languages. Excellent for semantic similarity tasks.",
        "is_active": True
    },
    # === OPENAI EMBEDDING MODELS (REFERENCE ONLY - REQUIRES API KEY) ===
    # Note: These models require OpenAI API access and are not directly usable in this app
    # Added for completeness and reference for other researchers
    "OpenAI text-embedding-ada-002": {
        "path": "text-embedding-ada-002",  # API model path
        "help": "OpenAI's 2nd generation embedding model (Dec 2022). 1536 dimensions. Industry standard that replaced 5 separate models with 99.8% cost reduction. Requires OpenAI API key.",
        "is_active": False
    },
    "OpenAI text-embedding-3-small": {
        "path": "text-embedding-3-small",  # API model path
        "help": "OpenAI's 3rd generation small model (Jan 2024). 1536 dimensions, 5x cheaper than ada-002. 44.0% vs 31.4% on MIRACL benchmark. Requires OpenAI API key.",
        "is_active": False
    },
    "OpenAI text-embedding-3-large": {
        "path": "text-embedding-3-large",  # API model path
        "help": "OpenAI's best performing 3rd generation model (Jan 2024). 3072 dimensions. SOTA performance: 54.9% on MIRACL, 64.6% on MTEB. Requires OpenAI API key.",
        "is_active": False
    },

    # === ADDITIONAL CHINESE-FOCUSED MODELS ===
    "Universal-Sentence-Encoder-Multilingual": {
        "path": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "help": "Google's multilingual sentence encoder. Strong performance across languages with efficient inference.",
        "is_active": False,
        "note": "not good, highly degenerate"
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
