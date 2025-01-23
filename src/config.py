# config.py
from pathlib import Path

# App Constants
ST_APP_NAME = "Multilingual Embedding Explorer"
ST_HEADER_1 = "View word embeddings in 2-3D spaces"
ST_ICON = "🚩"

# Default Settings
DEFAULT_N_CLUSTERS = 5
DEFAULT_MAX_WORDS = 15 
DEFAULT_MODEL = "Sentence-BERT Multilingual"
DEFAULT_METHOD = "PHATE"
DEFAULT_DIMENSIONS = "2D"

# Plot Settings
PLOT_CONFIG = {
    "width": 800,
    "height": 800,
    "color_map": {
        "chinese": "red",
        "english": "blue"
    }
}

# Sample Data
SAMPLE_DATA = {
    "chinese": """你好 男子汉 \n爱 "女子" \n天气\n书\n猫""",
    "english": """Hello\nLove\nWeather\nBook "Woman" "Gentle Man" \nCat"""
}

# File Paths
PATHS = {
    "images_dir": Path("images"),
    "chinese_file": Path("data/chn.txt"),
    "english_file": Path("data/enu.txt")
}

# Cache Settings
CACHE_CONFIG = {
    "ttl": 3600,  # Time to live for cached data (in seconds)
    "max_entries": 100
}