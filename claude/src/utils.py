import re
import os
from sentence_transformers import SentenceTransformer

# Ensure the charts folder exists
os.makedirs("charts", exist_ok=True)

# Load a pre-trained multilingual embedding model
@st.cache_resource
def load_model(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)

# Function to intelligently split input
def split_input(text):
    return re.split(r'\W+', text)

# Generate embeddings for a list of Chinese characters, words, or phrases
def generate_embeddings(model, text_list):
    return model.encode(text_list)