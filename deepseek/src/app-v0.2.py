import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from laserembeddings import Laser
from sklearn.manifold import TSNE, Isomap
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

st.set_page_config(layout="wide")

# Load environment variables from .env file
load_dotenv()

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

def do_sidebar():
    # Embedding model selection
    with st.sidebar:
        st.subheader("Settings")
        st.radio("Choose Embedding Model", ["XLM-R", "LASER"], key="cfg_model_choice")
        st.radio("Choose Dimensionality Reduction Method", ["t-SNE", "Isomap"], key="cfg_reduction_method")

# Main app function
def main():
    st.header("Multilingual Embedding Visualization")
    st.write("Visualize Chinese characters/phrases and (optionally) their English translations in 2D space.")

    # Input dataset
    st.subheader("Input Data")
    c1, _, c2 = st.columns([3,1,3])
    with c1:
        chinese_words = st.text_area("Enter Chinese words/phrases (one per line)", "你好\n爱\n天气\n书\n猫")
    with c2:
        english_words = st.text_area("Enter corresponding English translations (one per line)", "Hello\nLove\nWeather\nBook\nCat")

    chinese_words = chinese_words.strip().split("\n")
    english_words = english_words.strip().split("\n")

    # if len(chinese_words) != len(english_words):
    #     st.error("The number of Chinese words and English translations must match.")
    #     st.stop()


    # Add a button to trigger processing
    if st.button("Generate Visualization"):

        if (len(chinese_words) + len(english_words)) < 1:
            st.warning("Please enter at least one word or phrase.")
            st.stop()        
            # st.rerun()

        # Extract embeddings
        def get_xlmr_embeddings(words):
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            model = AutoModel.from_pretrained("xlm-roberta-base")
            embeddings = []
            for word in words:
                inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
            return np.vstack(embeddings)

        def get_laser_embeddings(words, lang="zh"):
            laser = Laser()
            return laser.embed_sentences(words, lang=lang)

        model_choice = st.session_state.get("cfg_model_choice", "XLM-R")
        if model_choice == "XLM-R":
            chinese_embeddings = get_xlmr_embeddings(chinese_words)
            english_embeddings = get_xlmr_embeddings(english_words)
        else:
            chinese_embeddings = get_laser_embeddings(chinese_words, lang="zh")
            english_embeddings = get_laser_embeddings(english_words, lang="en")

        embeddings = np.vstack([chinese_embeddings, english_embeddings])

        # Dimensionality reduction
        def reduce_dimensions(embeddings, method="tsne"):
            if method == "t-SNE":
                reducer = TSNE(n_components=2, random_state=42)
            elif method == "Isomap":
                reducer = Isomap(n_components=2)
            else:
                raise ValueError("Unsupported method.")
            return reducer.fit_transform(embeddings)

        reduction_method = st.session_state.get("cfg_reduction_method", "t-SNE")
        reduced_embeddings = reduce_dimensions(embeddings, method=reduction_method)

        # Plotting
        st.header("2D Scatter Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = chinese_words + english_words
        colors = ["blue"] * len(chinese_words) + ["red"] * len(english_words)
        for i, (x, y) in enumerate(reduced_embeddings):
            ax.scatter(x, y, color=colors[i], label=labels[i])
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_title(f"{model_choice} Embeddings ({reduction_method})")
        st.pyplot(fig)

        # Save plot
        if st.button("Save Plot"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
            file_png = f"images/viz_embedding_{ts}.png"
            fig.savefig(file_png)
            st.success(f"Plot saved as '{file_png}'.")

# Run the app
if __name__ == "__main__":
    check_login()  # Check login status
    do_sidebar()
    main()