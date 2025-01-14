import streamlit as st
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from db import init_db, save_result
from utils import load_model, split_input, generate_embeddings

# List of top 5 multilingual embedding models
MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "Lightweight and fast multilingual model.",
    "paraphrase-multilingual-mpnet-base-v2": "High-quality multilingual model with better performance.",
    "distiluse-base-multilingual-cased-v1": "Distilled version of multilingual USE, smaller and faster.",
    "xlm-r-base-en-ko-nli-stsb": "XLM-Roberta base model fine-tuned for multilingual tasks.",
    "LaBSE": "Language-agnostic BERT Sentence Embedding, supports 109 languages."
}

def main():
    st.title("Semantics Explorer")
    st.write("Enter a list of Chinese characters, words, or phrases to visualize their semantic closeness in 2D space using multiple multilingual embedding models.")

    # User input for Chinese characters, words, or phrases
    user_input = st.text_input("Enter Chinese characters, words, or phrases (separated by spaces, commas, or any special characters):", 
                               "猫 狗 车 树 书 水 火 山 人 天")

    # Intelligently split the input
    chinese_list = [item for item in split_input(user_input) if item.strip()]

    if not chinese_list:
        st.error("Please enter at least one Chinese character, word, or phrase!")
        return

    # Allow user to select multiple models
    st.sidebar.header("Select Embedding Models")
    selected_models = st.sidebar.multiselect(
        "Choose 2 or more models to compare:",
        list(MODELS.keys()),
        default=["paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-multilingual-mpnet-base-v2"]
    )

    if len(selected_models) < 2:
        st.error("Please select at least 2 models to compare!")
        return

    # Initialize database connection
    conn = init_db()

    # Generate embeddings and plot for each selected model
    for model_name in selected_models:
        st.header(f"Model: {model_name}")
        st.write(MODELS[model_name])

        # Load the model
        model = load_model(model_name)

        # Generate embeddings for the Chinese characters, words, or phrases
        embeddings = generate_embeddings(model, chinese_list)

        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(3, len(chinese_list) - 1))
        reduced_embeddings = tsne.fit_transform(embeddings)

        # Create a DataFrame for visualization
        df = pd.DataFrame({
            'Chinese': chinese_list,
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1]
        })

        # Plot using Plotly for hover functionality
        fig = px.scatter(df, x='x', y='y', text='Chinese', hover_name='Chinese',
                         title=f"Semantic Closeness (Model: {model_name})",
                         labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'})
        fig.update_traces(textposition='top center')
        fig.update_layout(showlegend=False)

        # Display the plot
        st.plotly_chart(fig)

        # Save the plot to the charts folder
        chart_path = f"charts/{model_name}_tsne_{user_input[:10]}.html"
        fig.write_html(chart_path)

        # Save results to the database
        save_result(conn, model_name, "t-SNE", user_input, chart_path)

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()