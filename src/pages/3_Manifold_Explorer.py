import streamlit as st
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, Isomap, MDS
from umap import UMAP
import plotly.express as px
from db import init_db, save_result
from utils import load_model, split_input, generate_embeddings

def main():
    st.title("Manifold Explorer")
    st.write("Compare different manifold learning techniques for visualizing semantic closeness of Chinese characters, words, or phrases.")

    # User input for Chinese characters, words, or phrases
    user_input = st.text_input("Enter Chinese characters, words, or phrases (separated by spaces, commas, or any special characters):", 
                               "猫 狗 车 树 书 水 火 山 人 天")

    # Intelligently split the input
    chinese_list = [item for item in split_input(user_input) if item.strip()]

    if not chinese_list:
        st.error("Please enter at least one Chinese character, word, or phrase!")
        return

    # Allow user to select multiple manifold learning methods
    st.sidebar.header("Select Manifold Learning Methods")
    selected_methods = st.sidebar.multiselect(
        "Choose 2 or more methods to compare:",
        ["t-SNE", "UMAP", "Isomap", "MDS"],
        default=["t-SNE", "UMAP"]
    )

    if len(selected_methods) < 2:
        st.error("Please select at least 2 methods to compare!")
        return

    # Load the multilingual embedding model
    model = load_model()

    # Generate embeddings for the Chinese characters, words, or phrases
    embeddings = generate_embeddings(model, chinese_list)

    # Initialize database connection
    conn = init_db()

    # Apply selected manifold learning methods
    for method in selected_methods:
        st.header(f"Method: {method}")
        
        if method == "t-SNE":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(3, len(chinese_list) - 1))
        elif method == "UMAP":
            reducer = UMAP(n_components=2, random_state=42)
        elif method == "Isomap":
            reducer = Isomap(n_components=2, n_neighbors=min(5, len(chinese_list) - 1))
        elif method == "MDS":
            reducer = MDS(n_components=2, random_state=42)
        
        # Reduce dimensionality
        reduced_embeddings = reducer.fit_transform(embeddings)

        # Create a DataFrame for visualization
        df = pd.DataFrame({
            'Chinese': chinese_list,
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1]
        })

        # Plot using Plotly for hover functionality
        fig = px.scatter(df, x='x', y='y', text='Chinese', hover_name='Chinese',
                         title=f"Semantic Closeness (Method: {method})",
                         labels={'x': 'Component 1', 'y': 'Component 2'})
        fig.update_traces(textposition='top center')
        fig.update_layout(showlegend=False)

        # Display the plot
        st.plotly_chart(fig)

        # Save the plot to the charts folder
        chart_path = f"charts/{method}_{user_input[:10]}.html"
        fig.write_html(chart_path)

        # Save results to the database
        save_result(conn, "paraphrase-multilingual-MiniLM-L12-v2", method, user_input, chart_path)

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()