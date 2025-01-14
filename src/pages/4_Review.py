import streamlit as st
import pandas as pd
from st_aggrid import (
    AgGrid, GridOptionsBuilder, GridUpdateMode
    , JsCode, DataReturnMode
)
from db import fetch_results

def main():
    st.title("Review Results")
    st.write("Review saved results from the Semantics Explorer and Manifold Explorer.")

    # Fetch data from the database
    df = fetch_results()

    # Allow filtering by model and method
    st.sidebar.header("Filter Results")
    selected_models = st.sidebar.multiselect("Filter by Model", df["model_name"].unique())
    selected_methods = st.sidebar.multiselect("Filter by Method", df["method"].unique())

    if selected_models:
        df = df[df["model_name"].isin(selected_models)]
    if selected_methods:
        df = df[df["method"].isin(selected_methods)]

    # Display the data in an interactive table
    st.write("### Saved Results")
    grid_response = AgGrid(
        df,
        editable=False,
        height=300,
        width="100%",
        reload_data=False,
    )

    # Show selected row details
    selected_rows = grid_response["selected_rows"]
    if selected_rows:
        selected_row = selected_rows[0]
        st.write("### Selected Row Details")
        st.write(f"**Model Name**: {selected_row['model_name']}")
        st.write(f"**Method**: {selected_row['method']}")
        st.write(f"**Input Text**: {selected_row['input_text']}")
        st.write(f"**Chart Path**: {selected_row['chart_path']}")

        # Display the associated plot
        st.write("### Associated Plot")
        with open(selected_row["chart_path"], "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=500)

if __name__ == "__main__":
    main()