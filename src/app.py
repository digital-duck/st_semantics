import streamlit as st

st.set_page_config(
    page_title="Semantics Explorer",
    page_icon="🔍",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Semantics Explorer", "Manifold Explorer", "Review"])

if page == "Welcome":
    st.switch_page("pages/1_Welcome.py")
elif page == "Semantics Explorer":
    st.switch_page("pages/2_Semantics_Explorer.py")
elif page == "Manifold Explorer":
    st.switch_page("pages/3_Manifold_Explorer.py")
elif page == "Review":
    st.switch_page("pages/4_Review.py")