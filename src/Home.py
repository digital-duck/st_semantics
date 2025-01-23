import streamlit as st
from config import ST_APP_NAME, ST_ICON

# Set page config
st.set_page_config(
    page_title=ST_APP_NAME,
    page_icon=ST_ICON,
    layout="wide"
)

# Title and introduction
st.title(f"{ST_ICON} {ST_APP_NAME}")

st.markdown("""
## Welcome to the Multilingual Language Learning Suite

This application helps you explore and understand languages through:

- 🔤 **Semantic Analysis**: Visualize word relationships across languages
- 🌐 **Translation**: Translate text with audio support
- 🤖 **Language Assistant**: Interactive learning with AI
- 📝 **Notes**: Track your learning progress

### Getting Started

Select a feature from the sidebar to begin exploring!

### Recent Updates
- Added Snowflake Arctic models for improved multilingual embeddings
- Enhanced visualization options with clustering
- Improved performance with session caching
""")

# Optional: Add quick access buttons or recent activity
col1, col2 = st.columns(2)
with col1:
    st.info("📊 **Quick Start**\n\nTry the semantics explorer to visualize word relationships across languages!")
with col2:
    st.success("💡 **Tip**\nUse the clustering feature to discover word groups automatically!")