# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based multilingual embedding explorer that visualizes semantic relationships between words across languages (primarily Chinese-English). The application uses various ML models to generate embeddings and dimensionality reduction techniques to create interactive visualizations.

## Quick Start Commands

### Running the Application
```bash
# Main application entry point
cd src && streamlit run Welcome.py

# Alternative (legacy)
cd src && ./000_run_app.sh
```

### Environment Setup
```bash
# Initialize project structure and dependencies
./init_project.sh

# Create conda environment
conda create -n zinets python=3.11
conda activate zinets
pip install -r requirements.txt

# Download LASER models for multilingual embeddings
python -m laserembeddings download-models
```

### Optional Ollama Setup
```bash
# Install Ollama models for local embedding generation
ollama pull snowflake-arctic-embed2
ollama pull snowflake-arctic-embed
ollama pull bge-m3
ollama pull paraphrase-multilingual
```

## Architecture Overview

### Application Structure
- **Multi-page Streamlit app** with Welcome.py as the main entry point
- **Component-based architecture** with reusable UI and processing components
- **Modular model management** supporting both Hugging Face and Ollama models
- **Service layer** for external integrations (Google Translate, TTS)

### Core Components

#### `/src/` - Main application code
- `Welcome.py` - Main entry point and navigation
- `config.py` - Configuration, model definitions, and constants
- `pages/1_🔤_Semantics_Explorer.py` - Main semantic visualization interface

#### `/src/models/` - Model management
- `model_manager.py` - Abstract model interface and factory pattern
  - `EmbeddingModel` abstract base class
  - `OllamaModel` for local Ollama models
  - `HuggingFaceModel` for cloud-based models

#### `/src/components/` - Reusable UI components
- `embedding_viz.py` - Embedding visualization logic
- `clustering.py` - Clustering algorithms and visualization
- `dimension_reduction.py` - Various dimensionality reduction methods
- `plotting.py` - Plotly-based interactive charts

#### `/src/services/` - External service integrations
- `google_translate.py` - Translation API integration
- `tts_service.py` - Text-to-speech functionality

#### `/src/utils/` - Utility functions
- `error_handling.py` - Error handling decorators and custom exceptions

### Data Flow
1. **User Input** → Text entered in Chinese/English text areas
2. **Model Selection** → Choose embedding model (Hugging Face/Ollama)
3. **Embedding Generation** → Text converted to high-dimensional vectors
4. **Dimensionality Reduction** → Vectors projected to 2D/3D space
5. **Visualization** → Interactive Plotly charts with clustering options

### Key Features
- **Multiple embedding models**: Sentence-BERT, LASER, XLM-R, mT5, Ollama models
- **Dimensionality reduction**: PHATE, t-SNE, UMAP, Isomap, PCA, and more
- **Interactive visualization**: 2D/3D plotting with clustering
- **Session caching**: Improved performance for repeated operations
- **Multilingual support**: Optimized for Chinese-English language pairs

### Model Performance Notes
- **Snowflake-Arctic-Embed2** (Ollama): Best for Chinese-English semantic alignment
- **LASER**: Good for multilingual sentence embeddings
- **Sentence-BERT Multilingual**: Fast and reliable general-purpose model

### Development Context
This application is part of a larger "digital-duck" ecosystem with related translation and language learning tools. The codebase includes multiple development iterations in `/devs/` subdirectories (claude/, claude4/, deepseek/, qwen/) representing different AI assistant implementations.

### Environment Variables
- `HF_API_KEY` - Hugging Face API key (set in .env or environment)
- Ollama server expected at `http://localhost:11434`

### Data Files
- `src/data/data-1-chn.txt` - Chinese sample text
- `src/data/data-1-enu.txt` - English sample text
- Fallback to hardcoded samples in config.py if files don't exist