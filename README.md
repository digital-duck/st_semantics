# Multilingual Embedding Explorer

A Streamlit application for visualizing and exploring multilingual word embeddings in 2D and 3D spaces.

## Significance and Applications

This tool serves multiple important purposes across different domains:

### Research Applications
- **Cross-lingual Studies**: Enables researchers to visualize and analyze semantic relationships between different languages
- **Linguistic Research**: Helps understand how different embedding models capture semantic meaning across languages
- **Model Evaluation**: Provides visual insights into the performance of different multilingual embedding models
- **Semantic Analysis**: Facilitates the study of semantic spaces and word relationships in multiple languages

### Educational Uses
- **Language Learning**: Helps students visualize relationships between words in different languages, making abstract concepts more concrete
- **Linguistics Education**: Demonstrates concepts like semantic similarity, word vectors, and cross-lingual relationships
- **Data Science Teaching**: Serves as a practical example of dimensionality reduction and visualization techniques
- **AI/ML Education**: Illustrates how neural networks understand and represent language

### Practical Applications
- **Translation Work**: Assists translators in understanding semantic equivalences between languages
- **Content Analysis**: Helps content creators ensure consistent meaning across multiple languages
- **Cultural Studies**: Reveals how different languages encode similar concepts
- **Educational Material Development**: Supports creation of multilingual educational resources


## Integration Possibilities

This semantic exploration tool is designed to be part of a larger ecosystem of language learning and analysis tools:

### Current Integrations
- **Translation Services**: Works alongside st_translator app which provides:
  - Google Translate integration
  - Text-to-speech capabilities
  - Real-time translation visualization

### Planned Integrations
- **Interactive Learning**:
  - Chatbot assistance for language learning
  - Note-taking system for vocabulary and semantic relationships
  - Personalized learning paths based on semantic relationships

### Potential Extensions
- **Advanced Analysis**:
  - Semantic relationship tracking over time
  - Custom vocabulary list building with semantic grouping
  - Cross-language idiom and expression matching
- **Educational Tools**:
  - Lesson planning based on semantic relationships
  - Student progress tracking through semantic space
  - Interactive exercises using semantic relationships

### Integration Benefits
- **Comprehensive Learning Environment**: Combines semantic understanding with practical translation
- **Enhanced User Experience**: Seamless transition between different language learning tools
- **Deeper Understanding**: Links theoretical semantic relationships with practical language use
- **Personalized Learning**: Adapts to user's learning style and progress

## Features

- Support for multiple embedding models:
  - Hugging Face models (BERT, XLM-R, etc.)
  - LASER embeddings
  - Local Ollama models (Snowflake-Arctic-Embed2)
- Multiple dimensionality reduction methods:
  - PHATE
  - t-SNE
  - UMAP
  - Isomap
  - And more
- Interactive visualizations:
  - 2D and 3D plotting
  - Clustering visualization
  - Color-coded language differentiation
- Support for Chinese-English word pairs
- Progress tracking for embedding generation
- Cached sessions for improved performance
- Error handling and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multilingual-embedding-explorer.git
cd multilingual-embedding-explorer
```

2. Initialize project structure (this will create all necessary directories and files):
```bash
./init_project.sh
```

3. Install dependencies:
```bash
conda create -n zinets python=3.11
conda activate zinets
pip install -r requirements.txt
```

4. Download required LASER models and Chinese support:
```bash
python -m laserembeddings download-models
```

5. Install Ollama (optional, for local embedding models):
Visit [Ollama's website](https://ollama.ai/) and follow installation instructions.

6. Pull required Ollama models (if using Ollama):
```bash
ollama pull snowflake-arctic-embed2
ollama pull snowflake-arctic-embed
```

## Project Structure
```
project/
├── config.py              # Configuration settings
├── utils/                 # Utility functions
│   ├── __init__.py
│   └── error_handling.py
├── models/               # Model management
│   ├── __init__.py
│   └── model_manager.py
├── app.py               # Main Streamlit application
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Enter Chinese and English words/phrases in the respective text areas.

3. Choose your preferred:
   - Embedding model
   - Dimensionality reduction method
   - Visualization type (2D/3D)
   - Clustering options

4. Click "Visualize" to generate the embedding visualization.

## Model Performance Notes

- **Snowflake-Arctic-Embed2**: Best performance for Chinese-English pairs, showing strong semantic alignment
- **LASER**: Good performance for multilingual embeddings, especially with longer phrases
- **Sentence-BERT Multilingual**: Fast and reliable for general use
- **Other Ollama models** (Neural-Chat, Mistral): Not recommended for multilingual tasks

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

[Your chosen license]