# Multilingual Embedding Explorer 🧠✨

**Geometry of Meaning: Visualizing Semantic Structure Across Languages**

A powerful Streamlit application for exploring multilingual word embeddings through interactive 2D and 3D visualizations. Discover how meaning has geometric structure using cutting-edge embedding models and manifold learning techniques.

*This Streamlit app is built with love in close collaboration with **Claude Code** ❤️*

> **Latest Version 2.8** - Now with Streamlit 1.50.x compatibility, enhanced Chinese character processing, robust 3D visualization, and publication-quality export capabilities!

## 🌟 Significance and Applications

This tool serves multiple important purposes across different domains:

### 🔬 **Research Applications**
- **Cross-lingual Studies**: Visualize and analyze semantic relationships between different languages
- **Linguistic Research**: Understand how embedding models capture semantic meaning across languages
- **Model Evaluation**: Visual insights into multilingual embedding model performance
- **Semantic Analysis**: Study semantic spaces and word relationships in multiple languages
- **"Geometry of Meaning"**: Discover linear patterns in numbers, branching in colors, clustering in concepts

### 🎓 **Educational Applications**
- **Language Learning**: Visualize relationships between words in different languages
- **Linguistics Education**: Demonstrate semantic similarity, word vectors, and cross-lingual relationships
- **Data Science Teaching**: Practical examples of dimensionality reduction and visualization
- **AI/ML Education**: Show how neural networks understand and represent language
- **Interactive Discovery**: Students can explore semantic patterns independently

### 🌍 **Practical Applications**
- **Translation Work**: Understand semantic equivalences between languages
- **Content Analysis**: Ensure consistent meaning across multiple languages
- **Cultural Studies**: Reveal how different languages encode similar concepts
- **Educational Material Development**: Create multilingual educational resources

## 🚀 Key Features

### 🔤 **Semantics Explorer**
- **Multiple Embedding Models**: 15+ models including Sentence-BERT Multilingual, E5-Base-v2, BGE-M3, Ollama models
- **Advanced Dimensionality Reduction**: PHATE, t-SNE, UMAP, Isomap, PCA, MDS, LLE, Kernel PCA, Spectral Embedding
- **Interactive 2D/3D Visualizations**: Real-time plotting with clustering analysis, zoom, rotate, and pan controls
- **Robust Chinese Character Support**: Enhanced tokenization with NaN error detection and handling
- **Multilingual Support**: Chinese (source) + selectable target languages (English, French, Spanish, German)
- **ISO Language Codes**: Standardized 3-letter codes (chn, enu, fra, spa, deu) for consistent file naming
- **Smart File Management**: Load/save text datasets with sanitized filenames and session persistence
- **Publication-Quality Export**: High-DPI PNG, SVG, PDF export with customizable styling
- **Advanced Image Management**: Search, filter, and organize generated visualizations

### 🔍 **Semantics Explorer - Dual View** (Enhanced!)
- **Overview + Detail Views**: Simultaneous global and zoomed perspectives
- **Interactive Zoom Controls**: Precise navigation through semantic space
- **Pan Functionality**: Smooth movement through embedding space
- **Enhanced Statistics**: Real-time metrics and word lists in zoom areas
- **Publication-Ready Export**: High-quality image downloads with standardized naming
- **Streamlit 1.50.x Compatible**: Fixed widget key/session state consistency issues

### 🖼️ **Review Images** (Completely Redesigned!)
- **Advanced Search System**: Find images using text patterns or glob expressions
- **Bulk Selection Operations**: Select All/Clear All with checkbox management
- **Flexible Grid Layouts**: 1, 2, or 3 images per row for optimal comparison
- **Smart File Organization**: Sort by modification time (newest first)
- **Batch Download & Delete**: Manage visualization collections efficiently
- **Session State Management**: Remember search patterns and selections

### 🌐 **Translator**
- **Professional Translation**: High-quality translation API integration for 30+ languages
- **Auto-detection**: Smart source language identification
- **Research-focused**: Perfect for creating semantic datasets across languages
- **Save Translations**: Store translation pairs for reference
- **Editable Results**: Refine translations for research accuracy

## 📊 Pre-built Semantic Categories

Explore the "Geometry of Meaning" with included datasets:

- **🎨 Colors** - Perfect branching patterns (warm/cool, light/dark families)
- **🔢 Numbers** - **Linear sequence relationships** (major discovery!)
- **😊 Emotions** - Positive/negative clustering patterns
- **🐾 Animals** - Taxonomic family structures (pets, wild, insects)
- **🍎 Food** - Category-based groupings (grains, fruits, meats)
- **🈵 子-network** - Chinese morpheme network with multilingual translations (fra, spa, deu, enu)
- **🔤 ASCII-words** - Control dataset for baseline comparisons

## 🛠️ Installation

### Quick Setup

```bash
# Clone and navigate
git clone git@github.com:digital-duck/st_semantics.git
cd st_semantics

# Create conda environment
conda create -n zinets python=3.11
conda activate zinets

# Install dependencies
pip install -r requirements.txt

# For PyTorch compatibility
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Optional: Download LASER models
python -m laserembeddings download-models
```

### Optional: Ollama Models

```bash
# Install Ollama (visit https://ollama.ai)
ollama pull snowflake-arctic-embed2
ollama pull bge-m3
ollama pull paraphrase-multilingual
```

### API Keys Setup

Create `.env` file in project root:

```bash
# For Hugging Face models (optional)
HF_API_KEY="<your_hugging_face_key>"

# For DeepL Translator (optional)
DEEPL_AUTH_KEY="<your_deepl_key>"
```

## 🚀 Usage

### Launch Application

```bash
cd src
streamlit run Welcome.py
```

### Basic Workflow

1. **Semantics Explorer**: Select target languages → Enter Chinese + target language words → Choose model & method → Visualize in 2D/3D
2. **Multilingual Analysis**: Compare semantic relationships across up to 5 languages simultaneously
3. **Review Images**: Search, select, and compare multiple visualizations side-by-side
4. **Translator**: Create multilingual datasets using professional translation

### Advanced Features

- **Load Text**: Use pre-built semantic categories or load custom datasets with automatic language detection
- **Save Text**: Create named datasets with automatic sanitization and ISO language codes
- **Language Selection**: Dynamic multi-target language support with color-coded visualization
- **3D Interactive Mode**: Rotate, zoom, and explore semantic spaces with clustering
- **File Naming**: Standardized format: `<dataset>-<lang_code>.txt` (e.g., `子-network-fra.txt`)
- **Cross-compare**: Analyze multiple semantic categories and languages simultaneously

## 🔬 Model Performance & Compatibility (v2.8)

### Recommended Models
- **Sentence-BERT Multilingual** (Default): Fast, reliable, excellent cross-lingual alignment
- **BGE-M3 (Ollama)**: Best performance for Chinese-English pairs
- **Snowflake-Arctic-Embed2 (Ollama)**: Optimized multilingual performance

### Known Issues & Solutions
- **E5-Base-v2**: ⚠️ May produce NaN errors with Chinese text → Enhanced error handling added
- **Streamlit 1.50.x**: ✅ Fixed widget key/session state consistency issues
- **3D Plotting**: ✅ Fixed Plotly griddash property errors for 3D scenes
- **Chinese Characters**: ✅ Enhanced tokenization and NaN detection

### Technical Improvements (v2.8)
- **Robust Error Handling**: NaN detection and replacement in embedding generation
- **Enhanced Tokenization**: Proper attention mask handling for Chinese text
- **3D Visualization Stability**: Removed unsupported properties for 3D scene axes
- **Session State Consistency**: Fixed widget key/value mismatches for Streamlit 1.50.x

## 📁 Project Structure

```
st_semantics/
├── src/
│   ├── Welcome.py                          # Main entry point with enhanced UI
│   ├── pages/
│   │   ├── 1_🔤_Semantics_Explorer.py      # Core visualization (v2.8 compatible)
│   │   ├── 2_🔍_Semantics_Explorer-Dual_View.py  # Advanced dual-view interface
│   │   ├── 3_🖼️_Review_Images.py           # Enhanced image management system
│   │   └── 9_🌐_Translator.py              # Translation services
│   ├── components/
│   │   ├── shared/                         # Shared UI components
│   │   │   └── publication_settings.py    # Reusable publication controls
│   │   ├── embedding_viz.py                # Core visualization logic (v2.8 enhanced)
│   │   ├── dimension_reduction.py
│   │   ├── plotting.py                     # Fixed 3D plotting issues
│   │   ├── clustering.py
│   │   └── geometric_analysis.py
│   ├── models/
│   │   └── model_manager.py               # Enhanced Chinese character handling
│   ├── services/                          # External integrations
│   │   ├── google_translate.py
│   │   └── tts_service.py
│   ├── utils/                             # Utility functions
│   │   ├── error_handling.py              # Enhanced error handling
│   │   ├── download_helpers.py            # Download utilities
│   │   └── filter_radicals.py
│   ├── config.py                          # Configuration settings (v2.8 model warnings)
│   └── data/
│       ├── input/                         # Text datasets (colors, numbers, etc.)
│       ├── images/                        # Saved visualizations
│       ├── metrics/                       # Analysis results
│       └── translations/                  # Translation pairs
├── requirements.txt                       # Updated for Streamlit 1.50.x
├── init_project.sh
├── CLAUDE.md                             # Development guidelines
├── README.md
└── LICENSE
```

## 🎯 Version 2.8 Enhancements

### 🛠️ **Streamlit 1.50.x Compatibility**
- ✅ **Fixed Widget Key Issues**: Resolved session state/widget key mismatches
- ✅ **Enhanced Text Area Handling**: Proper value initialization and persistence
- ✅ **Consistent State Management**: Unified approach to widget state handling

### 🎨 **3D Visualization Improvements**
- ✅ **Fixed Plotly Errors**: Removed unsupported griddash property from 3D scene axes
- ✅ **Interactive 3D Exploration**: Full zoom, rotate, and pan functionality
- ✅ **Clustering in 3D**: Advanced pattern discovery with color-coded clusters
- ✅ **Performance Optimization**: Smooth 3D rendering with large datasets

### 🈵 **Enhanced Chinese Character Processing**
- ✅ **Robust Tokenization**: Proper attention mask handling for Chinese text
- ✅ **NaN Error Prevention**: Detection and handling of embedding anomalies
- ✅ **Model-Specific Warnings**: User guidance for problematic model-text combinations
- ✅ **Preprocessing Pipeline**: Enhanced text normalization for Chinese characters

### 🖼️ **Advanced Image Management**
- ✅ **Intelligent Search**: Text-based and glob pattern image searching
- ✅ **Bulk Operations**: Select All/Clear All with persistent state management
- ✅ **Flexible Layouts**: 1/2/3 column grid options for comparison
- ✅ **Smart Organization**: Chronological sorting and efficient file handling

### 📈 **Performance & Reliability**
- ✅ **Memory Optimization**: Efficient caching and resource management
- ✅ **Error Recovery**: Graceful handling of model failures and edge cases
- ✅ **Session Persistence**: Reliable state management across app interactions
- ✅ **Compatibility Testing**: Verified functionality across Streamlit versions

## 🌐 Integration Ecosystem

This tool is designed as part of a larger language analysis ecosystem:

### Current Integration
- **Translation Services**: Built-in DeepL translator for dataset creation
- **Cross-platform Export**: High-quality visualizations for papers/presentations
- **Research Workflow**: Seamless data collection to publication pipeline

### Potential Extensions
- **Semantic Relationship Tracking**: Monitor concept evolution over time
- **Custom Vocabulary Building**: Create domain-specific semantic maps
- **Educational Tools**: Lesson planning based on semantic relationships
- **Interactive Learning**: Personalized paths through semantic space

## 🔬 Research Impact

### Major Discoveries
- ✅ **"Geometry of Meaning"** - Numbers form linear patterns in semantic space!
- ✅ **Cross-lingual Semantic Alignment** - Visual proof of shared conceptual structures
- ✅ **3D Manifold Visualization** - Complex semantic relationships revealed through interaction
- ✅ **Model Comparison Framework** - Systematic evaluation of embedding approaches

### Applications
- **Computational Linguistics**: Visual analysis of semantic structures
- **Language Learning**: Interactive exploration of cross-lingual concepts
- **AI Research**: Interpretability of embedding spaces
- **Educational Technology**: Data-driven language instruction

## 🤝 Contributing

We welcome contributions! Priority areas for v2.9:
- Additional embedding models and evaluation
- Enhanced error handling for edge cases
- New dimensionality reduction techniques
- More semantic category datasets
- Educational curriculum integration
- Performance optimizations for large datasets

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the intuitive web application framework ❤️
- **Claude Code** for development collaboration and debugging assistance 🤖
- **DeepL** for professional translation API
- **Hugging Face** for transformer models and embeddings
- **Plotly** for interactive visualization framework
- **PHATE** algorithm developers for manifold learning
- **Ollama** for local model serving
- **Open source community** for foundational tools

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{semantics_explorer_2025,
  title={Multilingual Embedding Explorer: Geometry of Meaning Visualization},
  author={Digital Duck Project},
  year={2025},
  url={https://github.com/digital-duck/st_semantics},
  note={Streamlit application for cross-lingual semantic analysis v2.8},
  version={2.8}
}
```

---

**Discover the hidden geometry of human meaning!** 🧠✨

*"Just as Descartes gave us coordinates for physical space, embeddings give us coordinates for mental space."*

> **Version 2.8 Breakthrough**: Enhanced compatibility with Streamlit 1.50.x, robust Chinese character processing, and stable 3D interactive visualization - ready for production research workflows!

### **The Numbers Discovery**
Chinese numerical concepts (零, 一, 二, 三...) form perfect linear patterns in semantic space, revealing the mathematical structure underlying human language cognition.

### **3D Interactive Visualization**
Rotate, zoom, and explore semantic manifolds in three dimensions - discover hidden clustering patterns that are invisible in 2D projections!