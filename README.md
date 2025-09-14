# Multilingual Embedding Explorer 🧠✨

**Geometry of Meaning: Visualizing Semantic Structure Across Languages**

A powerful Streamlit application for exploring multilingual word embeddings through interactive 2D and 3D visualizations. Discover how meaning has geometric structure using cutting-edge embedding models and manifold learning techniques.

*This Streamlit app is built with love in close collaboration with **Claude Code** ❤️*

> **Latest Version 3.0** - Now with publication-quality visualizations, advanced dual-view capabilities, and professional-grade code architecture!

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
- **Multiple Embedding Models**: Sentence-BERT Multilingual (fast), Ollama models (Snowflake-Arctic-Embed2, BGE-M3)
- **Advanced Dimensionality Reduction**: PHATE, t-SNE, UMAP, Isomap, PCA, MDS, LLE, Kernel PCA, Spectral Embedding
- **Interactive Visualizations**: 2D/3D plotting with clustering analysis and color-coded languages
- **Multilingual Support**: Chinese (source) + selectable target languages (English, French, Spanish, German)
- **ISO Language Codes**: Standardized 3-letter codes (chn, enu, fra, spa, deu) for consistent file naming
- **Real-time Rotation**: 90° plot rotation for different perspectives
- **Smart File Management**: Load/save text datasets with sanitized filenames
- **Dynamic Language Loading**: Automatic detection and loading of available language files
- **Session Caching**: Improved performance with cached embeddings
- **Publication Settings**: High-DPI export with customizable formatting
- **Geometric Analysis**: Advanced clustering, branching, and void analysis

### 🔍 **Semantics Explorer - Dual View** (Enhanced!)
- **Overview + Detail Views**: Simultaneous global and zoomed perspectives
- **Interactive Zoom Controls**: Precise navigation through semantic space
- **Pan Functionality**: Smooth movement through embedding space
- **Enhanced Statistics**: Real-time metrics and word lists in zoom areas
- **Publication-Ready Export**: High-quality image downloads with standardized naming
- **Geometric Analysis Integration**: Advanced pattern detection and visualization
- **Professional UI**: Clean, focused interface optimized for research workflow

### 🖼️ **Review Images**
- **Multi-image Comparison**: Side-by-side visualization analysis
- **Flexible Layouts**: 1, 2, or 3 images per row
- **Full Filename Visibility**: Complete filenames displayed in main panel
- **Batch Operations**: Select All/Clear All functionality
- **Download & Delete**: Manage saved visualizations efficiently
- **Smart Organization**: Sort by newest first

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

1. **Semantics Explorer**: Select target languages → Enter Chinese + target language words → Choose model & method → Visualize
2. **Multilingual Analysis**: Compare semantic relationships across up to 5 languages simultaneously
3. **Review Images**: Compare multiple visualizations side-by-side
4. **Translator**: Create multilingual datasets using professional translation

### Advanced Features

- **Load Text**: Use pre-built semantic categories or load custom datasets with automatic language detection
- **Save Text**: Create named datasets with automatic sanitization and ISO language codes
- **Language Selection**: Dynamic multi-target language support with color-coded visualization
- **File Naming**: Standardized format: `<dataset>-<lang_code>.txt` (e.g., `子-network-fra.txt`)
- **Rotate**: Adjust plot orientation for better pattern visibility
- **Save Image**: Export high-quality visualizations with descriptive filenames
- **Cross-compare**: Analyze multiple semantic categories and languages simultaneously

## 🔬 Model Performance Notes

- **Sentence-BERT Multilingual**: Recommended default - fast, reliable, excellent cross-lingual alignment
- **Snowflake-Arctic-Embed2**: Best performance for Chinese-English pairs
- **PHATE**: Excellent for revealing semantic manifold structures and branching patterns
- **Cross-lingual Discovery**: Chinese 红色 and English "Red" align geometrically in embedding space

## 📁 Project Structure

```
st_semantics/
├── src/
│   ├── Welcome.py                          # Main entry point with enhanced UI
│   ├── pages/
│   │   ├── 1_🔤_Semantics_Explorer.py      # Core visualization
│   │   ├── 2_🔍_Semantics_Explorer-Dual_View.py  # Advanced dual-view interface
│   │   ├── 3_🖼️_Review_Images.py           # Image comparison and management
│   │   └── 9_🌐_Translator.py              # Translation services
│   ├── components/
│   │   ├── shared/                         # Shared UI components (NEW)
│   │   │   └── publication_settings.py    # Reusable publication controls
│   │   ├── embedding_viz.py
│   │   ├── dimension_reduction.py
│   │   ├── plotting.py
│   │   ├── clustering.py
│   │   └── geometric_analysis.py           # Advanced geometric analysis
│   ├── models/                             # Model management
│   │   └── model_manager.py
│   ├── services/                           # External integrations
│   │   ├── google_translate.py
│   │   └── tts_service.py
│   ├── utils/                              # Utility functions
│   │   ├── error_handling.py               # Enhanced error handling
│   │   ├── download_helpers.py             # Download utilities (NEW)
│   │   └── filter_radicals.py
│   ├── config.py                           # Configuration settings
│   └── data/
│       ├── input/                          # Text datasets (colors, numbers, etc.)
│       ├── images/                         # Saved visualizations
│       ├── metrics/                        # Analysis results
│       └── translations/                   # Translation pairs
├── requirements.txt
├── init_project.sh
├── CODE_QUALITY_REPORT.md                 # Code quality documentation (NEW)
├── README-testing.md                      # Testing documentation (NEW)
├── FEEDBACK-anthropic.md                  # Development feedback (NEW)
├── README.md
└── LICENSE
```

## 🎯 Recent Enhancements (Version 3.0)

### 🚀 Major New Features
- ✅ **Advanced Dual View Interface** - Simultaneous overview and detail perspectives
- ✅ **Publication-Quality Export System** - High-DPI downloads with standardized naming
- ✅ **Multilingual Support Enhancement** - 5-language comparison with dynamic selection
- ✅ **ISO Language Codes** - Standardized 3-letter codes for file management
- ✅ **Geometric Analysis Integration** - Clustering, branching, and void analysis
- ✅ **Interactive Zoom & Pan Controls** - Precise navigation through semantic space
- ✅ **Enhanced Visualization Statistics** - Real-time metrics and word lists
- ✅ **Professional UI Components** - Shared, reusable interface elements

### 🏗️ Code Architecture Improvements (NEW!)
- ✅ **Eliminated Code Duplication** - 120+ lines removed via shared components
- ✅ **Function Decomposition** - Broke down 500+ line functions into focused modules
- ✅ **Enhanced Error Handling** - Consistent, professional error management
- ✅ **Download System Refactoring** - Centralized, extensible download functionality
- ✅ **Publication Standards** - Code quality suitable for academic publication

### 📊 Technical Improvements
- ✅ **Fixed torch compatibility** issues (torch 2.x support)
- ✅ **Streamlined model selection** (removed problematic models)
- ✅ **Better error handling** and user feedback
- ✅ **Session state persistence** for seamless workflow
- ✅ **Professional code architecture** with component separation
- ✅ **Performance optimizations** with smart caching (dimensionality reduction, file I/O)
- ✅ **Memory management** improvements for cleaner resource usage
- ✅ **Comprehensive Testing Framework** - Documentation and validation procedures

### 🔬 Research Impact
- ✅ **"Geometry of Meaning" discovery** - numbers form linear patterns!
- ✅ **Cross-lingual semantic alignment** visualization
- ✅ **Educational applications** for language learning
- ✅ **Publication-ready** visualizations and methodology
- ✅ **Advanced Pattern Detection** - Geometric analysis of semantic structures

## 🌐 Integration Ecosystem

This tool is designed as part of a larger language analysis ecosystem:

### Current Integration
- **Translation Services**: Built-in DeepL translator for dataset creation
- **Cross-platform Export**: High-quality visualizations for papers/presentations

### Potential Extensions
- **Semantic Relationship Tracking**: Monitor concept evolution over time
- **Custom Vocabulary Building**: Create domain-specific semantic maps
- **Educational Tools**: Lesson planning based on semantic relationships
- **Interactive Learning**: Personalized paths through semantic space

## Legacy Features

- Support for multiple embedding models (BERT, XLM-R, LASER, Ollama)
- Multiple dimensionality reduction methods (PHATE, t-SNE, UMAP, Isomap)
- Interactive 2D/3D visualizations with clustering
- Chinese-English word pair support
- Progress tracking and session caching
- Comprehensive error handling and logging


## 🤝 Contributing

We welcome contributions! Priority areas:
- Additional embedding models and evaluation
- New dimensionality reduction techniques
- Enhanced clustering algorithms
- More semantic category datasets
- Educational curriculum integration
- Performance optimizations

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the intuitive web application framework ❤️
- **Claude Code** for development collaboration and discussions 🤖
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
  note={Streamlit application for cross-lingual semantic analysis}
}
```

---

**Discover the hidden geometry of human meaning!** 🧠✨

*"Just as Descartes gave us coordinates for physical space, embeddings give us coordinates for mental space."*

> **The Numbers Discovery**: Chinese numerical concepts (零, 一, 二, 三...) form perfect linear patterns in semantic space, revealing the mathematical structure underlying human language cognition.