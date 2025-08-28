# Embedding Evaluation Dataset Documentation

## Purpose
Comprehensive bilingual dataset designed for systematic evaluation of multilingual embedding models using PHATE manifold analysis. Tests semantic organization across multiple conceptual domains.

## Dataset Statistics
- **Chinese Characters**: 225 characters across 13 semantic domains
- **English Words**: 225 corresponding words across same domains  
- **Coverage**: Fundamental concepts spanning concrete to abstract semantics
- **Cross-linguistic Alignment**: Direct semantic correspondence between languages

## Semantic Domain Structure

### 1. **Numbers (15 items)** - Linear Control Group
- **Purpose**: Test for linear arrangement in embedding space (proven pattern)
- **Expected PHATE Pattern**: Linear sequence (零→一→二...→一百)
- **Cross-linguistic**: Perfect numerical correspondence

### 2. **Colors (14 items)** - Perceptual Category  
- **Purpose**: Test perceptual clustering (warm/cool color families)
- **Expected PHATE Pattern**: Clustered by color families (red-orange-yellow vs blue-green-purple)
- **Cross-linguistic**: Universal color perception patterns

### 3. **Family/Kinship (16 items)** - Core 子 Network
- **Purpose**: Test traditional Chinese semantic field organization
- **Expected PHATE Pattern**: Hierarchical family structure clustering
- **Special**: Includes 子 character derivatives and semantic extensions

### 4. **Animals (15 items)** - Biological Taxonomy
- **Purpose**: Test biological categorization (pets vs wild, mammals vs birds)
- **Expected PHATE Pattern**: Clustered by habitat/domestication/size
- **Cross-linguistic**: Universal biological classification

### 5. **Body Parts (15 items)** - Physical Structure
- **Purpose**: Test anatomical organization (head/face vs limbs vs organs)
- **Expected PHATE Pattern**: Clustered by body region/function
- **Embodied cognition**: Universal human body conceptualization

### 6. **Emotions (15 items)** - Abstract Psychology
- **Purpose**: Test affective dimension organization (positive/negative valence)
- **Expected PHATE Pattern**: Organized by emotional valence and intensity
- **Cross-cultural**: Test universal vs culture-specific emotion concepts

### 7. **Time/Temporal (16 items)** - Cyclical Concepts
- **Purpose**: Test temporal organization (linear time vs cyclical seasons)
- **Expected PHATE Pattern**: Mixed linear (past→present→future) and cyclical (seasons)
- **Chinese specialty**: Includes traditional time markers (子时)

### 8. **Elements/Nature (15 items)** - Traditional 五行 System
- **Purpose**: Test physics-inspired Chinese character organization
- **Expected PHATE Pattern**: 五行 clustering (metal-wood-water-fire-earth groups)
- **Cultural specificity**: Tests Chinese traditional classification vs Western elements

### 9. **Food/Eating (15 items)** - Daily Life
- **Purpose**: Test practical concept organization (staples vs condiments)
- **Expected PHATE Pattern**: Clustered by food type/cooking method
- **Cultural variation**: Rice-centric (Chinese) vs diverse grains

### 10. **Education/Learning (15 items)** - 子 Semantic Branch
- **Purpose**: Test educational concept clustering linked to 子 character family
- **Expected PHATE Pattern**: Developmental progression (字→读→写→学)
- **ZiNets connection**: Direct test of 子 semantic expansion theory

### 11. **Tools/Objects (15 items)** - Functional Categories
- **Purpose**: Test functional clustering (furniture vs infrastructure)
- **Expected PHATE Pattern**: Grouped by function/location (home vs outdoor)
- **Practical semantics**: Everyday object categorization

### 12. **Actions/Verbs (15 items)** - Dynamic Concepts  
- **Purpose**: Test action categorization (motion vs cognitive vs sensory)
- **Expected PHATE Pattern**: Clustered by action type/body involvement
- **Embodied cognition**: Motor-semantic connections

### 13. **Directional/Spatial (15 items)** - Geometric Relations
- **Purpose**: Test spatial concept organization (cardinal directions, relative positions)
- **Expected PHATE Pattern**: Geometric arrangement reflecting spatial relationships
- **Cognitive mapping**: Universal spatial cognition patterns

### 14. **Abstract Qualities (16 items)** - Complex Concepts
- **Purpose**: Test high-level abstraction clustering (evaluative dimensions)
- **Expected PHATE Pattern**: Organized by evaluative dimensions (good/bad, big/small)
- **Semantic complexity**: Most abstract conceptual relationships

## Evaluation Criteria

### Expected PHATE Clustering Quality
1. **High Intra-domain Coherence**: Items within semantic domains cluster together
2. **Clear Inter-domain Separation**: Distinct boundaries between semantic categories
3. **Cross-linguistic Alignment**: Chinese-English semantic correspondence preserved
4. **Known Pattern Replication**: Numbers show linear arrangement, colors cluster by families

### Model Performance Indicators
- **Excellent**: Clear clustering with 90%+ intra-domain coherence
- **Good**: Recognizable clustering with some boundary overlap  
- **Poor**: Random distribution, no semantic organization detected
- **Failed**: Opposite patterns (e.g., numbers randomly distributed)

### Embedding Model Test Protocol
1. **Load both datasets** (Chinese + English) into embedding model
2. **Generate embeddings** for all 450 items (225 × 2 languages)
3. **Apply PHATE** with consistent parameters across models
4. **Visualize results** with semantic domain color coding
5. **Quantitative assessment** using clustering metrics
6. **Cross-linguistic analysis** measuring Chinese-English alignment

## Usage Instructions

```bash
# Load datasets in Streamlit app
Chinese file: data/input/embedding-eval-dataset-chn.txt  
English file: data/input/embedding-eval-dataset-enu.txt

# Recommended PHATE parameters for comparison
k-neighbors: 10
alpha: 10  
t (diffusion time): 20
dimensions: 2D for visualization, 3D for detailed analysis
```

## Expected Research Outcomes

This dataset will enable:
- **Systematic comparison** of 13 embedding models on Chinese character semantics
- **Identification** of best-performing model for Chinese character analysis  
- **Validation** of cross-linguistic semantic universals
- **Discovery** of model-specific strengths/weaknesses across semantic domains
- **Empirical evidence** for "Geometry of Meaning" hypothesis

The results will directly inform our CERN paper methodology and provide robust empirical foundation for Chinese character embedding analysis.