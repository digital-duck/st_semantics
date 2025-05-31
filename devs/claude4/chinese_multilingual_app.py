"""
Chinese Multilingual Embedding Models & Bilingual Character Visualizer
=====================================================================

This tool provides the best pre-trained models for Chinese text and creates
bilingual visualizations of character families (e.g., 日-derived characters).

Features:
- Multiple state-of-the-art Chinese/multilingual embedding models
- Bilingual Chinese-English character visualization  
- Character family analysis (e.g., all characters containing 日)
- Cross-lingual semantic structure exploration
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Embedding models
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import umap
import phate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Analysis tools
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class ChineseMultilingualEmbeddings:
    """
    Collection of the best Chinese/multilingual embedding models
    """
    
    def __init__(self):
        self.models = {}
        self.model_info = {
            # Best Chinese-specific models
            'chinese-roberta-wwm-ext': {
                'name': 'Chinese RoBERTa (Whole Word Masking)',
                'description': 'Best Chinese-only model, excellent for character semantics',
                'languages': ['Chinese'],
                'dimensions': 768,
                'model_id': 'hfl/chinese-roberta-wwm-ext',
                'type': 'transformers'
            },
            
            'chinese-macbert-base': {
                'name': 'Chinese MacBERT',
                'description': 'Improved Chinese BERT with better character understanding',
                'languages': ['Chinese'],
                'dimensions': 768,
                'model_id': 'hfl/chinese-macbert-base',
                'type': 'transformers'
            },
            
            # Best multilingual models for Chinese-English
            'multilingual-e5-large': {
                'name': 'Multilingual E5-Large',
                'description': 'State-of-the-art multilingual embeddings, excellent Chinese support',
                'languages': ['Chinese', 'English', '100+ others'],
                'dimensions': 1024,
                'model_id': 'intfloat/multilingual-e5-large',
                'type': 'sentence_transformers'
            },
            
            'multilingual-e5-base': {
                'name': 'Multilingual E5-Base',
                'description': 'Balanced performance/speed for Chinese-English tasks',
                'languages': ['Chinese', 'English', '100+ others'],
                'dimensions': 768,
                'model_id': 'intfloat/multilingual-e5-base',
                'type': 'sentence_transformers'
            },
            
            'paraphrase-multilingual-mpnet': {
                'name': 'Paraphrase Multilingual MPNet',
                'description': 'Excellent for semantic similarity across languages',
                'languages': ['Chinese', 'English', '50+ others'],
                'dimensions': 768,
                'model_id': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                'type': 'sentence_transformers'
            },
            
            'distiluse-multilingual': {
                'name': 'DistilUSE Multilingual',
                'description': 'Fast and efficient for Chinese-English bilingual tasks',
                'languages': ['Chinese', 'English', '15+ others'],
                'dimensions': 512,
                'model_id': 'sentence-transformers/distiluse-base-multilingual-cased',
                'type': 'sentence_transformers'
            },
            
            # Specialized Chinese models
            'chinese-clip': {
                'name': 'Chinese CLIP',
                'description': 'Vision-language model with strong Chinese text understanding',
                'languages': ['Chinese'],
                'dimensions': 512,
                'model_id': 'OFA-Sys/chinese-clip-vit-base-patch16',
                'type': 'clip'
            },
            
            'chinese-bert-wwm': {
                'name': 'Chinese BERT WWM',
                'description': 'Original Chinese BERT with whole word masking',
                'languages': ['Chinese'],
                'dimensions': 768,
                'model_id': 'hfl/chinese-bert-wwm-ext',
                'type': 'transformers'
            },
            
            # Latest multilingual models
            'bge-m3': {
                'name': 'BGE-M3',
                'description': 'Latest Chinese multilingual model with excellent performance',
                'languages': ['Chinese', 'English', '100+ others'],
                'dimensions': 1024,
                'model_id': 'BAAI/bge-m3',
                'type': 'sentence_transformers'
            },
            
            'gte-multilingual': {
                'name': 'GTE Multilingual',
                'description': 'General text embeddings with strong Chinese support',
                'languages': ['Chinese', 'English', '多语言'],
                'dimensions': 768,
                'model_id': 'thenlper/gte-multilingual-base',
                'type': 'sentence_transformers'
            }
        }
    
    def load_sentence_transformer_model(self, model_id):
        """Load sentence transformer model"""
        if model_id not in self.models:
            try:
                self.models[model_id] = SentenceTransformer(model_id)
                return True
            except Exception as e:
                st.error(f"Failed to load {model_id}: {str(e)}")
                return False
        return True
    
    def load_transformers_model(self, model_id):
        """Load transformers model for custom embedding extraction"""
        if model_id not in self.models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id)
                self.models[model_id] = {'tokenizer': tokenizer, 'model': model}
                return True
            except Exception as e:
                st.error(f"Failed to load {model_id}: {str(e)}")
                return False
        return True
    
    def get_sentence_transformer_embeddings(self, texts, model_id):
        """Get embeddings from sentence transformer models"""
        if not self.load_sentence_transformer_model(model_id):
            return None
        
        model = self.models[model_id]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
    def get_transformers_embeddings(self, texts, model_id):
        """Get embeddings from transformers models"""
        if not self.load_transformers_model(model_id):
            return None
        
        tokenizer = self.models[model_id]['tokenizer']
        model = self.models[model_id]['model']
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and encode
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                
                # Use [CLS] token embedding or mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling of all tokens
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                else:
                    # Use pooler output if available
                    embedding = outputs.pooler_output.squeeze().numpy()
                
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_embeddings(self, texts, model_key):
        """Get embeddings using the specified model"""
        model_info = self.model_info[model_key]
        model_id = model_info['model_id']
        model_type = model_info['type']
        
        if model_type == 'sentence_transformers':
            return self.get_sentence_transformer_embeddings(texts, model_id)
        elif model_type == 'transformers':
            return self.get_transformers_embeddings(texts, model_id)
        else:
            st.error(f"Model type {model_type} not supported yet")
            return None

def create_ri_character_dataset():
    """
    Create dataset of characters derived from 日 (sun/day) with English translations
    """
    
    ri_characters = {
        # Core 日 characters
        '日': {
            'meanings_zh': '太阳；日子；白天',
            'meanings_en': 'sun; day; daytime',
            'category': 'core',
            'stroke_count': 4,
            'frequency': 'very_high'
        },
        
        # Time-related characters with 日
        '时': {
            'meanings_zh': '时间；时候；小时',
            'meanings_en': 'time; hour; moment',
            'category': 'time',
            'stroke_count': 7,
            'frequency': 'very_high'
        },
        
        '明': {
            'meanings_zh': '明亮；聪明；明天',
            'meanings_en': 'bright; intelligent; tomorrow',
            'category': 'brightness',
            'stroke_count': 8,
            'frequency': 'very_high'
        },
        
        '早': {
            'meanings_zh': '早晨；早期；提前',
            'meanings_en': 'morning; early; ahead of time',
            'category': 'time',
            'stroke_count': 6,
            'frequency': 'high'
        },
        
        '晚': {
            'meanings_zh': '晚上；迟到；晚期',
            'meanings_en': 'evening; late; later period',
            'category': 'time',
            'stroke_count': 11,
            'frequency': 'high'
        },
        
        '春': {
            'meanings_zh': '春天；春季；青春',
            'meanings_en': 'spring; springtime; youth',
            'category': 'season',
            'stroke_count': 9,
            'frequency': 'high'
        },
        
        '昨': {
            'meanings_zh': '昨天；昨日',
            'meanings_en': 'yesterday; the day before',
            'category': 'time',
            'stroke_count': 9,
            'frequency': 'high'
        },
        
        '旧': {
            'meanings_zh': '旧的；以前的；过时的',
            'meanings_en': 'old; former; outdated',
            'category': 'time',
            'stroke_count': 5,
            'frequency': 'medium'
        },
        
        # Brightness/light characters
        '昭': {
            'meanings_zh': '明显；显著；昭示',
            'meanings_en': 'evident; obvious; to reveal',
            'category': 'brightness',
            'stroke_count': 9,
            'frequency': 'medium'
        },
        
        '晃': {
            'meanings_zh': '摇摆；闪烁；晃动',
            'meanings_en': 'to sway; to flash; to shake',
            'category': 'brightness',
            'stroke_count': 10,
            'frequency': 'medium'
        },
        
        '晨': {
            'meanings_zh': '早晨；清晨；黎明',
            'meanings_en': 'morning; dawn; daybreak',
            'category': 'time',
            'stroke_count': 11,
            'frequency': 'medium'
        },
        
        '暖': {
            'meanings_zh': '温暖；暖和；暖气',
            'meanings_en': 'warm; heat; heating',
            'category': 'temperature',
            'stroke_count': 13,
            'frequency': 'medium'
        },
        
        '暗': {
            'meanings_zh': '黑暗；暗淡；秘密',
            'meanings_en': 'dark; dim; secret',
            'category': 'brightness',
            'stroke_count': 13,
            'frequency': 'medium'
        },
        
        # Calendar/period characters
        '星': {
            'meanings_zh': '星星；星期；明星',
            'meanings_en': 'star; week; celebrity',
            'category': 'celestial',
            'stroke_count': 9,
            'frequency': 'high'
        },
        
        '期': {
            'meanings_zh': '期间；期望；日期',
            'meanings_en': 'period; to expect; date',
            'category': 'time',
            'stroke_count': 12,
            'frequency': 'high'
        },
        
        # Less common but semantically related
        '晶': {
            'meanings_zh': '晶体；水晶；明亮',
            'meanings_en': 'crystal; bright; clear',
            'category': 'brightness',
            'stroke_count': 12,
            'frequency': 'low'
        },
        
        '暴': {
            'meanings_zh': '暴力；暴雨；暴露',
            'meanings_en': 'violent; rainstorm; to expose',
            'category': 'intensity',
            'stroke_count': 15,
            'frequency': 'medium'
        },
        
        '曰': {
            'meanings_zh': '说；叫做；称为',
            'meanings_en': 'to say; to be called; namely',
            'category': 'speech',
            'stroke_count': 4,
            'frequency': 'low'
        },
        
        '曝': {
            'meanings_zh': '曝光；暴露；晒',
            'meanings_en': 'to expose; to reveal; to dry in sun',
            'category': 'exposure',
            'stroke_count': 19,
            'frequency': 'low'
        },
        
        '景': {
            'meanings_zh': '风景；景色；情景',
            'meanings_en': 'scenery; view; situation',
            'category': 'visual',
            'stroke_count': 12,
            'frequency': 'medium'
        }
    }
    
    # Convert to DataFrame
    data = []
    for char, info in ri_characters.items():
        data.append({
            'character': char,
            'meanings_chinese': info['meanings_zh'],
            'meanings_english': info['meanings_en'],
            'category': info['category'],
            'stroke_count': info['stroke_count'],
            'frequency': info['frequency'],
            'bilingual_text': f"Character: {char}. Chinese meanings: {info['meanings_zh']}. English meanings: {info['meanings_en']}"
        })
    
    return pd.DataFrame(data)

def apply_manifold_learning(embeddings, method='umap', **params):
    """Apply manifold learning to embeddings"""
    
    if method.lower() == 'umap':
        reducer = umap.UMAP(
            n_neighbors=params.get('n_neighbors', 10),
            min_dist=params.get('min_dist', 0.1),
            n_components=2,
            metric=params.get('metric', 'cosine'),
            random_state=42
        )
    elif method.lower() == 'phate':
        reducer = phate.PHATE(
            k=params.get('k', 8),
            t=params.get('t', 20),
            n_components=2,
            n_jobs=1
        )
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    return coords, reducer

def create_bilingual_visualization(embeddings, character_data, manifold_coords, model_name):
    """Create bilingual visualization of character family"""
    
    # Main scatter plot
    fig = px.scatter(
        x=manifold_coords[:, 0],
        y=manifold_coords[:, 1],
        color=character_data['category'],
        size=character_data['stroke_count'],
        hover_data={
            'Character': character_data['character'],
            'Chinese': character_data['meanings_chinese'],
            'English': character_data['meanings_english'],
            'Category': character_data['category'],
            'Frequency': character_data['frequency']
        },
        title=f'日-Family Characters: Bilingual Semantic Space ({model_name})',
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        width=900,
        height=600
    )
    
    # Add character labels
    for i, row in character_data.iterrows():
        fig.add_annotation(
            x=manifold_coords[i, 0],
            y=manifold_coords[i, 1],
            text=row['character'],
            showarrow=False,
            font=dict(size=16, color='black'),
            bgcolor='white',
            opacity=0.8
        )
    
    fig.update_layout(
        title_font_size=16,
        showlegend=True
    )
    
    return fig

def analyze_bilingual_patterns(embeddings, character_data):
    """Analyze patterns in bilingual embeddings"""
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find most similar character pairs
    n_chars = len(character_data)
    similar_pairs = []
    
    for i in range(n_chars):
        for j in range(i+1, n_chars):
            similarity = similarity_matrix[i, j]
            similar_pairs.append({
                'char1': character_data.iloc[i]['character'],
                'char2': character_data.iloc[j]['character'],
                'similarity': similarity,
                'category1': character_data.iloc[i]['category'],
                'category2': character_data.iloc[j]['category'],
                'same_category': character_data.iloc[i]['category'] == character_data.iloc[j]['category']
            })
    
    # Sort by similarity
    similar_pairs = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
    
    # Category coherence analysis
    category_coherence = {}
    categories = character_data['category'].unique()
    
    for cat in categories:
        cat_indices = character_data[character_data['category'] == cat].index
        if len(cat_indices) > 1:
            cat_similarities = []
            for i in cat_indices:
                for j in cat_indices:
                    if i < j:
                        cat_similarities.append(similarity_matrix[i, j])
            
            category_coherence[cat] = {
                'avg_similarity': np.mean(cat_similarities),
                'std_similarity': np.std(cat_similarities),
                'count': len(cat_indices)
            }
    
    return similar_pairs, category_coherence

def create_streamlit_app():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Chinese Multilingual Character Explorer",
        page_icon="🌏",
        layout="wide"
    )
    
    # Header
    st.markdown("# 🌏 Chinese Multilingual Character Explorer")
    st.markdown("### Visualize Chinese character families with bilingual embeddings")
    
    # Sidebar - Model selection
    st.sidebar.header("🧠 Embedding Model Selection")
    
    embedder = ChineseMultilingualEmbeddings()
    
    # Model category filter
    model_category = st.sidebar.selectbox(
        "Model Category",
        ["All Models", "Chinese-Only", "Multilingual", "Latest/Best"]
    )
    
    # Filter models based on category
    if model_category == "Chinese-Only":
        available_models = {k: v for k, v in embedder.model_info.items() 
                          if 'Chinese' in v['languages'] and len(v['languages']) == 1}
    elif model_category == "Multilingual":
        available_models = {k: v for k, v in embedder.model_info.items() 
                          if len(v['languages']) > 1}
    elif model_category == "Latest/Best":
        available_models = {k: v for k, v in embedder.model_info.items() 
                          if k in ['multilingual-e5-large', 'bge-m3', 'chinese-macbert-base']}
    else:
        available_models = embedder.model_info
    
    # Model selection
    selected_model_key = st.sidebar.selectbox(
        "Select Embedding Model",
        list(available_models.keys()),
        format_func=lambda x: available_models[x]['name']
    )
    
    # Display model info
    model_info = available_models[selected_model_key]
    st.sidebar.markdown(f"**{model_info['name']}**")
    st.sidebar.markdown(f"*{model_info['description']}*")
    st.sidebar.markdown(f"**Languages:** {', '.join(model_info['languages'])}")
    st.sidebar.markdown(f"**Dimensions:** {model_info['dimensions']}")
    
    # Manifold learning settings
    st.sidebar.subheader("🗺️ Manifold Learning")
    manifold_method = st.sidebar.selectbox("Method", ["UMAP", "PHATE", "PCA"])
    
    if manifold_method == "UMAP":
        n_neighbors = st.sidebar.slider("Neighbors", 3, 15, 8)
        min_dist = st.sidebar.slider("Min Distance", 0.0, 0.5, 0.1, 0.05)
    elif manifold_method == "PHATE":
        k_neighbors = st.sidebar.slider("K Neighbors", 3, 15, 8)
        diffusion_time = st.sidebar.slider("Diffusion Time", 10, 50, 20)
    
    # Analysis button
    run_analysis = st.sidebar.button("🚀 Analyze 日-Family Characters", type="primary")
    
    # Main content
    if not run_analysis:
        # Show model comparison table
        st.subheader("📊 Available Chinese/Multilingual Embedding Models")
        
        model_comparison = pd.DataFrame([
            {
                'Model': info['name'],
                'Languages': ', '.join(info['languages'][:2]) + ('...' if len(info['languages']) > 2 else ''),
                'Dimensions': info['dimensions'],
                'Best For': info['description'].split(',')[0],
                'Key': key
            }
            for key, info in embedder.model_info.items()
        ])
        
        st.dataframe(model_comparison, use_container_width=True)
        
        # Show example character family
        st.subheader("🌅 Example: 日-Family Characters")
        st.markdown("Characters derived from or related to 日 (sun/day):")
        
        example_data = create_ri_character_dataset()
        st.dataframe(example_data[['character', 'meanings_chinese', 'meanings_english', 'category']], 
                    use_container_width=True)
        
        st.info("👆 Select a model and click 'Analyze' to explore the semantic structure!")
        
        return
    
    # Run analysis
    with st.spinner(f"🧠 Loading {model_info['name']} model..."):
        # Create character dataset
        character_data = create_ri_character_dataset()
        
        # Extract embeddings
        texts = character_data['bilingual_text'].tolist()
        embeddings = embedder.get_embeddings(texts, selected_model_key)
        
        if embeddings is None:
            st.error("❌ Failed to load embeddings. Please try a different model.")
            return
    
    st.success(f"✅ Extracted {embeddings.shape[1]}D embeddings for {len(character_data)} characters")
    
    # Apply manifold learning
    with st.spinner(f"🗺️ Applying {manifold_method} manifold learning..."):
        if manifold_method == "UMAP":
            manifold_coords, reducer = apply_manifold_learning(
                embeddings, 'umap', n_neighbors=n_neighbors, min_dist=min_dist
            )
        elif manifold_method == "PHATE":
            manifold_coords, reducer = apply_manifold_learning(
                embeddings, 'phate', k=k_neighbors, t=diffusion_time
            )
        else:
            manifold_coords, reducer = apply_manifold_learning(embeddings, 'pca')
    
    # Analyze patterns
    with st.spinner("🔍 Analyzing bilingual patterns..."):
        similar_pairs, category_coherence = analyze_bilingual_patterns(embeddings, character_data)
    
    # Display results
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Bilingual Visualization",
        "📊 Similarity Analysis", 
        "🎯 Category Coherence",
        "🔢 Embedding Analysis"
    ])
    
    with tab1:
        st.header("日-Family Character Semantic Space")
        
        # Create and display visualization
        fig = create_bilingual_visualization(
            embeddings, character_data, manifold_coords, model_info['name']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Characters Analyzed", len(character_data))
        with col2:
            st.metric("Semantic Categories", len(character_data['category'].unique()))
        with col3:
            st.metric("Embedding Dimensions", embeddings.shape[1])
        with col4:
            avg_similarity = np.mean([pair['similarity'] for pair in similar_pairs])
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
        
        # Character details
        st.subheader("📝 Character Details")
        display_data = character_data[['character', 'meanings_chinese', 'meanings_english', 'category', 'stroke_count']]
        st.dataframe(display_data, use_container_width=True)
    
    with tab2:
        st.header("Character Similarity Analysis")
        
        # Most similar pairs
        st.subheader("🔗 Most Similar Character Pairs")
        similarity_df = pd.DataFrame(similar_pairs[:10])
        similarity_df['similarity'] = similarity_df['similarity'].round(3)
        st.dataframe(similarity_df, use_container_width=True)
        
        # Similarity heatmap
        st.subheader("🌡️ Character Similarity Matrix")
        
        similarity_matrix = cosine_similarity(embeddings)
        
        fig_heatmap = px.imshow(
            similarity_matrix,
            x=character_data['character'],
            y=character_data['character'],
            color_continuous_scale='Viridis',
            title="Character Semantic Similarity Matrix"
        )
        
        fig_heatmap.update_layout(width=600, height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.header("Semantic Category Analysis")
        
        # Category coherence metrics
        st.subheader("📊 Category Coherence Scores")
        
        coherence_data = []
        for cat, stats in category_coherence.items():
            coherence_data.append({
                'Category': cat,
                'Avg Similarity': round(stats['avg_similarity'], 3),
                'Std Deviation': round(stats['std_similarity'], 3),
                'Character Count': stats['count'],
                'Coherence Score': round(stats['avg_similarity'] - stats['std_similarity'], 3)
            })
        
        coherence_df = pd.DataFrame(coherence_data)
        st.dataframe(coherence_df, use_container_width=True)
        
        # Category visualization
        fig_categories = px.bar(
            coherence_df,
            x='Category',
            y='Avg Similarity',
            error_y='Std Deviation',
            title="Semantic Category Coherence",
            color='Coherence Score',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig_categories, use_container_width=True)
        
        # Category distribution in manifold space
        st.subheader("🎨 Category Distribution")
        
        fig_cat_dist = px.scatter(
            x=manifold_coords[:, 0],
            y=manifold_coords[:, 1],
            color=character_data['category'],
            size=character_data['stroke_count'],
            title="Category Distribution in Manifold Space",
            labels={'x': f'{manifold_method} 1', 'y': f'{manifold_method} 2'}
        )
        
        st.plotly_chart(fig_cat_dist, use_container_width=True)
    
    with tab4:
        st.header("Embedding Space Analysis")
        
        # PCA comparison
        st.subheader("📐 PCA Analysis of Original Embeddings")
        
        pca_coords, _ = apply_manifold_learning(embeddings, 'pca')
        
        fig_pca = px.scatter(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1],
            color=character_data['category'],
            text=character_data['character'],
            title="PCA of High-Dimensional Embeddings",
            labels={'x': 'PC1', 'y': 'PC2'}
        )
        
        fig_pca.update_traces(textposition="top center")
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Embedding statistics
        st.subheader("📊 Embedding Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Embedding Properties:**")
            st.write(f"- Dimensions: {embeddings.shape[1]}")
            st.write(f"- Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")
            st.write(f"- Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.3f}")
            
        with col2:
            st.write("**Similarity Distribution:**")
            all_similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            st.write(f"- Mean similarity: {np.mean(all_similarities):.3f}")
            st.write(f"- Std similarity: {np.std(all_similarities):.3f}")
            st.write(f"- Max similarity: {np.max(all_similarities):.3f}")
            st.write(f"- Min similarity: {np.min(all_similarities):.3f}")
        
        # Export options
        st.subheader("💾 Export Analysis Results")
        
        if st.button("📥 Download Results"):
            # Prepare export data
            export_data = character_data.copy()
            export_data[f'{manifold_method}_x'] = manifold_coords[:, 0]
            export_data[f'{manifold_method}_y'] = manifold_coords[:, 1]
            
            # Add embedding dimensions (first 10 for file size)
            for i in range(min(10, embeddings.shape[1])):
                export_data[f'embedding_dim_{i}'] = embeddings[:, i]
            
            # Add similarity scores to most similar character
            most_similar_chars = []
            for i in range(len(character_data)):
                similarities = similarity_matrix[i]
                similarities[i] = -1  # Exclude self
                most_similar_idx = np.argmax(similarities)
                most_similar_chars.append({
                    'most_similar_char': character_data.iloc[most_similar_idx]['character'],
                    'similarity_score': similarities[most_similar_idx]
                })
            
            export_data['most_similar_character'] = [item['most_similar_char'] for item in most_similar_chars]
            export_data['max_similarity_score'] = [item['similarity_score'] for item in most_similar_chars]
            
            csv_data = export_data.to_csv(index=False)
            st.download_button(
                label="Download Analysis CSV",
                data=csv_data,
                file_name=f"ri_family_analysis_{selected_model_key}_{manifold_method}.csv",
                mime="text/csv"
            )
            
            # Also export similarity matrix
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=character_data['character'],
                columns=character_data['character']
            )
            
            similarity_csv = similarity_df.to_csv()
            st.download_button(
                label="Download Similarity Matrix CSV",
                data=similarity_csv,
                file_name=f"ri_family_similarity_matrix_{selected_model_key}.csv",
                mime="text/csv"
            )

def create_extended_character_families():
    """
    Create datasets for other character families for future exploration
    """
    
    families = {
        'water_family': {
            'radical': '水',
            'characters': ['水', '河', '海', '湖', '江', '池', '流', '波', '洋', '泪', '汗', '洗', '游', '泡', '湿']
        },
        'fire_family': {
            'radical': '火',
            'characters': ['火', '炎', '燃', '烧', '热', '焰', '烟', '炸', '灭', '灯', '烤', '煮', '炒', '炉', '焦']
        },
        'heart_family': {
            'radical': '心',
            'characters': ['心', '思', '想', '愛', '怕', '怒', '悲', '喜', '憂', '懼', '念', '忘', '忙', '急', '感']
        },
        'wood_family': {
            'radical': '木',
            'characters': ['木', '树', '林', '森', '枝', '叶', '根', '果', '花', '草', '植', '桌', '椅', '床', '门']
        },
        'hand_family': {
            'radical': '手',
            'characters': ['手', '拿', '放', '打', '推', '拉', '握', '抓', '摸', '写', '画', '指', '掌', '拳', '抱']
        }
    }
    
    return families

def create_model_comparison_interface():
    """
    Interface for comparing multiple models on the same character set
    """
    
    st.subheader("🔬 Model Comparison Mode")
    
    selected_models = st.multiselect(
        "Select models to compare",
        list(ChineseMultilingualEmbeddings().model_info.keys()),
        default=['multilingual-e5-base', 'chinese-macbert-base'],
        format_func=lambda x: ChineseMultilingualEmbeddings().model_info[x]['name']
    )
    
    if len(selected_models) >= 2 and st.button("Compare Models"):
        embedder = ChineseMultilingualEmbeddings()
        character_data = create_ri_character_dataset()
        texts = character_data['bilingual_text'].tolist()
        
        comparison_results = {}
        
        for model_key in selected_models:
            with st.spinner(f"Processing {embedder.model_info[model_key]['name']}..."):
                embeddings = embedder.get_embeddings(texts, model_key)
                if embeddings is not None:
                    manifold_coords, _ = apply_manifold_learning(embeddings, 'umap', n_neighbors=8, min_dist=0.1)
                    similarity_matrix = cosine_similarity(embeddings)
                    
                    comparison_results[model_key] = {
                        'embeddings': embeddings,
                        'manifold_coords': manifold_coords,
                        'avg_similarity': np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
                        'name': embedder.model_info[model_key]['name']
                    }
        
        # Create comparison visualization
        if comparison_results:
            n_models = len(comparison_results)
            fig = make_subplots(
                rows=1, cols=n_models,
                subplot_titles=[results['name'] for results in comparison_results.values()],
                specs=[[{"type": "scatter"}] * n_models]
            )
            
            colors = px.colors.qualitative.Set1
            
            for i, (model_key, results) in enumerate(comparison_results.items()):
                coords = results['manifold_coords']
                
                for j, category in enumerate(character_data['category'].unique()):
                    mask = character_data['category'] == category
                    fig.add_trace(
                        go.Scatter(
                            x=coords[mask, 0],
                            y=coords[mask, 1],
                            mode='markers+text',
                            text=character_data[mask]['character'],
                            textposition="middle center",
                            name=f"{category}" if i == 0 else f"{category}_{i}",
                            marker=dict(color=colors[j % len(colors)], size=10),
                            showlegend=i == 0
                        ),
                        row=1, col=i+1
                    )
            
            fig.update_layout(
                title="Model Comparison: 日-Family Character Embeddings",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison metrics
            st.subheader("📊 Model Performance Comparison")
            
            metrics_data = []
            for model_key, results in comparison_results.items():
                metrics_data.append({
                    'Model': results['name'],
                    'Avg Similarity': round(results['avg_similarity'], 4),
                    'Embedding Dims': results['embeddings'].shape[1],
                    'Model Key': model_key
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

if __name__ == "__main__":
    # Add custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .character-text {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    create_streamlit_app()
    
    # Add model comparison section
    st.markdown("---")
    create_model_comparison_interface()
    
    # Footer with model recommendations
    st.markdown("---")
    st.markdown("### 🎯 Model Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🥇 Best Overall**
        - **BGE-M3**: Latest, best performance
        - **Multilingual E5-Large**: Excellent Chinese-English
        - **Chinese MacBERT**: Best Chinese-only
        """)
    
    with col2:
        st.markdown("""
        **⚡ Fast & Efficient**
        - **DistilUSE Multilingual**: Good speed/quality balance
        - **Multilingual E5-Base**: Smaller than large version
        - **Chinese RoBERTa**: Fast Chinese processing
        """)
    
    with col3:
        st.markdown("""
        **🎯 Specialized Uses**
        - **Chinese CLIP**: Visual + text understanding
        - **GTE Multilingual**: General text embeddings
        - **Paraphrase MPNet**: Semantic similarity tasks
        """)
    
    st.markdown("""
    ### 💡 Usage Tips:
    - **For your 6000 characters**: Try **BGE-M3** or **Multilingual E5-Large** first
    - **For Chinese-only analysis**: Use **Chinese MacBERT** or **Chinese RoBERTa**  
    - **For speed**: Start with **DistilUSE Multilingual**
    - **For research**: Compare multiple models to see which reveals the best semantic structure
    """)

"""
Quick Start Guide:
==================

1. Install requirements:
   pip install streamlit sentence-transformers transformers torch plotly umap-learn phate

2. Run the app:
   streamlit run chinese_multilingual_app.py

3. Upload your character dataset or use the built-in 日-family examples

4. Select a model and explore the bilingual semantic space!

Expected Results:
================
- Characters with similar meanings cluster together
- Time-related characters (早,晚,昨,期) form neighborhoods  
- Brightness characters (明,昭,晶) group separately
- Cross-lingual semantic relationships preserved
- Potential Fibonacci/spiral patterns in semantic evolution

This could reveal universal patterns in how humans organize meaning
across languages - exactly what your affinity theory predicts!
"""