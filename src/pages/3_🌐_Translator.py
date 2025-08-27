import streamlit as st
import os
import deepl
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from typing import Tuple, Optional, Dict

from config import check_login

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Translator",
    page_icon="🌐",
    layout="wide"
)

class DeepLTranslator:
    """DeepL Translation service for Semantics Explorer"""
    
    def __init__(self, auth_key: str):
        try:
            self.translator = deepl.Translator(auth_key)
        except Exception as e:
            raise Exception(f"Failed to initialize DeepL: {e}")

    def get_source_languages(self) -> Dict[str, str]:
        """Get available source languages"""
        try:
            return {lang.code: lang.name for lang in self.translator.get_source_languages()}
        except Exception as e:
            raise Exception(f"Error fetching source languages: {e}")

    def get_target_languages(self) -> Dict[str, str]:
        """Get available target languages"""
        try:
            return {lang.code: lang.name for lang in self.translator.get_target_languages()}
        except Exception as e:
            raise Exception(f"Error fetching target languages: {e}")

    def translate(self, text: str, target_lang: str, source_lang: str = "auto") -> Tuple[str, str]:
        """Translate text and return (translated_text, detected_language)"""
        try:
            if source_lang == "auto":
                # Auto-detect by translating to English first
                temp_result = self.translator.translate_text(text, target_lang="EN-US")
                detected_lang = temp_result.detected_source_lang
            else:
                detected_lang = source_lang
            
            # Perform actual translation
            result = self.translator.translate_text(
                text, 
                target_lang=target_lang,
                source_lang=detected_lang if detected_lang != "auto" else None
            )
            
            return result.text, detected_lang
            
        except Exception as e:
            raise Exception(f"Translation failed: {e}")

def main():
    # Check login status
    check_login()
    
    st.title("🌐 DeepL Translator")
    st.subheader("Professional translation for semantic research")
    
    # Check for DeepL API key
    DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY')
    if not DEEPL_AUTH_KEY:
        st.error("🔑 DeepL API key not found!")
        st.info("Please add your DeepL API key to your .env file:")
        st.code("DEEPL_AUTH_KEY=your_api_key_here", language="bash")
        st.markdown("Get your API key at: https://www.deepl.com/pro-api")
        return
    
    # Initialize translator
    try:
        translator = DeepLTranslator(DEEPL_AUTH_KEY)
        source_languages = translator.get_source_languages()
        target_languages = translator.get_target_languages()
    except Exception as e:
        st.error(f"❌ Error initializing DeepL translator: {e}")
        return
    
    # Language selection
    col_source, col_target = st.columns(2)
    
    with col_source:
        st.subheader("Source Language")
        
        # Source language selection
        source_langs = ['auto'] + list(source_languages.keys())
        source_display_names = ['Auto-detect'] + [source_languages[lang] for lang in source_languages.keys()]
        
        # Default to English if available
        default_source_idx = 0
        if 'EN' in source_languages:
            default_source_idx = source_langs.index('EN')
        
        selected_source_idx = st.selectbox(
            "Select source language:",
            range(len(source_langs)),
            index=default_source_idx,
            format_func=lambda x: source_display_names[x],
            key="source_lang_select"
        )
        source_lang = source_langs[selected_source_idx]
        
    with col_target:
        st.subheader("Target Language")
        
        # Target language selection  
        target_langs = list(target_languages.keys())
        target_display_names = [target_languages[lang] for lang in target_langs]
        
        # Default to Chinese Simplified if available
        default_target_idx = 0
        if 'ZH-HANS' in target_languages:
            default_target_idx = target_langs.index('ZH-HANS')
        elif 'ZH' in target_languages:
            default_target_idx = target_langs.index('ZH')
            
        selected_target_idx = st.selectbox(
            "Select target language:",
            range(len(target_langs)),
            index=default_target_idx,
            format_func=lambda x: target_display_names[x],
            key="target_lang_select"
        )
        target_lang = target_langs[selected_target_idx]
    
    st.divider()
    
    # Translation interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Source Text")
        source_text = st.text_area(
            "Enter text to translate:",
            height=300,
            placeholder="Type or paste your text here...",
            key="source_text_input"
        )
        
        # Quick action buttons
        col_clear, col_example = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear", help="Clear source text"):
                st.session_state.source_text_input = ""
                st.rerun()
        
        with col_example:
            if st.button("📋 Example", help="Load example text"):
                example_text = "The geometry of meaning reveals how concepts are structured in semantic space."
                st.session_state.source_text_input = example_text
                st.rerun()
    
    with col2:
        st.subheader("🔄 Translation")
        
        # Translation button
        if st.button("🚀 Translate", type="primary", use_container_width=True):
            if not source_text.strip():
                st.warning("Please enter text to translate.")
            else:
                try:
                    with st.spinner("Translating..."):
                        translated_text, detected_lang = translator.translate(
                            source_text,
                            target_lang,
                            source_lang
                        )
                        
                        # Store in session state
                        st.session_state.translated_text = translated_text
                        st.session_state.detected_language = detected_lang
                        
                        # Show detection info if auto-detect was used
                        if source_lang == "auto":
                            detected_name = source_languages.get(detected_lang, detected_lang)
                            st.success(f"✅ Detected source language: **{detected_name}**")
                            
                except Exception as e:
                    st.error(f"❌ Translation error: {e}")
        
        # Display translation
        if 'translated_text' in st.session_state:
            translated_text = st.text_area(
                "Translation result (editable):",
                value=st.session_state.translated_text,
                height=250,
                key="translated_text_display"
            )
            
            # Action buttons for translation
            col_copy, col_save, col_use = st.columns(3)
            
            with col_copy:
                if st.button("📋 Copy", help="Copy to clipboard"):
                    st.info("💡 Use Ctrl+C to copy the text above")
            
            with col_save:
                if st.button("💾 Save", help="Save translation pair"):
                    # Create translations directory if it doesn't exist
                    translations_dir = Path("data/translations")
                    translations_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save translation pair
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    filename = translations_dir / f"translation_{timestamp}.txt"
                    
                    content = f"Source ({source_languages.get(st.session_state.get('detected_language', source_lang), source_lang)}):\n{source_text}\n\nTarget ({target_languages[target_lang]}):\n{translated_text}\n"
                    
                    try:
                        filename.write_text(content, encoding='utf-8')
                        st.success(f"✅ Saved: {filename.name}")
                    except Exception as e:
                        st.error(f"❌ Save error: {e}")
            
            with col_use:
                if st.button("📥 Use for Research", help="Add to semantic datasets"):
                    st.info("💡 Copy the translation and use it in the Semantics Explorer!")
        else:
            st.text_area(
                "Translation will appear here:",
                value="",
                height=250,
                disabled=True,
                placeholder="Click 'Translate' to see results..."
            )
    
    st.divider()
    
    # Usage tips
    with st.expander("💡 Usage Tips", expanded=False):
        st.markdown("""
        ### 🎯 For Semantic Research:
        - **Translate concept lists** to explore cross-lingual patterns
        - **Use consistent terminology** for better semantic alignment
        - **Compare translations** to understand cultural differences
        
        ### 🔧 DeepL Features:
        - **High-quality translations** for research purposes
        - **Automatic language detection** for unknown text
        - **Supports 30+ languages** including Chinese, Japanese, Korean
        
        ### 📚 Research Applications:
        - Translate semantic categories (colors, emotions, animals)
        - Create multilingual datasets for embedding analysis  
        - Verify cross-lingual concept alignment
        """)
    
    # Statistics
    if 'translated_text' in st.session_state and 'detected_language' in st.session_state:
        st.sidebar.subheader("📊 Translation Info")
        st.sidebar.info(f"""
        **Source**: {source_languages.get(st.session_state.detected_language, 'Unknown')}  
        **Target**: {target_languages[target_lang]}  
        **Characters**: {len(source_text)} → {len(st.session_state.translated_text)}
        """)

if __name__ == "__main__":
    main()