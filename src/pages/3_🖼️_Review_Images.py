import streamlit as st
from pathlib import Path
import os

from config import (
    check_login
)

# Page config
st.set_page_config(
    page_title="Review Images",
    page_icon="🖼️",
    layout="wide"
)

def main():
    # Check login status
    check_login()
    
    st.subheader("🖼️ Review Saved Visualization Charts")
    
    images_dir = Path("data/images")
    
    if not images_dir.exists():
        st.info("📁 No images directory found. Generate some visualizations first!")
        st.markdown("Go to **Semantics Explorer** → Generate a plot → Click **Save Image**")
        return
        
    image_files = list(images_dir.glob("*.png"))
    
    if not image_files:
        st.info("🎨 No images found. Create some visualizations first!")
        st.markdown("Go to **Semantics Explorer** → Generate a plot → Click **Save Image**")
        return
    
    # Sort by modification time (newest first)
    image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Sidebar for selection
    with st.sidebar:
        st.subheader("Select Images")
        
        # Initialize session state first
        if 'selected_images' not in st.session_state:
            st.session_state.selected_images = []
        
        # Select All/None buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", help="Select all available images"):
                st.session_state.selected_images = [f.name for f in image_files]
                st.rerun()
        with col2:
            if st.button("Clear All", help="Clear all selections"):
                st.session_state.selected_images = []
                st.rerun()
        
        st.divider()
        
        # Display info using session state
        st.info(f"📊 **Total**: {len(image_files)} images\n📋 **Selected**: {len(st.session_state.selected_images)} images")
        
        # Layout options
        st.subheader("Display Options")
        
        layout_option = st.radio(
            "Grid Layout:",
            options=["1 per row", "2 per row", "3 per row"],
            index=0,
            help="Choose how to arrange the selected images"
        )
        
        show_filename = st.checkbox("Show filenames", value=True)
        show_download = st.checkbox("Show download buttons", value=True)
        show_delete = st.checkbox("Show delete buttons", value=False)
    
    # Main content area - Image selection
   
    # Multi-select for images (moved from sidebar for better visibility)
    selected_filenames = st.multiselect(
        "Select image(s):",
        options=[f.name for f in image_files],
        default=st.session_state.selected_images,
        help="Multiple selection enabled - filenames are fully visible here",
        key="main_image_selector"
    )
    
    # Update session state whenever selection changes
    if selected_filenames != st.session_state.selected_images:
        st.session_state.selected_images = selected_filenames
    
    if not st.session_state.selected_images:
        st.info("👆 Select images from the list above to display them below")
        return
    
    # Display selected images
    # st.write(f"Displaying {len(selected_filenames)} selected images:")
    
    # Determine columns based on layout option
    if layout_option == "1 per row":
        n_cols = 1
    elif layout_option == "2 per row":
        n_cols = 2
    elif layout_option == "3 per row":
        n_cols = 3
    else:
        n_cols = 1  # Default fallback
    
    # Create columns
    cols = st.columns(n_cols)
    
    # Display images
    for idx, filename in enumerate(st.session_state.selected_images):
        col = cols[idx % n_cols]
        image_path = images_dir / filename
        
        if not image_path.exists():
            continue
            
        with col:
            # Show filename if enabled
            if show_filename:
                st.markdown(f"**{filename}**")
            
            # Display image
            try:
                st.image(
                    str(image_path), 
                    caption=filename if not show_filename else None,
                    use_column_width=True
                )
                
                # Action buttons
                button_cols = []
                if show_download or show_delete:
                    if show_download and show_delete:
                        button_cols = st.columns(2)
                    else:
                        # For single button, don't use columns
                        button_cols = None
                
                # Download button
                if show_download:
                    if button_cols is not None:
                        button_col = button_cols[0]
                        with button_col:
                            # Read file data outside of download_button
                            try:
                                with open(image_path, "rb") as file:
                                    file_data = file.read()
                                st.download_button(
                                    label="⬇️ Download",
                                    data=file_data,
                                    file_name=filename,
                                    mime="image/png",
                                    key=f"download_{filename}_{idx}"
                                )
                            except Exception as e:
                                st.error(f"Error reading file {filename}: {e}")
                    else:
                        # No columns, display button directly
                        try:
                            with open(image_path, "rb") as file:
                                file_data = file.read()
                            st.download_button(
                                label="⬇️ Download",
                                data=file_data,
                                file_name=filename,
                                mime="image/png",
                                key=f"download_{filename}_{idx}"
                            )
                        except Exception as e:
                            st.error(f"Error reading file {filename}: {e}")
                
                # Delete button
                if show_delete:
                    if button_cols is not None:
                        button_col = button_cols[1] if len(button_cols) > 1 else button_cols[0]
                        with button_col:
                            if st.button(
                                "🗑️ Delete", 
                                key=f"delete_{filename}_{idx}",
                                help=f"Delete {filename}",
                                type="secondary"
                            ):
                                try:
                                    image_path.unlink()
                                    st.success(f"Deleted {filename}")
                                    # Remove from selected images
                                    if filename in st.session_state.selected_images:
                                        st.session_state.selected_images.remove(filename)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting {filename}: {e}")
                    else:
                        # No columns, display button directly
                        if st.button(
                            "🗑️ Delete", 
                            key=f"delete_{filename}_{idx}",
                            help=f"Delete {filename}",
                            type="secondary"
                        ):
                            try:
                                image_path.unlink()
                                st.success(f"Deleted {filename}")
                                # Remove from selected images
                                if filename in st.session_state.selected_images:
                                    st.session_state.selected_images.remove(filename)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting {filename}: {e}")
                
            except Exception as e:
                st.error(f"Error displaying {filename}: {e}")
            
            st.divider()
    
    # Summary at the bottom
    if len(st.session_state.selected_images) > 1:
        st.success(f"✅ Comparing {len(st.session_state.selected_images)} visualizations side by side")

if __name__ == "__main__":
    main()