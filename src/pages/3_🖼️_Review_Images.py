import streamlit as st
from pathlib import Path
import os
import glob

from config import (
    check_login
)

# Page config
st.set_page_config(
    page_title="Review Images",
    page_icon="🖼️",
    layout="wide"
)

def search_images(images_dir, pattern=""):
    """Search for images using glob pattern"""
    try:
        # If pattern is empty, return empty list (no search)
        if not pattern.strip():
            return []

        # If pattern doesn't contain glob characters, treat it as "contains" pattern
        if not any(char in pattern for char in ['*', '?', '[', ']']):
            # Split the search phrase and join with * for flexible matching
            search_words = pattern.split()
            if search_words:
                pattern = f"*{'*'.join(search_words)}*"
            else:
                return []

        full_pattern = str(images_dir / f"{pattern}.png")
        matching_files = glob.glob(full_pattern)
        return [Path(f) for f in matching_files]
    except Exception as e:
        st.error(f"Error searching with pattern '{pattern}': {e}")
        return []

def main():
    # Check login status
    check_login()

    st.subheader("🖼️ Review Saved Visualization Charts")

    images_dir = Path("data/images")

    if not images_dir.exists():
        st.info("📁 No images directory found. Generate some visualizations first!")
        st.markdown("Go to **Semantics Explorer** → Generate a plot → Click **Save Image**")
        return

    # Initialize session state
    if 'search_pattern' not in st.session_state:
        st.session_state.search_pattern = ""
    if 'selected_image_files' not in st.session_state:
        st.session_state.selected_image_files = []

    # Main layout with two columns
    col1, col2 = st.columns([2, 4])

    with col1:
        # Search Images widget
        search_pattern = st.text_input(
            "Search Images",
            value=st.session_state.search_pattern,
            placeholder="Enter pattern (e.g., ascii-words, gemma, phate)",
            help="Enter text to find filenames containing that text. Advanced users can use glob patterns: * (any), ? (single char), [abc] (character set)"
        )

        # Update session state if pattern changed
        if search_pattern != st.session_state.search_pattern:
            st.session_state.search_pattern = search_pattern

        # Select All and Clear All buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Select All", help="Select all found images"):
                # Get current search results and select all
                matching_files = search_images(images_dir, st.session_state.search_pattern)
                st.session_state.selected_image_files = [f.name for f in matching_files]
                st.rerun()

        with btn_col2:
            if st.button("Clear All", help="Clear all selections"):
                st.session_state.selected_image_files = []
                st.rerun()

    # Sidebar for display options
    with st.sidebar:
        st.subheader("📐 Display Options")

        layout_option = st.radio(
            "Grid Layout:",
            options=["1 per row", "2 per row", "3 per row"],
            index=0,
            help="Choose how to arrange the selected images"
        )

        show_filename = st.checkbox("Show filenames", value=True)
        show_download = st.checkbox("Show download buttons", value=True)
        show_delete = st.checkbox("Show delete buttons", value=False)

    with col2:
        # Search for matching files
        matching_files = search_images(images_dir, st.session_state.search_pattern)

        # Sort by modification time (newest first)
        if matching_files:
            matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Dynamic expander header based on search results
        if not st.session_state.search_pattern.strip():
            expander_header = "Enter search term to find images"
        elif not matching_files:
            expander_header = f"No matches for '{st.session_state.search_pattern}'"
        else:
            expander_header = f"Found {len(matching_files)} images"

        # Found Images expander
        with st.expander(expander_header, expanded=True):
            if not st.session_state.search_pattern.strip():
                st.info("🔍 Enter a search term above to find images")
                st.markdown("Examples: `ascii-words`, `gemma phate`, `2d method`")
                return

            if not matching_files:
                st.warning(f"No images match: `{st.session_state.search_pattern}`")
                st.markdown("Try a different search term or check your spelling")
                return

            # Create checkboxes for each found image
            for image_file in matching_files:
                filename = image_file.name
                is_selected = filename in st.session_state.selected_image_files

                # Checkbox for each image
                selected = st.checkbox(
                    filename,
                    value=is_selected,
                    key=f"checkbox_{filename}"
                )

                # Update selection state
                if selected and filename not in st.session_state.selected_image_files:
                    st.session_state.selected_image_files.append(filename)
                elif not selected and filename in st.session_state.selected_image_files:
                    st.session_state.selected_image_files.remove(filename)

    # Display selected images in main area
    st.divider()

    if not st.session_state.selected_image_files:
        st.info("👆 Select images from the **Found Images** list above to display them below")
        return

    st.subheader(f"📸 Viewing {len(st.session_state.selected_image_files)} Selected Images")

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

    # Sort selected images to match the order of found images
    # Get the current search results to maintain sort order
    current_matching_files = search_images(images_dir, st.session_state.search_pattern)
    if current_matching_files:
        current_matching_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        # Create ordered list of selected files based on found files order
        found_filenames_order = [f.name for f in current_matching_files]
        # Sort selected files to match the found files order
        ordered_selected_files = []
        for filename in found_filenames_order:
            if filename in st.session_state.selected_image_files:
                ordered_selected_files.append(filename)
    else:
        ordered_selected_files = st.session_state.selected_image_files

    # Display images
    for idx, filename in enumerate(ordered_selected_files):
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
                                    if filename in st.session_state.selected_image_files:
                                        st.session_state.selected_image_files.remove(filename)
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
                                if filename in st.session_state.selected_image_files:
                                    st.session_state.selected_image_files.remove(filename)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting {filename}: {e}")

            except Exception as e:
                st.error(f"Error displaying {filename}: {e}")

            st.divider()

    # Summary at the bottom
    if len(st.session_state.selected_image_files) > 1:
        st.success(f"✅ Comparing {len(st.session_state.selected_image_files)} visualizations side by side")

if __name__ == "__main__":
    main()