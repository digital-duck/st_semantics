#!/usr/bin/env python3
"""
Script to clean up remaining emoji character encoding issues
"""

import os

def fix_emoji_encoding():
    """Fix remaining emoji encoding issues in the ECharts page"""
    file_path = '/home/papagame/projects/digital-duck/st_semantics/src/pages/3_📊_Semantics_Explorer-ECharts.py'

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Define replacements for corrupted characters
    replacements = [
        ('=s Voids', '🕳️ Voids'),
        ('9 Geometric analysis', 'ℹ️ Geometric analysis'),
        ('9\x0f About ECharts Features', 'ℹ️ About ECharts Features'),
        ('9 About ECharts Features', 'ℹ️ About ECharts Features'),
        ('< Show Semantic Forces', '🌐 Show Semantic Forces'),
        ('=� **ECharts Visualization auto-saved as**', '📸 **ECharts Visualization auto-saved as**'),
        ('- **=� Advanced Clustering**', '- **📊 Advanced Clustering**'),
        ('- **=� Responsive Design**', '- **📱 Responsive Design**'),
        ('- **< Network Analysis**', '- **🌐 Network Analysis**'),
        ('- **� Performance**', '- **⚡ Performance**'),
        ('🌐 Show Semantic Forces', '🌐 Show Semantic Forces'),
    ]

    # Apply replacements
    changes_made = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changes_made += 1
            print(f"✅ Fixed: '{old}' → '{new}'")
        else:
            print(f"ℹ️ Not found: '{old}'")

    # Write back the file
    if changes_made > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n🎉 Fixed {changes_made} emoji encoding issues!")
    else:
        print("\n✅ No emoji encoding issues found!")

    return changes_made

if __name__ == "__main__":
    print("🧹 Cleaning up emoji encoding issues...")
    fix_emoji_encoding()