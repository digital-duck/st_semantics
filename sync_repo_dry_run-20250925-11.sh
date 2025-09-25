#!/bin/bash
# Generated sync commands from sync_repo.py dry-run
# Timestamp: 2025-09-25 11:24:44
# Execute this script to copy changed files from dev repo to release repo
# with -DEV suffix for manual comparison and merging
#
# Usage: bash sync_repo_dry_run-20250925-11.sh
# Or comment out individual lines to copy specific files only

echo "Syncing changed files from dev repo to release repo with -DEV suffix..."
echo "Generated: 2025-09-25 11:24:44"
echo "============================================================"

# Files detected by git diff analysis (8 files):
# Changed file: config.py  (X Done)
cp '/home/papagame/projects/digital-duck/st_semantics/src/config.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/config-DEV.py'


# Changed file: Welcome.py  (X Done)
cp '/home/papagame/projects/digital-duck/st_semantics/src/Welcome.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/Welcome-DEV.py'


# Changed file: components/embedding_viz.py  (X Done)
cp '/home/papagame/projects/digital-duck/st_semantics/src/components/embedding_viz.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/components/embedding_viz-DEV.py'

# Changed file: components/geometric_analysis.py   (X Done)
cp '/home/papagame/projects/digital-duck/st_semantics/src/components/geometric_analysis.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/components/geometric_analysis-DEV.py'


# Changed file: components/plotting.py   (X Done)
cp '/home/papagame/projects/digital-duck/st_semantics/src/components/plotting.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/components/plotting-DEV.py'


# Changed file: models/model_manager.py   (X Done)
cp '/home/papagame/projects/digital-duck/st_semantics/src/models/model_manager.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/models/model_manager-DEV.py'


# Changed file: pages/2_🔍_Semantics_Explorer-Dual_View.py   (X Done)
cp '/home/papagame/projects/digital-duck/st_semantics/src/pages/2_🔍_Semantics_Explorer-Dual_View.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/pages/2_🔍_Semantics_Explorer-Dual_View-DEV.py'

# Changed file: pages/9_🌐_Translator.py   (X Done)
cp '/home/papagame/projects/digital-duck/st_semantics/src/pages/9_🌐_Translator.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/pages/9_🌐_Translator-DEV.py'

echo "============================================================"
echo "Sync completed! 8 files copied with -DEV suffix."
echo "You can now manually compare and merge changes:"
echo ""
echo "Key files with important improvements:"
echo "  - embedding_viz-DEV.py (widget key fixes, Chinese processing)"
echo "  - plotting-DEV.py (3D visualization fixes)"
echo "  - config-DEV.py (model warnings and compatibility)"
echo "  - model_manager-DEV.py (enhanced Chinese character handling)"
echo ""
echo "Use diff tools to compare and merge:"
echo "  diff original_file.py original_file-DEV.py"
echo "  # or your preferred diff/merge tool"
