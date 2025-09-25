#!/bin/bash
# Generated sync commands from sync_repo.py dry-run
# Execute this script to copy changed files from st_semantics (dev) to semantics-explorer (release)
# with -DEV suffix for manual comparison and merging
#
# Usage: bash sync_repo_dry_run.sh
# Or uncomment individual lines to copy specific files only

echo "Syncing changed files from dev repo to release repo with -DEV suffix..."
echo "============================================================"

# Changed files detected by git diff analysis:
cp '/home/papagame/projects/digital-duck/st_semantics/src/Welcome.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/Welcome-DEV.py'
cp '/home/papagame/projects/digital-duck/st_semantics/src/components/embedding_viz.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/components/embedding_viz-DEV.py'
cp '/home/papagame/projects/digital-duck/st_semantics/src/components/geometric_analysis.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/components/geometric_analysis-DEV.py'
cp '/home/papagame/projects/digital-duck/st_semantics/src/components/plotting.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/components/plotting-DEV.py'
cp '/home/papagame/projects/digital-duck/st_semantics/src/config.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/config-DEV.py'
cp '/home/papagame/projects/digital-duck/st_semantics/src/models/model_manager.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/models/model_manager-DEV.py'
cp '/home/papagame/projects/digital-duck/st_semantics/src/pages/2_🔍_Semantics_Explorer-Dual_View.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/pages/2_🔍_Semantics_Explorer-Dual_View-DEV.py'
cp '/home/papagame/projects/digital-duck/st_semantics/src/pages/9_🌐_Translator.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/pages/9_🌐_Translator-DEV.py'

echo "============================================================"
echo "Sync completed! 8 files copied with -DEV suffix."
echo "You can now manually compare and merge changes:"
echo ""
echo "Key files with v2.8 improvements:"
echo "  - embedding_viz-DEV.py (widget key fixes, Chinese processing)"
echo "  - plotting-DEV.py (3D visualization fixes)"
echo "  - model_manager-DEV.py (enhanced Chinese character handling)"
echo "  - config-DEV.py (model warnings and compatibility)"
echo ""
echo "Use diff tools to compare:"
echo "  diff components/embedding_viz.py components/embedding_viz-DEV.py"
echo "  # or your preferred diff tool"