#!/bin/bash
# Generated sync commands from sync_repo.py dry-run
# Timestamp: 2025-09-25 12:07:28
# Execute this script to copy changed files from dev repo to release repo
# with -DEV suffix for manual comparison and merging
#
# Usage: bash sync_repo_dry_run-20250925-12.sh
# Or comment out individual lines to copy specific files only

echo "Syncing changed files from dev repo to release repo with -DEV suffix..."
echo "Generated: 2025-09-25 12:07:28"
echo "============================================================"

# Files detected by git diff analysis (3 files):
# Changed file: components/embedding_viz.py
cp '/home/papagame/projects/digital-duck/st_semantics/src/components/embedding_viz.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/components/embedding_viz-DEV.py'
# Changed file: config.py
cp '/home/papagame/projects/digital-duck/st_semantics/src/config.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/config-DEV.py'
# Changed file: pages/2_🔍_Semantics_Explorer-Dual_View.py
cp '/home/papagame/projects/digital-duck/st_semantics/src/pages/2_🔍_Semantics_Explorer-Dual_View.py' '/home/papagame/projects/digital-duck/semantics-explorer/src/pages/2_🔍_Semantics_Explorer-Dual_View-DEV.py'

echo "============================================================"
echo "Sync completed! 3 files copied with -DEV suffix."
echo "You can now manually compare and merge changes:"
echo ""
echo "Key files with important improvements:"
echo "  - embedding_viz-DEV.py (widget key fixes, Chinese processing)"
echo "  - config-DEV.py (model warnings and compatibility)"
echo ""
echo "Use diff tools to compare and merge:"
echo "  diff original_file.py original_file-DEV.py"
echo "  # or your preferred diff/merge tool"
echo ""
echo "============================================================"
echo "CLEANUP: Remove -DEV files after manual merge completion"
echo "============================================================"
echo "# Uncomment the following lines to remove -DEV files after merging:"
echo ""
# rm '/home/papagame/projects/digital-duck/semantics-explorer/src/components/embedding_viz-DEV.py'
# rm '/home/papagame/projects/digital-duck/semantics-explorer/src/config-DEV.py'
# rm '/home/papagame/projects/digital-duck/semantics-explorer/src/pages/2_🔍_Semantics_Explorer-Dual_View-DEV.py'

echo "# To cleanup all -DEV files at once:"
echo "# find '/home/papagame/projects/digital-duck/semantics-explorer/src' -name '*-DEV.*' -type f -delete"
echo ""
echo "# Or use this one-liner to cleanup:"
echo "# sed -n 's/^# rm //p' sync_repo_dry_run-20250925-12.sh | bash"
