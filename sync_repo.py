#!/usr/bin/env python3
"""
Repository Sync Utility

A reusable script to sync files between development and release repositories.
Compares files using git-style diff (ignoring whitespace/comments) and copies
only changed files with a configurable suffix tag.

Usage:
    python sync_repo.py --dry-run  # Preview changes
    python sync_repo.py            # Execute sync
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import click
import tempfile
from datetime import datetime
from typing import List, Set, Tuple


def run_git_diff(file1: str, file2: str) -> bool:
    """
    Compare two files using git diff with whitespace/comment ignoring options.
    Returns True if files are different, False if identical.
    """
    try:
        # Use git diff with options to ignore whitespace and be more lenient
        # -w: ignore whitespace changes
        # --ignore-blank-lines: ignore blank line changes
        result = subprocess.run([
            'git', 'diff', '--no-index', '-w', '--ignore-blank-lines',
            file1, file2
        ], capture_output=True, text=True, cwd=tempfile.gettempdir())

        # git diff returns 0 if files are identical, 1 if different, 2+ on error
        if result.returncode == 0:
            return False  # Files are identical
        elif result.returncode == 1:
            return True   # Files are different
        else:
            # Error occurred, fall back to basic comparison
            click.echo(f"Warning: git diff failed for {file1} vs {file2}, using basic comparison")
            return not files_identical_basic(file1, file2)

    except FileNotFoundError:
        click.echo("Warning: git not found, using basic file comparison")
        return not files_identical_basic(file1, file2)


def files_identical_basic(file1: str, file2: str) -> bool:
    """Basic file comparison fallback."""
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            return f1.read() == f2.read()
    except (FileNotFoundError, UnicodeDecodeError):
        return False


def should_exclude_folder(folder_path: str, exclude_folders: List[str]) -> bool:
    """Check if folder should be excluded based on exclude list."""
    folder_name = os.path.basename(folder_path)
    for exclude in exclude_folders:
        if exclude in folder_path or folder_name == exclude:
            return True
    return False


def get_matching_files(source_dir: str, include_extensions: List[str], exclude_folders: List[str]) -> List[str]:
    """Get all files matching include extensions, excluding specified folders."""
    matching_files = []
    source_path = Path(source_dir)

    for root, dirs, files in os.walk(source_path):
        # Remove excluded directories from dirs list to prevent os.walk from entering them
        dirs[:] = [d for d in dirs if not should_exclude_folder(os.path.join(root, d), exclude_folders)]

        # Check if current directory should be excluded
        if should_exclude_folder(root, exclude_folders):
            continue

        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix

            if file_ext in include_extensions:
                # Get relative path from source directory
                rel_path = os.path.relpath(file_path, source_dir)
                matching_files.append(rel_path)

    return sorted(matching_files)


def create_target_path_with_suffix(rel_path: str, target_dir: str, suffix: str) -> str:
    """Create target file path with suffix before extension."""
    target_path = Path(target_dir) / rel_path

    # Insert suffix before file extension
    name_parts = target_path.stem, suffix, target_path.suffix
    new_name = ''.join(name_parts)

    return str(target_path.parent / new_name)


def ensure_directory_exists(file_path: str, dry_run: bool = False):
    """Ensure directory structure exists for the target file."""
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        if dry_run:
            click.echo(f"# mkdir -p '{dir_path}'")
        else:
            os.makedirs(dir_path, exist_ok=True)
            click.echo(f"Created directory: {dir_path}")


@click.command()
@click.option('--source-repo', default='/home/papagame/projects/digital-duck/st_semantics/src',
              help='Source repository path (dev repo)')
@click.option('--target-repo', default='/home/papagame/projects/digital-duck/semantics-explorer/src',
              help='Target repository path (release repo)')
@click.option('--tag', default='-DEV',
              help='Suffix tag to add to copied files')
@click.option('--include-files', default='.py .yml .json',
              help='Space-delimited file extensions to include')
@click.option('--exclude-folders', default='__pycache__ .git tmp',
              help='Space-delimited folders to exclude')
@click.option('--dry-run', is_flag=True,
              help='Preview changes without executing them')
@click.option('--cleanup', is_flag=True,
              help='Remove tagged files from target repo after manual merge')
def sync_repos(source_repo: str, target_repo: str, tag: str, include_files: str,
               exclude_folders: str, dry_run: bool, cleanup: bool):
    """
    Sync changed files from source (dev) repo to target (release) repo with suffix tag.

    Uses git diff to intelligently compare files, ignoring whitespace and comments.
    Only copies files that have meaningful changes.
    """

    # Parse space-delimited parameters
    include_extensions = include_files.split()
    exclude_folder_list = exclude_folders.split()

    # Validate source directory
    if not os.path.exists(source_repo):
        click.echo(f"Error: Source repository not found: {source_repo}", err=True)
        sys.exit(1)

    # Validate target directory
    if not os.path.exists(target_repo):
        click.echo(f"Error: Target repository not found: {target_repo}", err=True)
        sys.exit(1)

    click.echo(f"{'='*60}")
    click.echo(f"Repository Sync Utility")
    click.echo(f"{'='*60}")
    click.echo(f"Source: {source_repo}")
    click.echo(f"Target: {target_repo}")
    click.echo(f"Tag: {tag}")
    click.echo(f"Include: {include_extensions}")
    click.echo(f"Exclude: {exclude_folder_list}")
    click.echo(f"Mode: {'CLEANUP' if cleanup else 'DRY RUN' if dry_run else 'EXECUTE'}")
    click.echo(f"{'='*60}")

    # Handle cleanup mode
    if cleanup:
        cleanup_tagged_files(target_repo, tag)
        return

    # Get all matching files from source
    matching_files = get_matching_files(source_repo, include_extensions, exclude_folder_list)
    click.echo(f"Found {len(matching_files)} files matching criteria")

    copied_files = []
    skipped_files = []
    new_files = []

    if dry_run:
        click.echo(f"\n# Dry run - Git bash commands that would be executed:")
        click.echo(f"# Remove comments (#) and run commands to execute sync\n")

    for rel_path in matching_files:
        source_file = os.path.join(source_repo, rel_path)
        target_file = os.path.join(target_repo, rel_path)
        target_dev_file = create_target_path_with_suffix(rel_path, target_repo, tag)

        # Check if original file exists in target repo
        if not os.path.exists(target_file):
            # New file - copy with suffix
            ensure_directory_exists(target_dev_file, dry_run)

            if dry_run:
                click.echo(f"# New file: {rel_path}")
                click.echo(f"cp '{source_file}' '{target_dev_file}'")
            else:
                shutil.copy2(source_file, target_dev_file)
                click.echo(f"NEW: {rel_path} -> {os.path.basename(target_dev_file)}")

            new_files.append(rel_path)

        else:
            # Compare files using git diff
            files_different = run_git_diff(source_file, target_file)

            if files_different:
                # Files are different - copy with suffix
                ensure_directory_exists(target_dev_file, dry_run)

                if dry_run:
                    click.echo(f"# Changed file: {rel_path}")
                    click.echo(f"cp '{source_file}' '{target_dev_file}'")
                else:
                    shutil.copy2(source_file, target_dev_file)
                    click.echo(f"COPY: {rel_path} -> {os.path.basename(target_dev_file)}")

                copied_files.append(rel_path)
            else:
                # Files are identical - skip
                if dry_run:
                    click.echo(f"# Identical: {rel_path} (skipping)")

                skipped_files.append(rel_path)

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo(f"SUMMARY")
    click.echo(f"{'='*60}")
    click.echo(f"New files: {len(new_files)}")
    click.echo(f"Changed files: {len(copied_files)}")
    click.echo(f"Identical files (skipped): {len(skipped_files)}")
    click.echo(f"Total processed: {len(matching_files)}")

    if new_files:
        click.echo(f"\nNew files:")
        for f in new_files:
            click.echo(f"  + {f}")

    if copied_files:
        click.echo(f"\nChanged files:")
        for f in copied_files:
            click.echo(f"  ~ {f}")

    if dry_run:
        click.echo(f"\nTo execute the sync, run without --dry-run flag:")
        click.echo(f"python {__file__} --source-repo '{source_repo}' --target-repo '{target_repo}'")

        # Generate timestamped shell script with copy commands
        if copied_files or new_files:
            generate_timestamped_shell_script(copied_files, new_files, source_repo, target_repo, tag)


def cleanup_tagged_files(target_repo: str, tag: str):
    """Remove all tagged files from the target repository."""

    click.echo(f"Searching for files with tag '{tag}' in: {target_repo}")

    tagged_files = []
    for root, dirs, files in os.walk(target_repo):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_folder(os.path.join(root, d), ['__pycache__', '.git', 'tmp'])]

        for file in files:
            if tag in file and not file.endswith('.pyc'):
                file_path = os.path.join(root, file)
                tagged_files.append(file_path)

    if not tagged_files:
        click.echo(f"No tagged files found with suffix '{tag}'")
        return

    click.echo(f"Found {len(tagged_files)} tagged files:")
    for file_path in tagged_files:
        rel_path = os.path.relpath(file_path, target_repo)
        click.echo(f"  - {rel_path}")

    if click.confirm(f"\nRemove all {len(tagged_files)} tagged files?"):
        removed_count = 0
        for file_path in tagged_files:
            try:
                os.remove(file_path)
                rel_path = os.path.relpath(file_path, target_repo)
                click.echo(f"  ✓ Removed: {rel_path}")
                removed_count += 1
            except OSError as e:
                rel_path = os.path.relpath(file_path, target_repo)
                click.echo(f"  ✗ Failed to remove: {rel_path} - {e}")

        click.echo(f"\n✅ Cleanup completed: {removed_count}/{len(tagged_files)} files removed")
    else:
        click.echo("Cleanup cancelled")


def generate_timestamped_shell_script(copied_files: List[str], new_files: List[str],
                                    source_repo: str, target_repo: str, tag: str):
    """Generate a timestamped shell script with copy commands for manual execution."""

    # Generate timestamp in format YYYYMMDD-HH
    timestamp = datetime.now().strftime("%Y%m%d-%H")
    script_name = f"sync_repo_dry_run-{timestamp}.sh"

    all_changed_files = new_files + copied_files

    if not all_changed_files:
        return

    script_content = f"""#!/bin/bash
# Generated sync commands from sync_repo.py dry-run
# Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Execute this script to copy changed files from dev repo to release repo
# with {tag} suffix for manual comparison and merging
#
# Usage: bash {script_name}
# Or comment out individual lines to copy specific files only

echo "Syncing changed files from dev repo to release repo with {tag} suffix..."
echo "Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
echo "============================================================"

# Files detected by git diff analysis ({len(all_changed_files)} files):
"""

    for rel_path in all_changed_files:
        source_file = os.path.join(source_repo, rel_path)
        target_dev_file = create_target_path_with_suffix(rel_path, target_repo, tag)

        file_type = "New file" if rel_path in new_files else "Changed file"
        script_content += f"# {file_type}: {rel_path}\n"
        script_content += f"cp '{source_file}' '{target_dev_file}'\n"

    script_content += f"""
echo "============================================================"
echo "Sync completed! {len(all_changed_files)} files copied with {tag} suffix."
echo "You can now manually compare and merge changes:"
echo ""
"""

    # Add specific guidance for key files
    key_files = {
        'embedding_viz.py': 'widget key fixes, Chinese processing',
        'plotting.py': '3D visualization fixes',
        'model_manager.py': 'enhanced Chinese character handling',
        'config.py': 'model warnings and compatibility'
    }

    found_key_files = [f for f in all_changed_files if any(key in f for key in key_files.keys())]
    if found_key_files:
        script_content += 'echo "Key files with important improvements:"\n'
        for file_path in found_key_files:
            for key, desc in key_files.items():
                if key in file_path:
                    filename = os.path.basename(file_path)
                    script_content += f'echo "  - {filename.replace(".py", f"{tag}.py")} ({desc})"\n'
                    break

    script_content += f"""echo ""
echo "Use diff tools to compare and merge:"
echo "  diff original_file.py original_file{tag}.py"
echo "  # or your preferred diff/merge tool"
echo ""
echo "============================================================"
echo "CLEANUP: Remove {tag} files after manual merge completion"
echo "============================================================"
echo "# Uncomment the following lines to remove {tag} files after merging:"
echo ""
"""

    # Add cleanup commands (commented out by default)
    for rel_path in all_changed_files:
        target_dev_file = create_target_path_with_suffix(rel_path, target_repo, tag)
        script_content += f"# rm '{target_dev_file}'\n"

    script_content += f"""
echo "# To cleanup all {tag} files at once:"
echo "# find '{target_repo}' -name '*{tag}.*' -type f -delete"
echo ""
echo "# Or use this one-liner to cleanup:"
echo "# sed -n 's/^# rm //p' {script_name} | bash"
"""

    # Write the script file
    with open(script_name, 'w') as f:
        f.write(script_content)

    # Make it executable
    os.chmod(script_name, 0o755)

    click.echo(f"\n📝 Generated timestamped shell script: {script_name}")
    click.echo(f"   Use as checklist for manual merge process")
    click.echo(f"   Execute: bash {script_name}")


if __name__ == '__main__':
    sync_repos()