"""
cleanup.py — cleans up all working files after processing is done.

By default runs in dry-run mode (shows what will be deleted, does nothing).
Pass --confirm to actually delete.

Usage:
  python scripts/cleanup.py                  # dry-run, shows all files to delete
  python scripts/cleanup.py --confirm        # actually deletes
  python scripts/cleanup.py episode_01       # dry-run for specific episode only
  python scripts/cleanup.py episode_01 --confirm
"""

import argparse
import shutil
from pathlib import Path


def get_size(path: Path) -> int:
    """Returns total size in bytes of a file or directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def collect_targets(episode: str = None) -> list[dict]:
    """
    Collects all paths to delete.
    Returns list of dicts: {path, size, description}
    """
    targets = []

    # --- input/ ---
    input_root = Path("input")
    if episode:
        ep_dirs = [input_root / episode] if (input_root / episode).exists() else []
    else:
        ep_dirs = [d for d in input_root.iterdir() if d.is_dir()] if input_root.exists() else []

    for ep_dir in ep_dirs:
        for item in ep_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            targets.append({
                "path": item,
                "size": get_size(item),
                "description": f"input/{ep_dir.name}/{item.name}",
            })

    # --- output/ ---
    output_root = Path("output")
    if episode:
        ep_dirs = [output_root / episode] if (output_root / episode).exists() else []
    else:
        ep_dirs = [d for d in output_root.iterdir() if d.is_dir()] if output_root.exists() else []

    for ep_dir in ep_dirs:
        for item in ep_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            targets.append({
                "path": item,
                "size": get_size(item),
                "description": f"output/{ep_dir.name}/{item.name}",
            })

    # --- separated/ (Demucs temp files) ---
    separated = Path("separated")
    if separated.exists():
        targets.append({
            "path": separated,
            "size": get_size(separated),
            "description": "separated/ (Demucs temp files)",
        })

    return targets


def run_cleanup(episode: str = None, confirm: bool = False):
    targets = collect_targets(episode)

    if not targets:
        print("Nothing to clean up.")
        return

    total_size = sum(t["size"] for t in targets)

    scope = f"episode: {episode}" if episode else "all episodes"
    mode = "DRY RUN" if not confirm else "DELETING"

    print(f"\n[{mode}] Cleanup scope: {scope}")
    print(f"{'─' * 50}")

    for t in targets:
        print(f"  {'would delete' if not confirm else 'deleting':12s}  {format_size(t['size']):>8}  {t['description']}")

    print(f"{'─' * 50}")
    print(f"  Total: {format_size(total_size)} across {len(targets)} items\n")

    if not confirm:
        print("This was a dry run. To actually delete, run with --confirm")
        return

    # Actually delete
    deleted = 0
    for t in targets:
        try:
            if t["path"].is_dir():
                shutil.rmtree(t["path"])
            else:
                t["path"].unlink()
            deleted += 1
        except Exception as e:
            print(f"  [ERROR] Could not delete {t['path']}: {e}")

    print(f"[OK] Deleted {deleted}/{len(targets)} items ({format_size(total_size)} freed)")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up input/output/separated files after processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup.py                   # dry-run, show everything to delete
  python scripts/cleanup.py --confirm         # delete everything
  python scripts/cleanup.py episode_01        # dry-run for episode_01 only
  python scripts/cleanup.py episode_01 --confirm
        """,
    )
    parser.add_argument(
        "episode", nargs="?", default=None,
        help="Episode name to clean (e.g. episode_01). Omit to clean all."
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually delete files (default is dry-run)"
    )
    args = parser.parse_args()

    run_cleanup(episode=args.episode, confirm=args.confirm)


if __name__ == "__main__":
    main()
