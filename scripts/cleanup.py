"""
cleanup.py — cleans up all working files after processing is done.

Shows what will be deleted, then asks: confirm or abort.

Usage:
  python scripts/cleanup.py                  # clean everything (interactive)
  python scripts/cleanup.py episode_01       # clean specific episode (interactive)
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


def collect_targets(episode: str = None) -> list:
    """Collects all paths to delete."""
    targets = []

    for root_name in ("input", "output"):
        root = Path(root_name)
        if not root.exists():
            continue

        # Collect files directly in root (legacy flat structure)
        for item in sorted(root.iterdir()):
            if item.name == ".gitkeep":
                continue
            if item.is_file():
                if episode and not item.stem.startswith(episode):
                    continue
                targets.append({
                    "path": item,
                    "size": get_size(item),
                    "description": f"{root_name}/{item.name}",
                })

        # Collect files inside episode subdirectories (new structure)
        for ep_dir in sorted(root.iterdir()):
            if not ep_dir.is_dir():
                continue
            if episode and ep_dir.name != episode:
                continue
            for item in sorted(ep_dir.iterdir()):
                if item.name == ".gitkeep":
                    continue
                targets.append({
                    "path": item,
                    "size": get_size(item),
                    "description": f"{root_name}/{ep_dir.name}/{item.name}",
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


def run_cleanup(episode: str = None):
    targets = collect_targets(episode)

    if not targets:
        print("Nothing to clean up.")
        return

    total_size = sum(t["size"] for t in targets)
    scope = f"episode: {episode}" if episode else "all episodes"

    print(f"\nCleanup scope: {scope}")
    print(f"{'─' * 52}")
    for t in targets:
        print(f"  {format_size(t['size']):>8}  {t['description']}")
    print(f"{'─' * 52}")
    print(f"  Total: {format_size(total_size)} across {len(targets)} items\n")

    # Interactive prompt
    while True:
        answer = input("Delete all listed files? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            break
        elif answer in ("", "n", "no"):
            print("Aborted. Nothing was deleted.")
            return
        else:
            print("Please enter y or n.")

    # Delete
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

    print(f"\n[OK] Deleted {deleted}/{len(targets)} items — {format_size(total_size)} freed.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up input/output/separated files after processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup.py                  # clean all episodes (interactive)
  python scripts/cleanup.py episode_01       # clean specific episode (interactive)
        """,
    )
    parser.add_argument(
        "episode", nargs="?", default=None,
        help="Episode name to clean (e.g. episode_01). Omit to clean all."
    )
    args = parser.parse_args()
    run_cleanup(episode=args.episode)


if __name__ == "__main__":
    main()
