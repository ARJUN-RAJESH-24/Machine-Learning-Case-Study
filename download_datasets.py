#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

DATA_DIR = Path("data")
KAGGLE_DATASETS = {
    "twitter": "vkrahul/twitter-hate-speech",
    "reddit": "mrmorj/hate-speech-and-offensive-language-dataset",
    "youtube": "rmisra/news-category-dataset",
}

def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def check_kaggle_cli() -> None:
    try:
        _run(["kaggle", "--version"])
        print("✅ Kaggle CLI found.")
    except Exception:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)

    cred1 = Path("~/.kaggle/kaggle.json").expanduser()
    cred2 = Path("~/.config/kaggle/kaggle.json").expanduser()
    if not (cred1.exists() or cred2.exists()):
        print("❌ Kaggle credentials not found. Run: kaggle configure")
        sys.exit(1)
    print("✅ Kaggle authentication verified.")

def download_from_kaggle(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".downloaded"
    if marker.exists():
        print(f"↪️  Skipping {slug} (already downloaded).")
        return
    print(f"\n⬇️  Downloading: {slug} → {dest}")
    try:
        _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"])
        marker.touch()
        print(f"✅ Downloaded & extracted: {slug.split('/')[-1]}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {slug}: {e}")

def download_all_datasets() -> None:
    print("\n=== Unified Dataset Downloader ===\n")
    check_kaggle_cli()
    DATA_DIR.mkdir(exist_ok=True)

    for name, slug in KAGGLE_DATASETS.items():
        download_from_kaggle(slug, DATA_DIR / name)

    print("\n🎉 All downloads done. Check the 'data/' folder.\n")

if __name__ == "__main__":
    download_all_datasets()

