import os
import subprocess
import importlib

DATA_DIR = "data"

DATASETS = {
    "twitter": "vkrahul/twitter-hate-speech",
    "reddit": "mrmorj/hate-speech-and-offensive-language-dataset",
    "youtube": "rmisra/news-category-dataset",
}

HUGGINGFACE_ALTERNATIVES = [
    "abercowsky/autotrain-data-sexual-content-classification",
    "PKU-Alignment/SafeSora-Label",
    "catalyst-sexual-content-dataset",
    "uClarity/NSFW-Text-Classification",
]


def check_kaggle_cli():
    """Ensure Kaggle CLI is installed and authenticated."""
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        print("✅ Kaggle CLI found.")
    except Exception:
        print("❌ Kaggle CLI not found. Please install it via `pip install kaggle`.")
        exit(1)

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle credentials not found. Run: `kaggle configure`")
        exit(1)
    else:
        print("✅ Kaggle authentication verified.")


def download_from_kaggle(slug, dest):
    """Download dataset from Kaggle and unzip."""
    print(f"\n⬇️  Downloading dataset: {slug}")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"], check=True)
        print(f"✅ Successfully downloaded and extracted: {slug.split('/')[-1]}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {slug}: {e}")


def ensure_datasets_module():
    """Ensure Hugging Face `datasets` module is available."""
    try:
        importlib.import_module("datasets")
    except ImportError:
        print("📦 Installing missing module: datasets")
        subprocess.run(["pip", "install", "datasets", "-q"])
    finally:
        global load_dataset
        from datasets import load_dataset


def download_from_huggingface(dest):
    """Try downloading adult dataset from multiple Hugging Face sources."""
    ensure_datasets_module()
    os.makedirs(dest, exist_ok=True)

    for hf_ds in HUGGINGFACE_ALTERNATIVES:
        print(f"\n⬇️  Attempting to download Hugging Face dataset: {hf_ds}")
        try:
            dataset = load_dataset(hf_ds, split="train")
            df = dataset.to_pandas()
            output_path = os.path.join(dest, "adult_dataset.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Successfully downloaded and saved: {hf_ds} → {output_path}")
            return
        except Exception as e:
            print(f"⚠️  Failed to download {hf_ds}: {e}")

    print("❌ All Hugging Face adult dataset downloads failed.")


def main():
    print("\n=== Unified Dataset Downloader ===\n")

    check_kaggle_cli()
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download Kaggle datasets
    for name, slug in DATASETS.items():
        download_from_kaggle(slug, os.path.join(DATA_DIR, name))

    # Download adult dataset from Hugging Face
    download_from_huggingface(os.path.join(DATA_DIR, "adult"))

    print("\n🎉 All downloads completed. Check the 'data/' folder for extracted files.\n")


if __name__ == "__main__":
    main()
