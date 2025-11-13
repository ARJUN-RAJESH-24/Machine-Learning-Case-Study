#!/usr/bin/env python3
"""
run_experiments.py
CPU-first training pipeline for datasets in ./data/*
Saves models under models/<dataset>/*.joblib and reports in results/performance_reports/
"""
import argparse
from pathlib import Path
from typing import List
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local imports (assumes src/ is correct)
from src.feature_engineering import build_vectorizer, to_dense_if_needed
from src.preprocess import normalize_corpus
from src.train_model import get_model
from src.evaluate_model import evaluate_and_save
from src.utils import Paths, ensure_dirs, save_joblib, set_global_seed

MODEL_KEYS = ["lr", "svm", "lgbm", "xgb"]
DATASETS = ["twitter", "reddit", "youtube", "adult"]


def detect_gpu() -> bool:
    # We force CPU-first behaviour in this script. This function only prints info.
    try:
        import cupy  # type: ignore
        cnt = cupy.cuda.runtime.getDeviceCount()
        if cnt > 0:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            name = props["name"].decode()
            print(f"⚡ GPU detected: {name} (info only) — running CPU mode by default.")
            return True
    except Exception:
        pass
    print("🧠 Running in CPU mode (scikit-learn).")
    return False


def load_dataset(paths: Paths) -> pd.DataFrame:
    dataset_name = paths.dataset_name.lower()
    base = Path("data") / dataset_name
    print(f"📂 Loading dataset: {dataset_name} from {base}")

    if dataset_name == "twitter":
        file_path = base / "train_E6oV3lV.csv"
        df = pd.read_csv(file_path)
        # many twitter CSVs have column 'tweet' or 'text'; standardize
        if "tweet" in df.columns:
            df = df.rename(columns={"tweet": "text"})
        if "label" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "label"})

    elif dataset_name == "reddit":
        file_path = base / "labeled_data.csv"
        df = pd.read_csv(file_path)
        if "comment" in df.columns:
            df = df.rename(columns={"comment": "text"})
        if "class" in df.columns:
            df = df.rename(columns={"class": "label"})

    elif dataset_name == "youtube":
        csv_path = base / "youtube_balanced.csv"
        json_path = base / "News_Category_Dataset_v3.json"
        if csv_path.exists():
            print("✅ Found balanced youtube CSV. Loading that.")
            df = pd.read_csv(csv_path)
        else:
            print("⚠️ Balanced CSV not found — attempting to load JSON lines.")
            # Some versions are JSON lines
            try:
                df = pd.read_json(json_path, lines=True)
            except ValueError:
                # fallback to non-lines JSON
                df = pd.read_json(json_path)
            # Normalize columns from dataset
            if "headline" in df.columns:
                df = df.rename(columns={"headline": "text"})
            if "category" in df.columns:
                df = df.rename(columns={"category": "label"})
            # map categorical labels to integers
            if not np.issubdtype(df["label"].dtype, np.number):
                df["label"] = df["label"].astype("category").cat.codes

    elif dataset_name == "adult":
        # The original code here was indented incorrectly, making it fall outside the elif block.
        file_path = base / "adult_dataset.csv"
        df = pd.read_csv(file_path)
        
    else:
        # This 'else' correctly handles the case where none of the specific dataset names match.
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Try dataset-specific fixes
    if "prompt_text" in df.columns and "prompt_labels" in df.columns:
        df = df.rename(columns={"prompt_text": "text", "prompt_labels": "label"})
    elif "video_text" in df.columns and "video_labels" in df.columns:
        df = df.rename(columns={"video_text": "text", "video_labels": "label"})
    else:
        # generic fallback
        possible_text_cols = ["text", "content", "sentence", "comment", "body", "review", "message"]
        possible_label_cols = ["label", "labels", "target", "category", "class"]
        text_col = next((c for c in df.columns if c.lower() in possible_text_cols), None)
        label_col = next((c for c in df.columns if c.lower() in possible_label_cols), None)
        if text_col:
            df = df.rename(columns={text_col: "text"})
        if label_col:
            df = df.rename(columns={label_col: "label"})

    # final sanitation
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset {dataset_name} missing required columns 'text' or 'label'. Found: {list(df.columns)}")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    # try convert labels to numeric codes when not numeric
    if not np.issubdtype(df["label"].dtype, np.number):
        df["label"] = df["label"].astype("category").cat.codes
    df["label"] = df["label"].astype(int)

    print(f"✅ Loaded {len(df):,} samples from {dataset_name} (classes: {df['label'].nunique()})")
    return df


def run_for_dataset(dataset: str, models: List[str], test_size: float = 0.2, seed: int = 42, use_gpu: bool = False) -> None:
    set_global_seed(seed)
    paths = Paths(dataset)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(paths)

    # Normalize corpus — normalize_corpus should accept list/Series and return list/Series
    df["text"] = normalize_corpus(df["text"])
    # ensure list type for vectorizers that expect sequences
    texts = list(df["text"])
    labels = df["label"].to_numpy()

    # Split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # Optional double-clean (safe if normalize_corpus returns same type)
    X_train_texts = list(normalize_corpus(X_train_texts))
    X_test_texts = list(normalize_corpus(X_test_texts))

    # Vectorize (build_vectorizer handles CPU fallback)
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)
    save_joblib(vectorizer, paths.vectorizer_path)

    unique_labels = sorted(np.unique(labels).tolist())
    label_names = [str(l) for l in unique_labels]

    for mk in models:
        print(f"\n⚙️ Training model '{mk}' on {dataset} (CPU mode)...")
        model = get_model(mk, random_state=seed)

        Xd_train = to_dense_if_needed(X_train)
        Xd_test = to_dense_if_needed(X_test)

        model.fit(Xd_train, y_train)
        y_pred = model.predict(Xd_test)

        save_joblib(model, paths.model_path(mk))

        evaluate_and_save(
            y_test, y_pred, unique_labels, label_names,
            paths.report_json_path(mk),
            paths.report_csv_path(mk),
            paths.confusion_png_path(mk),
        )
        print(f"✅ Completed model '{mk}' on {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate text models (CPU mode).")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Datasets to process")
    parser.add_argument("--models", nargs="*", default=MODEL_KEYS, help="Models to train")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ensure_dirs()
    # Just informational: don't force GPU usage.
    use_gpu = detect_gpu()

    print(f"\n🚀 Starting Experiment Suite (CPU Mode)")
    for dataset in tqdm(args.datasets, desc="Datasets"):
        try:
            run_for_dataset(dataset, args.models, args.test_size, args.seed, use_gpu)
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")


if __name__ == "__main__":
    main()
