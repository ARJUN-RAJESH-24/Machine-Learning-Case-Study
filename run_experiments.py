import argparse
from pathlib import Path
from typing import List
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.feature_engineering import build_vectorizer, to_dense_if_needed
from src.preprocess import normalize_corpus
from src.train_model import get_model
from src.evaluate_model import evaluate_and_save
from src.utils import Paths, ensure_dirs, save_joblib, set_global_seed


# ===============================================
# CONFIG
# ===============================================
MODEL_KEYS = ["lr", "svm", "lgbm", "xgb"]
DATASETS = ["twitter", "reddit", "youtube", "adult"]


# ===============================================
# GPU DETECTION (Linux / WSL Safe)
# ===============================================
def detect_gpu() -> bool:
    """Detect if a CUDA-enabled GPU is available."""
    try:
        import cupy
        gpu_count = cupy.cuda.runtime.getDeviceCount()
        if gpu_count > 0:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            name = props["name"].decode()
            cuda_version = cupy.cuda.runtime.runtimeGetVersion() / 1000
            print(f"⚡ GPU detected: {name} (CUDA {cuda_version:.1f}, Total GPUs: {gpu_count})")
            return True
        else:
            print("⚠️ No CUDA-capable GPU detected.")
    except Exception as e:
        print(f"⚠️ GPU detection failed: {e}")
    return False


# ===============================================
# DATASET LOADING
# ===============================================
def load_dataset(paths: Paths) -> pd.DataFrame:
    dataset_name = paths.dataset_name.lower()
    dataset_dir = paths.dataset_csv.parent

    print(f"📂 Loading dataset: {dataset_name} from {dataset_dir}")

    if dataset_name == "twitter":
        df = pd.read_csv(dataset_dir / "train_E6oV3lV.csv")
        df = df.rename(columns={"tweet": "text", "label": "label"})

    elif dataset_name == "reddit":
        df = pd.read_csv(dataset_dir / "labeled_data.csv")
        if "tweet" in df.columns:
            df = df.rename(columns={"tweet": "text", "class": "label"})
        elif "comment" in df.columns:
            df = df.rename(columns={"comment": "text", "class": "label"})

    elif dataset_name == "youtube":
        path = dataset_dir / "News_Category_Dataset_v3.json"
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        df = df.rename(columns={"headline": "text", "category": "label"})
        df["label"] = df["label"].astype("category").cat.codes

    elif dataset_name == "adult":
        df = pd.read_csv(dataset_dir / "adult_dataset.csv")
        for col in ["content", "sentence", "comment", "body"]:
            if "text" not in df.columns and col in df.columns:
                df = df.rename(columns={col: "text"})
        for col in ["labels", "target", "category", "class"]:
            if "label" not in df.columns and col in df.columns:
                df = df.rename(columns={col: "label"})

    else:
        raise ValueError(f"❌ Unsupported dataset: {dataset_name}")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    print(f"✅ Loaded {len(df)} samples from {dataset_name}")
    return df


# ===============================================
# RUN PER DATASET
# ===============================================
def run_for_dataset(dataset: str, models: List[str], test_size: float = 0.2, seed: int = 42, use_gpu: bool = False) -> None:
    """Train and evaluate all models on a given dataset."""
    set_global_seed(seed)
    paths = Paths(dataset)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(paths)
    texts = df["text"].tolist()
    labels = df["label"].to_numpy()

    # Split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # Preprocess
    X_train_texts = normalize_corpus(X_train_texts)
    X_test_texts = normalize_corpus(X_test_texts)

    # Feature engineering (GPU TF-IDF if available)
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)
    save_joblib(vectorizer, paths.vectorizer_path)

    unique_labels = sorted(np.unique(labels).tolist())
    label_names = [str(l) for l in unique_labels]

    # Train + Evaluate
    for mk in models:
        print(f"\n⚙️ Training model '{mk}' on {dataset} ({'GPU' if use_gpu else 'CPU'}) ...")
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


# ===============================================
# MAIN ENTRY POINT
# ===============================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate text models on multiple datasets (GPU ready).")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Datasets to process")
    parser.add_argument("--models", nargs="*", default=MODEL_KEYS, help="Models to train")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ensure_dirs()
    use_gpu = detect_gpu()

    print(f"\n🚀 Starting Experiment Suite ({'GPU' if use_gpu else 'CPU'} Mode)")
    for dataset in tqdm(args.datasets, desc="Datasets"):
        try:
            run_for_dataset(dataset, args.models, test_size=args.test_size, seed=args.seed, use_gpu=use_gpu)
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")


if __name__ == "__main__":
    main()
