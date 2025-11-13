#!/usr/bin/env python3
"""
tune_parameter.py – Safe, CPU-only hyperparameter tuning.
Handles:
  ✓ huge datasets (downsampling)
  ✓ rare class filtering
  ✓ sparse TF-IDF training (no RAM explosions)
  ✓ XGB/LGBM/SVM/LR tuning
Usage:
  python tune_parameter.py --dataset youtube --model xgb
  python tune_parameter.py --dataset youtube --model xgb lgbm
  python tune_parameter.py --dataset adult --model all
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from src.preprocess import normalize_corpus
from src.feature_engineering import build_vectorizer
from src.train_model import get_model
from src.evaluate_model import evaluate_and_save
from src.utils import Paths, ensure_dirs, save_joblib, set_global_seed


# ============================================================
# SAFE PARAMETER GRIDS (kept small for large datasets)
# ============================================================
PARAM_GRIDS = {
    "lr": {"C": [0.1, 1.0, 3.0]},
    "svm": {"C": [0.1, 1.0, 3.0]},
    "lgbm": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31],
        "max_depth": [-1, 8],
    },
    "xgb": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "subsample": [1.0],
    },
}


# ============================================================
# DATA LOADER
# ============================================================
def load_dataset(dataset_name: str) -> pd.DataFrame:
    base = Path("data") / dataset_name

    if dataset_name == "youtube":
        csv_path = base / "youtube_balanced.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_json(base / "News_Category_Dataset_v3.json", lines=True)
            df = df.rename(columns={"headline": "text", "category": "label"})

    elif dataset_name == "adult":
        df = pd.read_csv(base / "adult_dataset.csv")
        # autodetect
        text_col = next((c for c in df.columns if "text" in c.lower()), None)
        label_col = next((c for c in df.columns if "label" in c.lower()), None)
        df = df.rename(columns={text_col: "text", label_col: "label"})

    elif dataset_name == "twitter":
        df = pd.read_csv(base / "train_E6oV3lV.csv")
        df = df.rename(columns={"tweet": "text"})

    elif dataset_name == "reddit":
        df = pd.read_csv(base / "labeled_data.csv")
        if "comment" in df.columns:
            df = df.rename(columns={"comment": "text", "class": "label"})

    else:
        raise ValueError("Unsupported dataset")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    # convert categorical labels
    if not np.issubdtype(df["label"].dtype, np.number):
        df["label"] = df["label"].astype("category").cat.codes

    return df


# ============================================================
# SAFE TUNING PIPELINE
# ============================================================
def tune_model(dataset: str, model_key: str, seed=42, n_jobs=1):
    print(f"\n🔧 Tuning {model_key.upper()} on {dataset} (CPU mode)")

    set_global_seed(seed)
    paths = Paths(dataset)
    ensure_dirs()

    df = load_dataset(dataset)

    # ----------------------------
    # CLEAN & NORMALIZE TEXT
    # ----------------------------
    df["text"] = normalize_corpus(df["text"])
    df["text"].replace({None: "", np.nan: ""}, inplace=True)

    # ----------------------------
    # FAST RARE-CLASS FILTERING
    # ----------------------------
    value_counts = df["label"].value_counts()
    valid_classes = value_counts[value_counts > 1].index
    df = df[df["label"].isin(valid_classes)]

    if len(valid_classes) == 0:
        raise ValueError("Dataset has no class with >=2 samples.")

    # ----------------------------
    # OPTIONAL DOWNSAMPLING
    # ----------------------------
    if len(df) > 25000:
        df = df.sample(25000, random_state=seed)
        print("⚖️ Downsampled to 25,000 for safe GridSearch.")

    texts = df["text"].tolist()
    labels = df["label"].to_numpy()

    # ----------------------------
    # SAFE STRATIFIED SPLIT
    # ----------------------------
    stratify_opt = labels if len(np.unique(labels)) > 1 else None

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_opt,
    )

    # Make sure no NaN strings appear
    X_train_texts = [" " if pd.isna(x) else str(x) for x in X_train_texts]
    X_test_texts = [" " if pd.isna(x) else str(x) for x in X_test_texts]

    # ----------------------------
    # TF-IDF VECTORIZE (sparse)
    # ----------------------------
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    save_joblib(vectorizer, paths.vectorizer_path)

    # ----------------------------
    # GRID SEARCH
    # ----------------------------
    model = get_model(model_key, random_state=seed)
    grid_params = PARAM_GRIDS[model_key]

    print(f"🔍 Grid Params: {grid_params}")

    grid = GridSearchCV(
        model,
        grid_params,
        scoring="f1_macro",
        cv=3,
        verbose=1,
        n_jobs=n_jobs,
    )

    grid.fit(X_train, y_train)

    print("🎯 Best Params:", grid.best_params_)

    best_model = grid.best_estimator_

    # ----------------------------
    # EVALUATION
    # ----------------------------
    y_pred = best_model.predict(X_test)
    y_proba = None

    if hasattr(best_model, "predict_proba"):
        try:
            y_proba = best_model.predict_proba(X_test)
        except Exception:
            pass

    # ----------------------------
    # SAVE EVERYTHING
    # ----------------------------
    save_joblib(best_model, paths.model_path(f"{model_key}_tuned"))

    evaluate_and_save(
        y_test,
        y_pred,
        sorted(np.unique(labels)),
        [str(c) for c in sorted(np.unique(labels))],
        paths.report_json_path(f"{model_key}_tuned"),
        paths.report_csv_path(f"{model_key}_tuned"),
        paths.confusion_png_path(f"{model_key}_tuned"),
    )

    print(f"✅ Finished tuning {model_key.upper()} on {dataset}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", nargs="+", required=True,
                        choices=["lr", "svm", "lgbm", "xgb", "all"])
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    models = ["lr", "svm", "lgbm", "xgb"] if "all" in args.model else args.model

    for mk in models:
        tune_model(args.dataset, mk, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()

