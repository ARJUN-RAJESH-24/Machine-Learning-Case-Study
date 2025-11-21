#!/usr/bin/env python3
"""
tune_parameter.py – Fully automatic, CPU-only hyperparameter tuning.
NO ARGUMENTS NEEDED. Just run:

    python tune_parameter.py

It will:
    ✓ Loop over all datasets (twitter, reddit, youtube, adult)
    ✓ Loop over all models (lr, svm, lgbm, xgb)
    ✓ Filter rare classes
    ✓ Downsample massive datasets
    ✓ Normalize text safely
    ✓ Train using sparse TF-IDF
    ✓ Perform GridSearchCV
    ✓ Save tuned models + JSON + CSV + confusion matrix

"""

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
# Datasets + Models
# ============================================================
ALL_DATASETS = ["twitter", "reddit", "youtube"]
ALL_MODELS = ["lr", "svm", "lgbm", "xgb"]

# Parameter grids (kept small for fast CPU tuning)
PARAM_GRIDS = {
    "lr": {"C": [0.1, 1.0, 3.0]},
    "svm": {"C": [0.1, 1.0, 3.0]},
    "lgbm": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31],
    },
    "xgb": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
    },
}

# ============================================================
# DATA LOADER
# ============================================================
def load_dataset(name):
    base = Path("data") / name

    if name == "youtube":
        csv_path = base / "youtube_balanced.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_json(base / "News_Category_Dataset_v3.json", lines=True)
            df = df.rename(columns={"headline": "text", "category": "label"})


    elif name == "twitter":
        df = pd.read_csv(base / "train_E6oV3lV.csv")
        if "tweet" in df.columns:
            df = df.rename(columns={"tweet": "text"})
        if "label" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "label"})

    elif name == "reddit":
        df = pd.read_csv(base / "labeled_data.csv")
        # Reddit dataset has 'tweet' and 'class' columns
        if "tweet" in df.columns:
            df = df.rename(columns={"tweet": "text"})
        if "class" in df.columns:
            df = df.rename(columns={"class": "label"})

    else:
        raise ValueError("Unsupported dataset")

    # Drop NA AFTER renaming columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset {name} missing 'text' or 'label' columns after renaming. Columns: {list(df.columns)}")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    if not np.issubdtype(df["label"].dtype, np.number):
        df["label"] = df["label"].astype("category").cat.codes

    return df


# ============================================================
# TUNING FUNCTION
# ============================================================
def tune_one(dataset, model_key, seed=42, n_jobs=1):
    print(f"\n==============================")
    print(f"🔧 TUNING {model_key.upper()} ON {dataset.upper()}")
    print(f"==============================")

    set_global_seed(seed)
    paths = Paths(dataset)
    ensure_dirs()

    df = load_dataset(dataset)

    # Clean text
    df["text"] = normalize_corpus(df["text"])
    df["text"].replace({None: "", np.nan: ""}, inplace=True)

    # Remove rare labels
    vc = df["label"].value_counts()
    valid = vc[vc > 1].index
    df = df[df["label"].isin(valid)]

    if len(df) > 30000:
        df = df.sample(30000, random_state=seed)
        print("⚖️ Downsampled → 30k for speed")

    # Split
    X = df["text"].tolist()
    y = df["label"].to_numpy()

    strat = y if len(np.unique(y)) > 1 else None

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=strat
    )

    X_train_texts = ["" if pd.isna(x) else str(x) for x in X_train_texts]
    X_test_texts  = ["" if pd.isna(x) else str(x) for x in X_test_texts]

    # Vectorizer
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    save_joblib(vectorizer, paths.vectorizer_path)

    # Model
    model = get_model(model_key, random_state=seed)
    grid_params = PARAM_GRIDS[model_key]

    print(f"🔍 Grid: {grid_params}")

    grid = GridSearchCV(
        model,
        grid_params,
        scoring="f1_macro",
        n_jobs=n_jobs,
        cv=3,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    print("🎯 BEST PARAMS:", grid.best_params_)

    # Evaluate
    y_pred = best.predict(X_test)
    y_proba = None
    if hasattr(best, "predict_proba"):
        try:
            y_proba = best.predict_proba(X_test)
        except:
            pass

    # Save
    save_joblib(best, paths.model_path(f"{model_key}_tuned"))

    evaluate_and_save(
        y_test,
        y_pred,
        sorted(np.unique(y)),
        [str(x) for x in sorted(np.unique(y))],
        paths.report_json_path(f"{model_key}_tuned"),
        paths.report_csv_path(f"{model_key}_tuned"),
        paths.confusion_png_path(f"{model_key}_tuned"),
    )

    print(f"✔ DONE: {dataset} — {model_key}_tuned")


# ============================================================
# MAIN LOOP (no arguments)
# ============================================================
def main():
    print("\n🔥 AUTO-TUNING STARTED (NO ARGUMENTS REQUIRED) 🔥")

    for dataset in ALL_DATASETS:
        for model in ALL_MODELS:
            tune_one(dataset, model, seed=42, n_jobs=1)

    print("\n🎉 ALL DATASETS + ALL MODELS TUNED SUCCESSFULLY!")
    print("📁 Check results/performance_reports/ for metrics.")
    print("📁 Check models/<dataset>/ for saved tuned models.")


if __name__ == "__main__":
    main()
