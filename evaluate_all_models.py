#!/usr/bin/env python3
"""
evaluate_all_models.py
-----------------------------------------
FINAL VERSION (Minimal Plots)

Includes:
 - Safe dataset loading
 - Safe text normalization
 - YouTube retraining
 - Consistent evaluation across all datasets/models
 - ROC Curves (only binary datasets)
 - Minimal summary plots (F1 + Accuracy)
 - Clean groupby summary
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

# Project functions
from src.feature_engineering import build_vectorizer
from src.train_model import get_model
from src.utils import save_joblib


# --------------------------------------
# CONFIG
# --------------------------------------
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 150

DATASETS = ["twitter", "reddit", "youtube"]
MODELS = ["lr", "svm", "lgbm", "xgb"]


# ======================================================================
# 1. DATA LOADING + CLEANING
# ======================================================================
def load_and_clean_dataset(dataset):
    """Load dataset in a robust + uniform way."""
    base = Path("data") / dataset

    if dataset == "youtube":
        csv_path = base / "youtube_balanced.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            json_path = base / "News_Category_Dataset_v3.json"
            try:
                df = pd.read_json(json_path, lines=True)
            except:
                df = pd.read_json(json_path)

            if "headline" in df.columns:
                df.rename(columns={"headline": "text"}, inplace=True)
            if "category" in df.columns:
                df.rename(columns={"category": "label"}, inplace=True)

    elif dataset == "twitter":
        df = pd.read_csv(base / "train_E6oV3lV.csv")
        df.rename(columns={"tweet": "text"}, inplace=True)
        if "class" in df.columns:
            df.rename(columns={"class": "label"}, inplace=True)

    elif dataset == "reddit":
        df = pd.read_csv(base / "labeled_data.csv")
        df.rename(columns={"tweet": "text"}, inplace=True)
        if "class" in df.columns:
            df.rename(columns={"class": "label"}, inplace=True)

    else:
        raise ValueError("Unknown dataset")

    # --- CLEANING ---
    df["text"] = df["text"].astype(str).fillna("")
    df = df[df["text"].str.strip() != ""]

    if not np.issubdtype(df["label"].dtype, np.number):
        df["label"] = df["label"].astype("category").cat.codes

    df["label"] = df["label"].astype(int)

    return df


# ======================================================================
# 2. SAFE TEXT NORMALIZER
# ======================================================================
def safe_normalize_corpus(texts):
    """Lowercase, remove punctuation, extra spaces."""
    import re

    cleaned = []
    for t in texts:
        t = str(t)
        t = t.lower()
        t = re.sub(r"[^\w\s]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        cleaned.append(t)

    return cleaned


# ======================================================================
# 3. RETRAIN YOUTUBE MODELS
# ======================================================================
def retrain_youtube():
    print("\n🔄 RETRAINING YOUTUBE DATASET...")

    try:
        df = load_and_clean_dataset("youtube")
        df["text"] = safe_normalize_corpus(df["text"])

        df = df[df["text"].str.strip() != ""]
        texts = df["text"].tolist()
        labels = df["label"].to_numpy()

        stratify = labels if len(np.unique(labels)) > 1 else None

        X_train_t, X_test_t, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=stratify
        )

        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(X_train_t)
        X_test = vectorizer.transform(X_test_t)

        model_dir = Path("models/youtube")
        model_dir.mkdir(parents=True, exist_ok=True)

        save_joblib(vectorizer, model_dir / "vectorizer.joblib")
        save_joblib({"X_test_texts": X_test_t, "y_test": y_test}, model_dir / "test_data.joblib")

        for m in MODELS:
            print(f"🤖 Training {m.upper()} ...")
            model = get_model(m, random_state=42)
            model.fit(X_train, y_train)
            save_joblib(model, model_dir / f"{m}.joblib")

        print("🎉 YouTube retraining done.")
        return True

    except Exception as e:
        print("❌ Retraining error:", e)
        return False


# ======================================================================
# 4. PLOT ROC CURVES
# ======================================================================
def plot_roc_curves(dataset, roc_dict):
    """Plot ROC curves for a single dataset ONLY if binary classification."""
    plot_dir = Path("results/overall_summary/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not roc_dict:
        print(f"⚠ No ROC data for {dataset}. Skipping.")
        return

    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    for model_name, (y_test, scores) in roc_dict.items():
        scores = np.asarray(scores).ravel()
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})", lw=2)

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"ROC Curve – {dataset.upper()}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    out = plot_dir / f"ROC_{dataset}.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f" Saved ROC curve: {out}")


# ======================================================================
# 5. EVALUATION
# ======================================================================
def evaluate_with_consistent_data():
    all_results = []

    for dataset in DATASETS:
        print("\n===========================================")
        print(f"Evaluating dataset: {dataset.upper()}")
        print("===========================================\n")

        results = []
        roc_dict = {}

        for model_key in MODELS:
            mpath = Path(f"models/{dataset}/{model_key}.joblib")
            tpath = Path(f"models/{dataset}/test_data.joblib")
            vpath = Path(f"models/{dataset}/vectorizer.joblib")

            if not mpath.exists() or not tpath.exists() or not vpath.exists():
                print(f"⚠ Missing model/test/vectorizer for {dataset}/{model_key}")
                continue

            model = joblib.load(mpath)
            test_data = joblib.load(tpath)
            vectorizer = joblib.load(vpath)

            X_test_texts = [str(t) for t in test_data["X_test_texts"]]
            y_test = np.array(test_data["y_test"])

            X_test = vectorizer.transform(X_test_texts)
            y_pred = model.predict(X_test)

            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except:
                    pass

            y_score = None
            if y_proba is None and hasattr(model, "decision_function"):
                try:
                    y_score = model.decision_function(X_test)
                except:
                    pass

            # ROC (only if binary)
            if len(np.unique(y_test)) == 2:
                if y_proba is not None:
                    roc_dict[model_key.upper()] = (y_test, y_proba[:, 1])
                elif y_score is not None:
                    roc_dict[model_key.upper()] = (y_test, np.asarray(y_score).ravel())

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

            roc_auc = 0.0
            try:
                if y_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                elif y_score is not None:
                    roc_auc = roc_auc_score(y_test, y_score)
            except:
                roc_auc = 0.0

            results.append({
                "dataset": dataset,
                "model": model_key.upper(),
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
                "roc_auc": roc_auc
            })

        # Save ROC
        plot_roc_curves(dataset, roc_dict)

        all_results.extend(results)

    return all_results


# ======================================================================
# 6. MINIMAL SUMMARY PLOTS
# ======================================================================
def create_minimal_plots(df):
    plot_dir = Path("results/overall_summary/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -- F1 Barplot --
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x="dataset", y="f1_macro", hue="model")
    plt.title("F1 Macro Across Datasets")
    plt.tight_layout()
    plt.savefig(plot_dir / "f1_summary.png", dpi=300)
    plt.close()

    # -- Accuracy Barplot --
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x="dataset", y="accuracy", hue="model")
    plt.title("Accuracy Across Datasets")
    plt.tight_layout()
    plt.savefig(plot_dir / "accuracy_summary.png", dpi=300)
    plt.close()


# ======================================================================
# 7. SUMMARY TEXT
# ======================================================================
def print_summary(df):
    print("\n============================")
    print(" FINAL PERFORMANCE SUMMARY")
    print("============================")

    print("\n🏆 Best Model Per Dataset:")
    best = df.loc[df.groupby("dataset")["f1_macro"].idxmax()]
    print(best[["dataset", "model", "f1_macro", "accuracy", "roc_auc"]])

    print("\n📈 Average Performance by Model:")
    numeric_cols = ["accuracy", "f1_macro", "roc_auc"]
    print(df.groupby("model")[numeric_cols].mean())


# ======================================================================
# MAIN
# ======================================================================
def main():
    outdir = Path("results/overall_summary")
    outdir.mkdir(parents=True, exist_ok=True)

    retrain_youtube()
    results = evaluate_with_consistent_data()

    df = pd.DataFrame(results)
    df.to_csv(outdir / "ALL_MODELS_SUMMARY.csv", index=False)

    create_minimal_plots(df)
    print_summary(df)

    print("\nDONE! Minimal plots + ROC curves generated successfully.")


if __name__ == "__main__":
    main()

