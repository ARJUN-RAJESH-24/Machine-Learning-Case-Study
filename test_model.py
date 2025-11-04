# test_model.py
import os
import pandas as pd
import numpy as np
from src.preprocess import clean_text
from src.feature_engineering import load_vectorizer
from src.train_model import load_model
from src.evaluate_model import evaluate_metrics, plot_confusion_matrix, plot_roc_curve

DATASETS = {
    "twitter": "data/twitter_hate.csv",
    "reddit": "data/reddit_toxic.csv",
    "youtube": "data/youtube_comments.csv",
    "adult": "data/adult_content.csv"
}

MODELS_DIR = "models"
VECT_DIR = os.path.join(MODELS_DIR, "vectorizers")
RESULTS_DIR = "results"

def _prepare_test_df(path, sample_frac=0.2, random_state=42):
    df = pd.read_csv(path, low_memory=False)
    # attempt to detect text/label columns similarly to train pipeline
    # reuse same prepare logic from train_and_evaluate but simplified
    # first drop NA and try common names
    df = df.dropna(how='any')
    cols = df.columns.str.lower().tolist()
    text_col, label_col = None, None
    for c in df.columns:
        lc = c.lower()
        if lc in ("text", "tweet", "comment", "content", "body"):
            text_col = c
        if lc in ("label", "class", "target", "hate", "toxic"):
            label_col = c
    if text_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    if label_col is None:
        for c in df.columns[::-1]:
            if np.issubdtype(df[c].dtype, np.number):
                label_col = c
                break
    if text_col is None or label_col is None:
        raise ValueError("Could not detect text/label columns for testing dataset.")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    # basic label normalization
    if df['label'].dtype == object:
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df['label'] = df['label'].map(lambda x: 1 if any(k in x for k in ["hate","offensive","toxic","adult","porn","1","yes","true"]) else 0)
    else:
        # numeric -> binary if necessary
        unique = df['label'].unique()
        if set(unique) <= {0,1}:
            df['label'] = df['label'].astype(int)
        else:
            mn = min(unique)
            df['label'] = (df['label'] != mn).astype(int)
    # sample a holdout
    if sample_frac < 1.0:
        df_test = df.sample(frac=sample_frac, random_state=random_state)
    else:
        df_test = df
    df_test['cleaned'] = df_test['text'].apply(clean_text)
    return df_test

def test_model_cli(dataset_key, model_name, fold, vectorizer_path=None, model_path=None):
    ds_path = DATASETS.get(dataset_key)
    if ds_path is None or not os.path.exists(ds_path):
        raise ValueError(f"Dataset not found: {dataset_key} -> {ds_path}")

    df_test = _prepare_test_df(ds_path, sample_frac=0.2, random_state=42)
    if vectorizer_path is None:
        vectorizer_path = os.path.join(VECT_DIR, f"{dataset_key}_fold{fold}_vect.joblib")
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f"{dataset_key}_fold{fold}_{model_name}.joblib")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    vect = load_vectorizer(vectorizer_path)
    model = load_model(model_path)

    X_test = vect.transform(df_test['cleaned'].values)
    y_test = df_test['label'].values

    metrics = evaluate_metrics(model, X_test, y_test)
    print("\n=== Test Evaluation ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v}")

    # save plots
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices", f"cm_test_{dataset_key}_fold{fold}_{model_name}.png")
    roc_path = os.path.join(RESULTS_DIR, "plots", f"roc_test_{dataset_key}_fold{fold}_{model_name}.png")
    try:
        plot_confusion_matrix(model, X_test, y_test, filepath=cm_path)
    except Exception as e:
        print("Warning: could not plot confusion matrix:", e)
    try:
        plot_roc_curve(model, X_test, y_test, filepath=roc_path)
    except Exception as e:
        print("Warning: could not plot ROC:", e)

    return metrics
