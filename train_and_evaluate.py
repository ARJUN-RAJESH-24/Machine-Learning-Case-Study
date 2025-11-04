# train_and_evaluate.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.preprocess import clean_text
from src.feature_engineering import build_tfidf, save_vectorizer
from src.train_model import get_models, save_model
from src.evaluate_model import evaluate_metrics, plot_confusion_matrix, plot_roc_curve

DATASETS = {
    "twitter": "data/twitter_hate.csv",
    "reddit": "data/reddit_toxic.csv",
    "youtube": "data/youtube_comments.csv",
    "adult": "data/adult_content.csv"
}

RESULTS_DIR = "results"
MODELS_DIR = "models"
VECT_DIR = os.path.join(MODELS_DIR, "vectorizers")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VECT_DIR, exist_ok=True)

def _ensure_columns(df):
    # try to detect text and label columns if not named 'text' / 'label'
    cols = df.columns.str.lower().tolist()
    text_col = None
    label_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("text", "tweet", "comment", "content", "body"):
            text_col = c
        if lc in ("label", "class", "target", "hate", "sentiment", "toxic"):
            label_col = c
    if text_col is None:
        # fallback to first string column
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    if label_col is None:
        # fallback to last numeric column
        for c in df.columns[::-1]:
            if np.issubdtype(df[c].dtype, np.number):
                label_col = c
                break
    if text_col is None or label_col is None:
        raise ValueError("Could not detect text and label columns. Please ensure dataset has text and label columns.")
    return text_col, label_col

def prepare_dataframe(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(how='any')
    text_col, label_col = _ensure_columns(df)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    # ensure binary labels 0/1
    if df['label'].dtype == object:
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        # try to map common label values
        mapping = {}
        unique = df['label'].unique().tolist()
        if set(unique) <= {"hate", "offensive", "abusive", "toxic", "1", "0", "adult", "porn"}:
            # map common positive tokens to 1
            for u in unique:
                if any(k in u for k in ["hate", "offensive", "abusive", "toxic", "adult", "porn", "1", "true", "yes"]):
                    mapping[u] = 1
                else:
                    mapping[u] = 0
            df['label'] = df['label'].map(mapping)
        else:
            # last resort: label the first unique as 0, others as 1
            vals = sorted(unique)
            mapr = {vals[0]: 0}
            for v in vals[1:]:
                mapr[v] = 1
            df['label'] = df['label'].map(mapr)
    else:
        # numeric: try to convert non-binary to binary by thresholding unique values
        unique = np.unique(df['label'])
        if set(unique) <= {0,1}:
            df['label'] = df['label'].astype(int)
        else:
            # map min -> 0, others ->1
            mn = unique.min()
            df['label'] = (df['label'] != mn).astype(int)
    return df

def run_cv_on_dataset(key, path, n_splits=5, random_state=42):
    print(f"\n=== Running CV on dataset: {key} (file: {path}) ===")
    df = prepare_dataframe(path)
    df['cleaned'] = df['text'].apply(clean_text)
    texts = df['cleaned'].values
    labels = df['label'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"  Fold {fold}/{n_splits}  —  train: {len(train_idx)}  test: {len(test_idx)}")
        X_train_text = texts[train_idx]
        X_test_text = texts[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        X_train_vec, X_test_vec, vect = build_tfidf(X_train_text, X_test_text, max_features=10000)
        vect_path = os.path.join(VECT_DIR, f"{key}_fold{fold}_vect.joblib")
        save_vectorizer(vect, vect_path)

        models = get_models(random_state=random_state + fold)
        for mname, model in models.items():
            print(f"    Training {mname} ...")
            model.fit(X_train_vec, y_train)
            model_path = os.path.join(MODELS_DIR, f"{key}_fold{fold}_{mname}.joblib")
            save_model(model, model_path)

            metrics = evaluate_metrics(model, X_test_vec, y_test)
            metrics.update({"dataset": key, "fold": fold, "model": mname})
            fold_results.append(metrics)

            # save confusion matrix and ROC plot
            try:
                cm_path = os.path.join(RESULTS_DIR, "confusion_matrices", f"cm_{key}_fold{fold}_{mname}.png")
                plot_confusion_matrix(model, X_test_vec, y_test, filepath=cm_path)
            except Exception as e:
                print("Warning: failed to plot/save confusion matrix:", e)

            try:
                roc_path = os.path.join(RESULTS_DIR, "plots", f"roc_{key}_fold{fold}_{mname}.png")
                plot_roc_curve(model, X_test_vec, y_test, filepath=roc_path)
            except Exception as e:
                print("Warning: failed to plot/save ROC curve:", e)

    df_res = pd.DataFrame(fold_results)
    out_csv = os.path.join(RESULTS_DIR, f"{key}_cv_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"Saved CV results to {out_csv}")
    return df_res

def train_all_datasets(n_splits=5):
    all_dfs = []
    for key, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"Dataset file not found: {path}. Skipping {key}.")
            continue
        res = run_cv_on_dataset(key, path, n_splits=n_splits)
        all_dfs.append(res)
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(os.path.join(RESULTS_DIR, "all_datasets_cv_results.csv"), index=False)
        print("Saved combined results.")

def train_single_dataset(dataset_key, n_splits=5):
    path = DATASETS.get(dataset_key)
    if path is None or not os.path.exists(path):
        raise ValueError(f"Dataset key not found or file missing: {dataset_key} -> {path}")
    run_cv_on_dataset(dataset_key, path, n_splits=n_splits)
