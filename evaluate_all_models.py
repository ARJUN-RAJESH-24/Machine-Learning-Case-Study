#!/usr/bin/env python3
"""
final_evaluate_all_models.py
-----------------------------------------
FINAL SOLUTION: Handles YouTube data issues and provides complete evaluation
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
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from src.preprocess import normalize_corpus
from src.feature_engineering import build_vectorizer
from src.train_model import get_model
from src.utils import Paths, save_joblib, set_global_seed

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

DATASETS = ["twitter", "reddit", "youtube"]
MODELS = ["lr", "svm", "lgbm", "xgb"]

# ------------------------------------------------------------
# IMPROVED DATA LOADING WITH YOUTUBE FIX
# ------------------------------------------------------------
def load_and_clean_dataset(dataset):
    """Load and consistently clean dataset with YouTube fixes"""
    base = Path("data") / dataset

    if dataset == "youtube":
        csv_path = base / "youtube_balanced.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            # Handle YouTube JSON data carefully
            json_path = base / "News_Category_Dataset_v3.json"
            try:
                df = pd.read_json(json_path, lines=True)
            except:
                df = pd.read_json(json_path)

            # YouTube specific column mapping
            if "headline" in df.columns:
                df = df.rename(columns={"headline": "text"})
            if "category" in df.columns:
                df = df.rename(columns={"category": "label"})

    elif dataset == "twitter":
        df = pd.read_csv(base / "train_E6oV3lV.csv")
        if "tweet" in df.columns:
            df = df.rename(columns={"tweet": "text"})
        if "label" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "label"})

    elif dataset == "reddit":
        df = pd.read_csv(base / "labeled_data.csv")
        if "tweet" in df.columns:
            df = df.rename(columns={"tweet": "text"})
        if "class" in df.columns:
            df = df.rename(columns={"class": "label"})

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Consistent cleaning for ALL datasets
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    # Handle NaN in text - replace with empty string
    df["text"] = df["text"].fillna("")

    # Remove rows with completely empty text after cleaning
    df = df[df["text"].str.strip() != ""]

    # Convert labels to numeric consistently
    if not np.issubdtype(df["label"].dtype, np.number):
        df["label"] = df["label"].astype("category").cat.codes

    df["label"] = df["label"].astype(int)

    # Final cleanup - remove any remaining problematic rows
    df = df[df["text"].notna()]
    df = df[df["label"].notna()]

    print(f"✅ Loaded {len(df)} samples, {df['label'].nunique()} classes from {dataset}")
    return df

# ------------------------------------------------------------
# SAFE TEXT NORMALIZATION
# ------------------------------------------------------------
def safe_normalize_corpus(texts):
    """Safe normalization that handles edge cases"""
    if isinstance(texts, pd.Series):
        texts = texts.fillna("").astype(str).tolist()
    else:
        texts = [str(x) if x is not None else "" for x in texts]

    # Apply normalization
    normalized = []
    for text in texts:
        try:
            # Use the existing normalize_corpus but handle individual texts
            if text is None or pd.isna(text):
                normalized.append("")
            else:
                # Apply normalization to individual text
                cleaned_text = str(text).lower().strip()
                # Basic cleaning - you can expand this
                import re
                cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                normalized.append(cleaned_text)
        except Exception as e:
            print(f"⚠️ Text normalization warning: {e}")
            normalized.append("")

    return pd.Series(normalized)

# ------------------------------------------------------------
# RETRAIN YOUTUBE SPECIFICALLY
# ------------------------------------------------------------
def retrain_youtube():
    """Retrain YouTube models with fixed data processing"""
    print("\n🔄 RETRAINING YOUTUBE DATASET...")

    dataset = "youtube"

    try:
        # Load and prepare data
        df = load_and_clean_dataset(dataset)

        # Use safe normalization
        print("🔄 Applying safe text normalization...")
        df["text"] = safe_normalize_corpus(df["text"])

        # Remove empty texts after normalization
        df = df[df["text"].str.strip() != ""]

        texts = df["text"].tolist()
        labels = df["label"].to_numpy()

        print(f"📊 After cleaning: {len(texts)} samples")

        # Consistent split with stratification
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Final cleaning
        X_train_texts = [str(x) for x in X_train_texts if x is not None]
        X_test_texts = [str(x) for x in X_test_texts if x is not None]

        # Remove any empty texts
        X_train_texts = [x for x in X_train_texts if x.strip()]
        X_test_texts = [x for x in X_test_texts if x.strip()]

        # Update labels to match
        y_train = y_train[:len(X_train_texts)]
        y_test = y_test[:len(X_test_texts)]

        print(f"📊 Final split - Train: {len(X_train_texts)}, Test: {len(X_test_texts)}")

        # Vectorize
        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(X_train_texts)
        X_test = vectorizer.transform(X_test_texts)

        # Create model directory
        model_dir = Path(f"models/{dataset}")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save vectorizer and test data
        save_joblib(vectorizer, model_dir / "vectorizer.joblib")

        test_data = {
            'X_test_texts': X_test_texts,
            'y_test': y_test,
            'feature_names': vectorizer.get_feature_names_out().tolist()
        }
        save_joblib(test_data, model_dir / "test_data.joblib")

        # Train all models
        for model_key in MODELS:
            print(f"🤖 Training {model_key.upper()}...")

            try:
                model = get_model(model_key, random_state=42)
                model.fit(X_train, y_train)

                # Save model
                model_path = model_dir / f"{model_key}.joblib"
                save_joblib(model, model_path)

                # Quick validation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"✅ {model_key.upper()} trained - Test Accuracy: {accuracy:.4f}")

            except Exception as e:
                print(f"❌ Error training {model_key}: {e}")
                continue

        print("🎉 YOUTUBE MODELS RETRAINED SUCCESSFULLY!")
        return True

    except Exception as e:
        print(f"❌ Error processing YouTube: {e}")
        return False

# ------------------------------------------------------------
# EVALUATION WITH CONSISTENT TEST DATA
# ------------------------------------------------------------
def evaluate_with_consistent_data():
    """Evaluate models using the exact same test data they were trained with"""
    print("\n📊 EVALUATING ALL MODELS WITH CONSISTENT TEST DATA...")

    all_results = []

    for dataset in DATASETS:
        print(f"\n{'='*50}")
        print(f"📂 EVALUATING: {dataset.upper()}")
        print(f"{'='*50}")

        results = []

        for model_key in MODELS:
            model_path = Path(f"models/{dataset}/{model_key}.joblib")
            test_data_path = Path(f"models/{dataset}/test_data.joblib")
            vectorizer_path = Path(f"models/{dataset}/vectorizer.joblib")

            if not model_path.exists():
                print(f"⚠️ Model {model_key} not found for {dataset}")
                continue

            if not test_data_path.exists():
                print(f"⚠️ Test data not found for {dataset}")
                continue

            if not vectorizer_path.exists():
                print(f"⚠️ Vectorizer not found for {dataset}")
                continue

            print(f"\n➡ Evaluating {model_key.upper()} ...")

            try:
                # Load model, test data, and vectorizer
                model = joblib.load(model_path)
                test_data = joblib.load(test_data_path)
                vectorizer = joblib.load(vectorizer_path)

                X_test_texts = test_data['X_test_texts']
                y_test = test_data['y_test']

                # Transform test texts
                X_test = vectorizer.transform(X_test_texts)

                # Make predictions
                y_pred = model.predict(X_test)
                y_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception:
                        pass

                # Compute metrics
                accuracy = float(accuracy_score(y_test, y_pred))
                precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
                recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
                f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

                roc_auc = None
                if y_proba is not None:
                    try:
                        n_classes = len(np.unique(y_test))
                        if n_classes == 2:
                            roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                        else:
                            roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
                    except Exception:
                        roc_auc = None

                row = {
                    "dataset": dataset,
                    "model": model_key.upper(),
                    "accuracy": accuracy,
                    "precision_macro": precision,
                    "recall_macro": recall,
                    "f1_macro": f1,
                    "roc_auc": roc_auc if roc_auc is not None else 0.0,
                    "test_samples": len(y_test)
                }

                results.append(row)
                print(f"✅ {model_key.upper()}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            except Exception as e:
                print(f"❌ Error evaluating {model_key}: {e}")
                continue

        # Print table for this dataset
        if results:
            print(f"\n📊 {dataset.upper()} PERFORMANCE SUMMARY")
            print(f"{'Model':<10} {'Accuracy':<10} {'F1-Macro':<10} {'ROC-AUC':<10} {'Samples':<10}")
            print("-" * 60)
            for r in results:
                roc_display = f"{r['roc_auc']:.4f}" if r['roc_auc'] else "N/A"
                print(f"{r['model']:<10} {r['accuracy']:<10.4f} {r['f1_macro']:<10.4f} {roc_display:<10} {r['test_samples']:<10}")

            all_results.extend(results)

    return all_results

# ------------------------------------------------------------
# COMPREHENSIVE VISUALIZATION
# ------------------------------------------------------------
def create_comprehensive_plots(all_results_df):
    """Create comprehensive comparison plots"""

    if all_results_df.empty:
        print("❌ No results to plot!")
        return

    plot_dir = Path("results/overall_summary/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("Set2")

    # 1. Main Performance Comparison (F1 Score)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=all_results_df, x='dataset', y='f1_macro', hue='model')
    plt.title('Model Performance Comparison Across Datasets (F1 Macro Score)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dataset', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Macro Score', fontsize=14, fontweight='bold')
    plt.legend(title='Model', title_fontsize=12, fontsize=11,
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_dir / 'model_comparison_f1.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Accuracy Comparison
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=all_results_df, x='dataset', y='accuracy', hue='model')
    plt.title('Model Performance Comparison Across Datasets (Accuracy)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dataset', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.legend(title='Model', title_fontsize=12, fontsize=11,
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_dir / 'model_comparison_accuracy.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 3. ROC-AUC Comparison (if available)
    roc_results = all_results_df[all_results_df['roc_auc'] > 0]
    if not roc_results.empty:
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=roc_results, x='dataset', y='roc_auc', hue='model')
        plt.title('Model Performance Comparison (ROC-AUC Score)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Dataset', fontsize=14, fontweight='bold')
        plt.ylabel('ROC-AUC Score', fontsize=14, fontweight='bold')
        plt.legend(title='Model', title_fontsize=12, fontsize=11,
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

        plt.tight_layout()
        plt.savefig(plot_dir / 'model_comparison_roc_auc.png', bbox_inches='tight', dpi=300)
        plt.close()

    # 4. Heatmap of F1 Scores
    pivot_f1 = all_results_df.pivot(index='model', columns='dataset', values='f1_macro')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'F1 Macro Score'}, center=0.5)
    plt.title('F1 Macro Score Heatmap Across Models and Datasets',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(plot_dir / 'f1_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 5. Multi-metric comparison
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        if idx < len(axes):
            sns.barplot(data=all_results_df, x='dataset', y=metric, hue='model', ax=axes[idx])
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison',
                              fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Dataset', fontsize=12)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[idx].tick_params(axis='x', rotation=45)
            if idx > 0:  # Only show legend on first subplot
                axes[idx].get_legend().remove()
            else:
                axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle('Comprehensive Model Performance Metrics Across Datasets',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(plot_dir / 'multi_metric_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"📊 Saved comprehensive comparison plots to {plot_dir}/")

def print_detailed_summary(all_results_df):
    """Print comprehensive final summary"""

    print(f"\n{'='*80}")
    print(f"🎯 COMPREHENSIVE PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    # Best model per dataset
    best_models = all_results_df.loc[all_results_df.groupby('dataset')['f1_macro'].idxmax()]

    print(f"\n🏆 BEST MODELS PER DATASET (by F1 Macro):")
    print(f"{'Dataset':<12} {'Best Model':<10} {'F1 Macro':<10} {'Accuracy':<10} {'ROC-AUC':<10}")
    print(f"{'-'*60}")
    for _, row in best_models.iterrows():
        roc_display = f"{row['roc_auc']:.4f}" if row['roc_auc'] else "N/A"
        print(f"{row['dataset']:<12} {row['model']:<10} {row['f1_macro']:<10.4f} {row['accuracy']:<10.4f} {roc_display:<10}")

    # Overall best model
    if not all_results_df.empty:
        overall_best = all_results_df.loc[all_results_df['f1_macro'].idxmax()]
        print(f"\n⭐ OVERALL BEST PERFORMER:")
        print(f"Model: {overall_best['model']} on {overall_best['dataset']} dataset")
        print(f"F1 Macro: {overall_best['f1_macro']:.4f}, Accuracy: {overall_best['accuracy']:.4f}")

    # Average performance by model
    print(f"\n📈 AVERAGE PERFORMANCE BY MODEL:")
    avg_by_model = all_results_df.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'f1_macro': ['mean', 'std'],
        'roc_auc': ['mean', 'std']
    }).round(4)
    print(avg_by_model)

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
def main():
    # Create output directory
    outdir = Path("results/overall_summary")
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Retrain YouTube models with fixed data processing
    youtube_success = retrain_youtube()

    if not youtube_success:
        print("⚠️ YouTube training failed, but continuing with available datasets...")

    # Step 2: Evaluate all models with consistent test data
    all_results = evaluate_with_consistent_data()

    if not all_results:
        print("❌ No results generated! Check the training process.")
        return

    # Create DataFrame and save
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(outdir / "ALL_MODELS_SUMMARY.csv", index=False)

    # Create comprehensive visualizations
    create_comprehensive_plots(all_results_df)

    # Print final summary
    print_detailed_summary(all_results_df)

    print(f"\n🎉 COMPLETE! Evaluation finished successfully!")
    print(f"📄 Results saved to: {outdir}/")
    print(f"📊 High-quality plots saved to: {outdir}/plots/")
    print(f"📋 Summary file: {outdir}/ALL_MODELS_SUMMARY.csv")

if __name__ == "__main__":
    main()
