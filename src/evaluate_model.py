# src/evaluate_model.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)

from .utils import save_json


def evaluate_metrics(model, X, y_true):
    y_pred = model.predict(X)
    result = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # try to get probabilities for ROC AUC
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            probs = None
    if probs is None and hasattr(model, "decision_function"):
        try:
            probs = model.decision_function(X)
        except Exception:
            probs = None

    if probs is not None:
        try:
            result["ROC-AUC"] = float(roc_auc_score(y_true, probs))
        except Exception:
            result["ROC-AUC"] = None
    else:
        result["ROC-AUC"] = None

    return result

def plot_confusion_matrix(model, X, y_true, labels=None, filepath=None):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def plot_roc_curve(model, X, y_true, filepath=None):
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            probs = None
    if probs is None and hasattr(model, "decision_function"):
        try:
            probs = model.decision_function(X)
        except Exception:
            probs = None

    if probs is None:
        raise RuntimeError("Model does not provide probability or decision_function for ROC curve.")

    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
		"recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
		"f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
	}


def save_confusion_matrix(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	labels: List[int],
	label_names: List[str],
	out_path: Path,
) -> None:
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names, ax=ax)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_title("Confusion Matrix")
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def evaluate_and_save(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	labels: List[int],
	label_names: List[str],
	report_json: Path,
	report_csv: Path,
	cm_png: Path,
) -> Tuple[Dict[str, float], str]:
	metrics = compute_metrics(y_true, y_pred)
	report_text = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
	# Save JSON and CSV (metrics)
	save_json({"metrics": metrics, "classification_report": report_text}, report_json)
	pd.DataFrame([metrics]).to_csv(report_csv, index=False)
	# Save confusion matrix plot
	save_confusion_matrix(y_true, y_pred, labels, label_names, cm_png)
	return metrics, report_text
