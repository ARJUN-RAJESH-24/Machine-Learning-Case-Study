# src/evaluate_model.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

def evaluate_and_save(y_true, y_pred, unique_labels, label_names, out_json, out_csv, out_cm_png):
    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # ROC-AUC (multiclass)
    try:
        if len(unique_labels) == 2:
            roc = float(roc_auc_score(y_true, y_pred))
        else:
            y_true_bin = label_binarize(y_true, classes=unique_labels)
            # for predictions, if prob not available, binarize prediction
            if hasattr(y_pred, "shape") and y_pred.ndim == 2:
                y_score = y_pred
            else:
                # convert y_pred to one-hot
                y_score = label_binarize(y_pred, classes=unique_labels)
            roc = float(roc_auc_score(y_true_bin, y_score, average='weighted', multi_class='ovr'))
    except Exception:
        roc = None

    metrics = {
        "accuracy": acc,
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
        "roc_auc_weighted": (roc if roc is not None else "NA")
    }

    # Save json
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save csv
    df = pd.DataFrame([metrics])
    df.to_csv(out_csv, index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_cm_png)
    plt.close()
    print(f"💾 Saved metrics: {out_json}, {out_csv} and confusion matrix {out_cm_png}")

