# src/evaluate_model.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize

def evaluate_and_save(y_true, y_pred, unique_labels, label_names, out_json, out_csv, out_cm_png):
    """
    Comprehensive evaluation with all metrics needed for model_comparison.py
    """
    # Basic accuracy
    acc = float(accuracy_score(y_true, y_pred))
    
    # Weighted metrics
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Macro metrics (ADDED for model_comparison.py)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Additional statistical metrics (ADDED for model_comparison.py)
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0
    
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = 0.0

    # ROC-AUC (multiclass)
    roc_auc = None
    roc_auc_ovr = None
    roc_auc_ovo = None
    
    try:
        if len(unique_labels) == 2:
            # Binary classification
            roc_auc = float(roc_auc_score(y_true, y_pred))
        else:
            # Multiclass - need to binarize
            y_true_bin = label_binarize(y_true, classes=unique_labels)
            
            # For predictions, if prob not available, binarize prediction
            if hasattr(y_pred, "shape") and y_pred.ndim == 2:
                y_score = y_pred
            else:
                # convert y_pred to one-hot
                y_score = label_binarize(y_pred, classes=unique_labels)
            
            # Try both OVR and OVO strategies
            try:
                roc_auc_ovr = float(roc_auc_score(
                    y_true_bin, y_score, average='weighted', multi_class='ovr'
                ))
            except Exception:
                pass
            
            try:
                roc_auc_ovo = float(roc_auc_score(
                    y_true_bin, y_score, average='weighted', multi_class='ovo'
                ))
            except Exception:
                pass
            
            # Use OVR as primary
            roc_auc = roc_auc_ovr
    except Exception as e:
        print(f"⚠️ Could not calculate ROC-AUC: {e}")

    # Build comprehensive metrics dictionary (compatible with model_comparison.py)
    metrics = {
        "accuracy": acc,
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(p_m),      # ADDED for model_comparison.py
        "recall_macro": float(r_m),         # ADDED for model_comparison.py
        "f1_macro": float(f1_m),            # ADDED for model_comparison.py
        "matthews_corrcoef": mcc,           # ADDED for model_comparison.py
        "cohen_kappa": kappa,               # ADDED for model_comparison.py
    }
    
    # Add ROC-AUC metrics if available
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
        metrics["roc_auc_weighted"] = roc_auc
    else:
        metrics["roc_auc_weighted"] = "NA"  # For backward compatibility
        
    if roc_auc_ovr is not None:
        metrics["roc_auc_ovr"] = roc_auc_ovr
    if roc_auc_ovo is not None:
        metrics["roc_auc_ovo"] = roc_auc_ovo
    
    # Save JSON
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save CSV
    df = pd.DataFrame([metrics])
    df.to_csv(out_csv, index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    plt.figure(figsize=(max(8, len(unique_labels)), max(6, len(unique_labels))))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=label_names, 
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted", fontsize=12, fontweight='bold')
    plt.ylabel("Actual", fontsize=12, fontweight='bold')
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_cm_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved metrics: {out_json}, {out_csv} and confusion matrix {out_cm_png}")
    
    return metrics
