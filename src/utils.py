# src/utils.py
import json
from pathlib import Path
import joblib
import random
import numpy as np
import os

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "performance_reports").mkdir(exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(exist_ok=True)

class Paths:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.model_dir = MODELS_DIR / dataset_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer_path = self.model_dir / "vectorizer.joblib"
    def model_path(self, mk):
        return MODELS_DIR / self.dataset_name / f"{mk}.joblib"
    def report_json_path(self, mk):
        return RESULTS_DIR / "performance_reports" / f"{self.dataset_name}__{mk}__report.json"
    def report_csv_path(self, mk):
        return RESULTS_DIR / "performance_reports" / f"{self.dataset_name}__{mk}__metrics.csv"
    def confusion_png_path(self, mk):
        return RESULTS_DIR / "confusion_matrices" / f"{self.dataset_name}__{mk}__cm.png"

def save_joblib(obj, path):
    joblib.dump(obj, path)
    print(f"💾 Joblib object saved → {path}")

def set_global_seed(s):
    import random, numpy as _np
    random.seed(s); _np.random.seed(s)

