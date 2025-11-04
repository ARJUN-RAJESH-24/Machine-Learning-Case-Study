"""
Utility Module for ML Project (Linux + GPU Ready)
------------------------------------------------
- Handles paths, saving/loading, dataset validation.
- Provides global seeding for reproducibility (CPU + GPU).
- Detects GPU availability using CuPy.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np

# ================================================================
# ✅ PROJECT PATHS (Linux / WSL compatible)
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "performance_reports"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"


# ================================================================
# ✅ Ensure Directory Structure Exists
# ================================================================
def ensure_dirs() -> None:
    for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR, CONFUSION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("📁 Verified all directories exist.")


# ================================================================
# ✅ Global Seed Setup (CPU + GPU)
# ================================================================
def set_global_seed(seed: int = 42) -> None:
    """
    Sets global random seed for reproducibility across CPU and GPU computations.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import cupy
        cupy.random.seed(seed)
        print(f"🎯 Global random seed set (CPU + GPU): {seed}")
    except ImportError:
        print(f"🎯 Global random seed set (CPU only): {seed}")


# ================================================================
# ✅ GPU Detection
# ================================================================
def gpu_available(verbose: bool = True) -> bool:
    """
    Detects CUDA-capable GPU using CuPy.
    Returns True if at least one GPU is found.
    """
    try:
        import cupy
        gpu_count = cupy.cuda.runtime.getDeviceCount()
        if gpu_count > 0:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            gpu_name = props["name"].decode()
            cuda_ver = cupy.cuda.runtime.runtimeGetVersion() / 1000
            if verbose:
                print(f"⚡ GPU detected: {gpu_name} (CUDA {cuda_ver:.1f})")
            return True
        if verbose:
            print("⚠️ No GPU devices detected.")
    except Exception as e:
        if verbose:
            print(f"⚠️ GPU check failed: {e}")
    return False


# ================================================================
# ✅ Dataset and Model Path Management
# ================================================================
@dataclass
class Paths:
    dataset_name: str

    @property
    def dataset_dir(self) -> Path:
        return DATA_DIR / self.dataset_name

    @property
    def dataset_csv(self) -> Path:
        return self.dataset_dir / "dataset.csv"

    @property
    def model_dir(self) -> Path:
        return MODELS_DIR / self.dataset_name

    @property
    def vectorizer_path(self) -> Path:
        return self.model_dir / "vectorizer.joblib"

    def model_path(self, model_key: str) -> Path:
        return self.model_dir / f"{model_key}.joblib"

    def report_json_path(self, model_key: str) -> Path:
        return REPORTS_DIR / f"{self.dataset_name}__{model_key}__report.json"

    def report_csv_path(self, model_key: str) -> Path:
        return REPORTS_DIR / f"{self.dataset_name}__{model_key}__metrics.csv"

    def confusion_png_path(self, model_key: str) -> Path:
        return CONFUSION_DIR / f"{self.dataset_name}__{model_key}__cm.png"


# ================================================================
# ✅ File Save / Load Utilities
# ================================================================
def save_json(obj: Dict[str, Any], path: Path) -> None:
    """
    Saves dictionary as JSON file (UTF-8, Linux-safe).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON saved → {path}")


def save_joblib(obj: Any, path: Path) -> None:
    """
    Saves model/vectorizer using joblib (GPU-safe if object CPU-backed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(obj, path)
        print(f"💾 Joblib object saved → {path}")
    except Exception as e:
        print(f"❌ Failed to save joblib object ({e})")


def load_joblib(path: Path) -> Any:
    """
    Loads joblib file from disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")
    return joblib.load(path)


# ================================================================
# ✅ Dataset Validation
# ================================================================
def validate_dataset_csv(path: Path) -> None:
    """
    Ensures dataset CSV exists and has required columns.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"❌ Expected dataset CSV at {path}. Ensure it includes columns: 'text' and 'label'."
        )
