# src/utils.py
import json
import random
from pathlib import Path
import joblib
import random
import numpy as np
import os

# ------------------------------------------------------------------
# Project directories (absolute, works from anywhere)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()  # Root of the project
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
REPORTS_DIR = RESULTS_DIR / "performance_reports"

def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "performance_reports").mkdir(exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(exist_ok=True)

def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for p in [MODELS_DIR, RESULTS_DIR, CONFUSION_DIR, REPORTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    print("All project directories ready.")


# ------------------------------------------------------------------
# Paths helper class
# ------------------------------------------------------------------
class Paths:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.lower()
        self.model_dir = MODELS_DIR / self.dataset_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Common files
        self.vectorizer_path = self.model_dir / "vectorizer.joblib"
        self.label_map_path = self.model_dir / "label_map.json"

    def model_path(self, model_key: str) -> Path:
        return self.model_dir / f"{model_key}.joblib"

    def model_path_tuned(self, model_key: str) -> Path:
        return self.model_dir / f"{model_key}_tuned.joblib"

    def report_json_path(self, model_key: str) -> Path:
        return REPORTS_DIR / f"{self.dataset_name}__{model_key}__report.json"

    def report_csv_path(self, model_key: str) -> Path:
        return REPORTS_DIR / f"{self.dataset_name}__{model_key}__metrics.csv"

    def confusion_png_path(self, model_key: str) -> Path:
        base = f"{self.dataset_name}__{model_key}__cm"
        return CONFUSION_DIR / f"{base}.png"

    def confusion_normalized_path(self, model_key: str) -> Path:
        base = f"{self.dataset_name}__{model_key}__cm"
        return CONFUSION_DIR / f"{base}_normalized.png"


# ------------------------------------------------------------------
# Save / load helpers
# ------------------------------------------------------------------
def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f"Joblib saved → {path}")


def load_joblib(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)


# ------------------------------------------------------------------
# Seeding
# ------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except:
        pass
    print(f"Global random seed set to {seed}")


# ------------------------------------------------------------------
# Human-readable label names (CRITICAL for good plots!)
# ------------------------------------------------------------------
def get_readable_label_names(dataset_name: str, label_codes: list) -> list:
    """
    Convert numeric class codes → human-readable names.
    Add new datasets here as needed.
    """
    dataset = dataset_name.lower()

    if dataset == "youtube":
        # Full list from HuffPost / News Category Dataset v3 (41 classes)
        YT_CATS = [
            "U.S. NEWS", "COMEDY", "PARENTING", "WORLD NEWS", "ARTS & CULTURE",
            "TECH", "BUSINESS", "SPORTS", "ENTERTAINMENT", "POLITICS",
            "SCIENCE", "STYLE & BEAUTY", "TRAVEL", "FOOD & DRINK", "HEALTHY LIVING",
            "QUEER VOICES", "HOME & LIVING", "WOMEN", "BLACK VOICES", "LATINO VOICES",
            "ASIAN VOICES", "EDUCATION", "COLLEGE", "IMPACT", "FIFTY", "GOOD NEWS",
            "THE WORLDPOST", "WEIRD NEWS", "TASTE", "GREEN", "WELLNESS",
            "DIVORCE", "WEDDINGS", "MONEY", "ENVIRONMENT", "CULTURE & ARTS",
            "CELEBRITY", "MEDIA", "CRIME", "RELIGION", "MISCELLANEOUS"
        ]
        return [YT_CATS[i] if i < len(YT_CATS) else f"class_{i}" for i in label_codes]



    elif dataset in ["twitter", "reddit"]:
        # Common hate speech labeling
        HATE_MAP = {0: "Normal", 1: "Hate Speech", 2: "Offensive Language"}
        return [HATE_MAP.get(code, f"class_{code}") for code in label_codes]

    else:
        # Fallback
        return [f"class_{code}" for code in label_codes]


def save_label_map(label_map: dict, path: Path) -> None:
    """Save {code: name} mapping for later use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"Label map saved → {path}")


def load_label_map(path: Path) -> dict:
    """Load saved label map {code: name} """
    with open(path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}
