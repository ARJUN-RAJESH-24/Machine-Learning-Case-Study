import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "performance_reports"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"


def ensure_dirs() -> None:
	for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR, CONFUSION_DIR]:
		d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)


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


def save_json(obj: Dict[str, Any], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def save_joblib(obj: Any, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
	return joblib.load(path)


def validate_dataset_csv(path: Path) -> None:
	if not path.exists():
		raise FileNotFoundError(
			f"Expected dataset CSV at {path}. Place a CSV with columns 'text' and 'label'."
		)
