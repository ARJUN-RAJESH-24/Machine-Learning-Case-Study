"""
Scaffold for downloading datasets and normalizing them to `data/{dataset}/dataset.csv`
with columns: text, label

Notes:
- Kaggle API requires credentials (~/.kaggle/kaggle.json). See Kaggle docs.
- Hugging Face datasets can be loaded via `datasets` library.
- YouTube comments and others may require manual preprocessing; ensure final CSV matches the schema.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.utils import DATA_DIR, ensure_dirs


def save_standard_csv(df: pd.DataFrame, dataset: str) -> None:
	out_dir = DATA_DIR / dataset
	out_dir.mkdir(parents=True, exist_ok=True)
	(df[["text", "label"]]).to_csv(out_dir / "dataset.csv", index=False)


def example_manual_usage() -> None:
	# Example: Create a tiny dummy dataset
	df = pd.DataFrame({
		"text": ["I hate you", "I love you"],
		"label": [1, 0],
	})
	save_standard_csv(df, "twitter")


if __name__ == "__main__":
	ensure_dirs()
	print("This is a scaffold. Fill in dataset-specific download and normalization logic as needed.")
