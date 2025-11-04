import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.feature_engineering import build_vectorizer, to_dense_if_needed
from src.preprocess import normalize_corpus
from src.train_model import get_model
from src.evaluate_model import evaluate_and_save
from src.utils import Paths, ensure_dirs, save_joblib, set_global_seed, validate_dataset_csv


MODEL_KEYS = ["lr", "svm", "lgbm", "xgb"]
DATASETS = ["twitter", "reddit", "youtube", "adult"]


def load_dataset(paths: Paths) -> pd.DataFrame:
	validate_dataset_csv(paths.dataset_csv)
	df = pd.read_csv(paths.dataset_csv)
	if "text" not in df.columns or "label" not in df.columns:
		raise ValueError("dataset.csv must contain 'text' and 'label' columns")
	return df[["text", "label"]].dropna()


def run_for_dataset(dataset: str, models: List[str], test_size: float = 0.2, seed: int = 42) -> None:
	set_global_seed(seed)
	paths = Paths(dataset)
	paths.model_dir.mkdir(parents=True, exist_ok=True)
	
	df = load_dataset(paths)
	texts = df["text"].astype(str).tolist()
	labels = df["label"].astype(int).to_numpy()

	X_train_texts, X_test_texts, y_train, y_test = train_test_split(
		texts, labels, test_size=test_size, random_state=seed, stratify=labels
	)

	X_train_texts = normalize_corpus(X_train_texts)
	X_test_texts = normalize_corpus(X_test_texts)

	vectorizer = build_vectorizer()
	X_train = vectorizer.fit_transform(X_train_texts)
	X_test = vectorizer.transform(X_test_texts)

	save_joblib(vectorizer, paths.vectorizer_path)

	unique_labels = sorted(np.unique(labels).tolist())
	label_names = [str(l) for l in unique_labels]

	for mk in models:
		model = get_model(mk, random_state=seed)
		# Some models (GBMs) may prefer dense input
		Xd_train = to_dense_if_needed(X_train)
		Xd_test = to_dense_if_needed(X_test)
		model.fit(Xd_train, y_train)
		y_pred = model.predict(Xd_test)

		save_joblib(model, paths.model_path(mk))
		evaluate_and_save(
			y_test,
			y_pred,
			unique_labels,
			label_names,
			paths.report_json_path(mk),
			paths.report_csv_path(mk),
			paths.confusion_png_path(mk),
		)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run experiments per dataset and model")
	parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Datasets to run")
	parser.add_argument("--models", nargs="*", default=MODEL_KEYS, help="Models to train")
	parser.add_argument("--test_size", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	ensure_dirs()
	for dataset in tqdm(args.datasets, desc="Datasets"):
		run_for_dataset(dataset, args.models, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
	main()
