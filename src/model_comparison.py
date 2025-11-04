from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from .utils import REPORTS_DIR


def collect_reports() -> pd.DataFrame:
	rows = []
	for p in REPORTS_DIR.glob("*__metrics.csv"):
		# File pattern: {dataset}__{model}__metrics.csv
		stem = p.stem
		parts = stem.split("__")
		if len(parts) < 3:
			continue
		dataset, model, _ = parts[0], parts[1], parts[2]
		df = pd.read_csv(p)
		df["dataset"] = dataset
		df["model"] = model
		rows.append(df)
	if not rows:
		return pd.DataFrame()
	return pd.concat(rows, ignore_index=True)


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
	if df.empty:
		return
	# Bar chart of macro F1 per model for each dataset
	fig, ax = plt.subplots(figsize=(8, 5))
	pivot = df[["dataset", "model", "f1_macro"]]
	pivot.sort_values(["dataset", "f1_macro"], ascending=[True, False], inplace=True)
	for dataset, sub in pivot.groupby("dataset"):
		sub.plot(kind="bar", x="model", y="f1_macro", ax=ax, label=dataset)
	ax.set_ylabel("Macro F1")
	ax.set_title("Model Comparison by Dataset (Macro F1)")
	ax.legend(title="Dataset")
	fig.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def main(output_png: str = "results/performance_reports/model_comparison.png") -> None:
	df = collect_reports()
	plot_comparison(df, Path(output_png))


if __name__ == "__main__":
	main()
