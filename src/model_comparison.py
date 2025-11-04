from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from .utils import REPORTS_DIR

# Try GPU-based RAPIDS libraries
try:
    import cudf
    import cupy as cp
    GPU_ENABLED = True
    print("⚡ Using GPU acceleration for report aggregation (cuDF + CuPy)")
except ImportError:
    import pandas as pd
    GPU_ENABLED = False
    print("⚠️ cuDF not found, using CPU (pandas)")

def collect_reports():
    """Collect all __metrics.csv reports into one DataFrame (GPU if available)."""
    rows = []
    for p in REPORTS_DIR.glob("*__metrics.csv"):
        stem = p.stem
        parts = stem.split("__")
        if len(parts) < 3:
            continue
        dataset, model, _ = parts[0], parts[1], parts[2]

        # Load using cuDF or pandas
        if GPU_ENABLED:
            df = cudf.read_csv(p)
            df["dataset"] = dataset
            df["model"] = model
        else:
            df = pd.read_csv(p)
            df["dataset"] = dataset
            df["model"] = model

        rows.append(df)

    if not rows:
        return cudf.DataFrame() if GPU_ENABLED else pd.DataFrame()

    if GPU_ENABLED:
        return cudf.concat(rows, ignore_index=True)
    return pd.concat(rows, ignore_index=True)


def plot_comparison(df, out_path: Path):
    """Plot F1 comparison using Matplotlib (CPU rendering)."""
    if df.empty:
        print("⚠️ No report files found.")
        return

    # Convert to pandas for plotting (Matplotlib can't use cuDF directly)
    if GPU_ENABLED and isinstance(df, cudf.DataFrame):
        df = df.to_pandas()

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
    print(f"✅ Saved performance comparison to {out_path}")


def main(output_png: str = "results/performance_reports/model_comparison.png"):
    df = collect_reports()
    plot_comparison(df, Path(output_png))


if __name__ == "__main__":
    main()
