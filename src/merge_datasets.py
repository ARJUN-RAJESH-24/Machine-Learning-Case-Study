import os
import pandas as pd

BASE_DIR = "data"
OUTPUT_PATH = os.path.join(BASE_DIR, "combined_dataset.csv")

def load_any(path):
    """Load CSV/JSON/TSV files robustly."""
    if path.endswith(".csv"):
        return pd.read_csv(path, encoding_errors="ignore")
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", encoding_errors="ignore")
    elif path.endswith(".json"):
        return pd.read_json(path, encoding_errors="ignore", lines=True)
    return pd.DataFrame()

def collect_datasets():
    all_dfs = []

    for subdir in os.listdir(BASE_DIR):
        full = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(full):
            continue

        for file in os.listdir(full):
            if file.endswith((".csv", ".tsv", ".json")):
                df = load_any(os.path.join(full, file))
                if not df.empty:
                    df["source"] = subdir
                    all_dfs.append(df)
                    print(f"✅ Loaded {file} from {subdir} with {len(df)} rows")

    if not all_dfs:
        print("❌ No datasets found.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True, sort=False)

def main():
    df = collect_datasets()
    if not df.empty:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n🎉 Combined dataset saved → {OUTPUT_PATH} ({len(df)} rows)")
    else:
        print("⚠️ Nothing to merge.")

if __name__ == "__main__":
    main()
