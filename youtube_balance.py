#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

SRC = Path("data/youtube/News_Category_Dataset_v3.json")
OUT = Path("data/youtube/youtube_balanced.csv")

def main():
    print("📥 Loading YouTube dataset...")

    # Load JSON Lines correctly
    df = pd.read_json(SRC, lines=True)
    print(f"✅ Loaded {len(df)} rows")

    # Standardize columns
    df = df.rename(columns={"headline": "text", "category": "label"})
    df = df[["text", "label"]].dropna()

    print("📊 Finding top-10 categories...")
    top10 = df["label"].value_counts().head(10).index.tolist()
    df = df[df["label"].isin(top10)]

    print(f"✅ After filtering top-10 labels: {len(df)} rows")

    # Balance
    min_count = df["label"].value_counts().min()
    print(f"🔧 Balancing each class to {min_count} samples...")

    df_bal = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(min_count, random_state=42))
          .reset_index(drop=True)
    )

    print(f"✅ Final balanced size: {len(df_bal)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_csv(OUT, index=False)

    print(f"🎉 Saved → {OUT}")

if __name__ == "__main__":
    main()

