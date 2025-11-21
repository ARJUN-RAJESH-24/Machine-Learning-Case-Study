#!/usr/bin/env python3
"""
Comprehensive Dataset Analyzer
Identifies ALL columns, features, classes, and generates exact mapping code
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json


# ============================================================
# FIX: NUMPY → PYTHON SERIALIZER
# ============================================================
def to_python(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


def analyze_dataset(name, filepath):
    """Deep analysis of dataset structure"""

    print(f"\n{'='*100}")
    print(f"📊 ANALYZING: {name.upper()}")
    print(f"{'='*100}")
    print(f"📂 Path: {filepath}")

    if not filepath.exists():
        print(f"❌ FILE NOT FOUND\n")
        return None

    try:
        # Read dataset
        df = pd.read_csv(filepath)

        # ============================================================
        # SECTION 1: BASIC METADATA
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"📈 BASIC INFORMATION")
        print(f"{'─'*100}")
        print(f"Total Rows:      {len(df):,}")
        print(f"Total Columns:   {len(df.columns)}")
        print(f"Memory:          {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Duplicates:      {df.duplicated().sum():,}")

        # ============================================================
        # SECTION 2: ALL COLUMNS WITH COMPLETE DETAILS
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"📋 COMPLETE COLUMN INVENTORY")
        print(f"{'─'*100}")
        print(f"{'#':<4} {'Column Name':<30} {'Type':<12} {'Nulls':<10} {'Unique':<10} {'Sample Value':<40}")
        print(f"{'─'*100}")

        column_info = []
        for idx, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            nulls = df[col].isnull().sum()
            unique = df[col].nunique()

            non_null_vals = df[col].dropna()
            if len(non_null_vals) > 0:
                sample = str(non_null_vals.iloc[0])
                if len(sample) > 37:
                    sample = sample[:37] + "..."
            else:
                sample = "N/A"

            print(f"{idx:<4} {col:<30} {dtype:<12} {nulls:<10,} {unique:<10,} {sample:<40}")

            column_info.append({
                'index': int(idx),
                'name': col,
                'dtype': dtype,
                'nulls': int(nulls),
                'null_pct': float(round(nulls / len(df) * 100, 2)),
                'unique': int(unique),
                'sample': sample
            })

        # ============================================================
        # SECTION 3: TEXT COLUMN IDENTIFICATION
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"📝 TEXT COLUMN CANDIDATES (Long String Fields)")
        print(f"{'─'*100}")

        text_candidates = []
        for col in df.columns:
            if df[col].dtype == 'object':
                lengths = df[col].astype(str).str.len()
                avg_len = lengths.mean()
                max_len = lengths.max()

                if avg_len > 15:
                    text_candidates.append({
                        'column': col,
                        'avg_length': float(avg_len),
                        'max_length': int(max_len),
                        'min_length': int(lengths.min()),
                        'contains_spaces': int(df[col].astype(str).str.contains(' ').sum()),
                    })

        if text_candidates:
            text_candidates.sort(key=lambda x: x['avg_length'], reverse=True)

            print(f"{'Column':<30} {'Avg Len':<10} {'Max Len':<10} {'Min Len':<10} {'Has Spaces':<12}")
            print(f"{'─'*100}")
            for tc in text_candidates:
                print(f"{tc['column']:<30} {tc['avg_length']:<10.1f} {tc['max_length']:<10} "
                      f"{tc['min_length']:<10} {tc['contains_spaces']:<12,}")

            print(f"\n✅ RECOMMENDED TEXT COLUMN: '{text_candidates[0]['column']}'")
        else:
            print("⚠️  No obvious text columns found")

        # ============================================================
        # SECTION 4: LABEL/CLASS COLUMN IDENTIFICATION
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"🏷️  LABEL/CLASS COLUMN CANDIDATES")
        print(f"{'─'*100}")

        label_candidates = []
        for col in df.columns:
            unique_count = df[col].nunique()

            if 2 <= unique_count <= 500:
                balance = (df[col].value_counts().min() /
                           df[col].value_counts().max())

                label_candidates.append({
                    'column': col,
                    'n_classes': int(unique_count),
                    'dtype': str(df[col].dtype),
                    'balance': float(balance)
                })

        if label_candidates:
            label_candidates.sort(key=lambda x: x['n_classes'])

            print(f"{'Column':<30} {'Classes':<10} {'Type':<12} {'Balance':<10}")
            print(f"{'─'*100}")
            for lc in label_candidates:
                print(f"{lc['column']:<30} {lc['n_classes']:<10} {lc['dtype']:<12} {lc['balance']:<10.3f}")
        else:
            print("⚠️  No obvious label columns found")

        # ============================================================
        # CLASS DISTRIBUTION FOR LABELS
        # ============================================================
        if label_candidates:
            print(f"\n{'─'*100}")
            print(f"📊 CLASS DISTRIBUTION FOR TOP LABEL CANDIDATES")
            print(f"{'─'*100}")

            for lc in label_candidates[:3]:
                col = lc['column']
                print(f"\n🏷️  Column: '{col}' ({lc['n_classes']} classes)")
                print(f"   {'─'*80}")

                value_counts = df[col].value_counts().sort_index()
                if len(value_counts) <= 20:
                    for val, count in value_counts.items():
                        pct = (count / len(df)) * 100
                        bar = '█' * int(pct / 2)
                        print(f"   {str(val):<30} {count:>8,} ({pct:>5.2f}%) {bar}")
                else:
                    print(f"   Top 10 classes:")
                    for val, count in value_counts.head(10).items():
                        pct = (count / len(df)) * 100
                        bar = '█' * int(pct / 2)
                        print(f"   {str(val):<30} {count:>8,} ({pct:>5.2f}%) {bar}")

        # ============================================================
        # GENERATED MAPPING CODE
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"💻 GENERATED PYTHON CODE FOR THIS DATASET")
        print(f"{'─'*100}")

        best_text = text_candidates[0]['column'] if text_candidates else None
        best_label = label_candidates[0]['column'] if label_candidates else None

        print(f"\n# For dataset: {name}")
        print(f"elif name == \"{name.lower()}\":")
        print(f"    df = pd.read_csv(base / \"{filepath.name}\")")

        if best_text and best_text != 'text':
            print(f"    if \"{best_text}\" in df.columns:")
            print(f"        df = df.rename(columns={{\"{best_text}\": \"text\"}})")

        if best_label and best_label != 'label':
            print(f"    if \"{best_label}\" in df.columns:")
            print(f"        df = df.rename(columns={{\"{best_label}\": \"label\"}})")

        # ============================================================
        # DATA QUALITY CHECKS
        # ============================================================
        print(f"\n{'─'*100}")
        print(f"🔍 DATA QUALITY CHECKS")
        print(f"{'─'*100}")

        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Missing Values Found:")
            for col in missing[missing > 0].index:
                pct = (missing[col] / len(df)) * 100
                print(f"   {col:<30} {missing[col]:>8,} ({pct:>5.2f}%)")
        else:
            print("✅ No missing values")

        if df.duplicated().sum() > 0:
            print(f"\n⚠️  Duplicate rows: {df.duplicated().sum():,}")
        else:
            print("✅ No duplicate rows")

        # ============================================================
        # END: RETURN RESULTS
        # ============================================================
        return {
            'dataframe': df,
            'text_column': best_text,
            'label_column': best_label,
            'n_classes': int(label_candidates[0]['n_classes']) if label_candidates else None,
            'column_info': column_info,
            'text_candidates': text_candidates,
            'label_candidates': label_candidates
        }

    except Exception as e:
        print(f"❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def generate_unified_loader(results):
    print("\n" + "="*100)
    print("🚀 COMPLETE LOAD_DATASET() FUNCTION")
    print("="*100 + "\n")

    print("""def load_dataset(name):
    base = Path("data") / name

    # ============================================================
    # LOAD DATASETS
    # ============================================================
""")

    for name, result in results.items():
        if result is None:
            continue

        text_col = result['text_column']
        label_col = result['label_column']

        print(f"    if name == \"{name.lower()}\":")
        print(f"        df = pd.read_csv(base / \"data.csv\")")

        if text_col and text_col != 'text':
            print(f"        if \"{text_col}\" in df.columns:")
            print(f"            df = df.rename(columns={{\"{text_col}\": \"text\"}})")

        if label_col and label_col != 'label':
            print(f"        if \"{label_col}\" in df.columns:")
            print(f"            df = df.rename(columns={{\"{label_col}\": \"label\"}})")

        print()

    print("""    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # ============================================================
    # VALIDATION & CLEANUP
    # ============================================================
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    if not np.issubdtype(df["label"].dtype, np.number):
        df["label"] = df["label"].astype("category").cat.codes

    return df
""")


def main():
    print("\n" + "🔬 " * 40)
    print("COMPREHENSIVE DATASET FEATURE & CLASS ANALYZER")
    print("🔬 " * 40)

    datasets = {
        "Twitter": Path("data/twitter/train_E6oV3lV.csv"),
        "Reddit": Path("data/reddit/labeled_data.csv"),
        "YouTube": Path("data/youtube/youtube_balanced.csv"),
        "Adult": Path("data/adult/adult_dataset.csv"),
    }

    results = {}

    for name, filepath in datasets.items():
        results[name] = analyze_dataset(name, filepath)

    generate_unified_loader(results)

    print("\n" + "="*100)
    print("📋 FINAL SUMMARY TABLE")
    print("="*100)
    print(f"{'Dataset':<15} {'Text Column':<25} {'Label Column':<25} {'Classes':<10} {'Rows':<12}")
    print("─"*100)

    for name, result in results.items():
        if result:
            print(f"{name:<15}"
                  f"{(result['text_column'] or 'NOT FOUND'):<25}"
                  f"{(result['label_column'] or 'NOT FOUND'):<25}"
                  f"{str(result['n_classes'] or 'N/A'):<10}"
                  f"{len(result['dataframe']):<12,}")

    print("="*100)

    # ============================================================
    # FIXED JSON SAVE
    # ============================================================
    summary = {}
    for name, result in results.items():
        if result:
            summary[name] = to_python({
                'text_column': result['text_column'],
                'label_column': result['label_column'],
                'n_classes': result['n_classes'],
                'n_rows': len(result['dataframe']),
                'columns': result['column_info']
            })

    output_file = Path("dataset_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Detailed analysis saved to: {output_file}")
    print("✅ Analysis complete!\n")


if __name__ == "__main__":
    main()

