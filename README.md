# Hate Speech & Adult Content Classification (Classic ML)

This project provides a comparative evaluation framework for text-based hate speech and adult content detection using classic machine learning models:
- Logistic Regression
- Linear SVM
- LightGBM
- XGBoost

## Key Principles
- Datasets are treated independently (no merging)
- Unified preprocessing, vectorization (TF-IDF), and metrics
- Per-dataset training, evaluation, and visualization

## Project Structure

```
HateSpeech_AdultContent_Classification/
├── data/
│   ├── twitter/
│   ├── reddit/
│   ├── youtube/
│   └── adult/
├── models/
│   ├── twitter/
│   ├── reddit/
│   ├── youtube/
│   └── adult/
├── results/
│   ├── performance_reports/
│   └── confusion_matrices/
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── model_comparison.py
│   └── utils.py
├── download_datasets.py
├── run_experiments.py
├── test_model.py
├── main.py
├── requirements.txt
└── README.md
```

## Data Preparation
Place a standardized CSV per dataset under `data/{dataset}/dataset.csv` with columns:
- `text`: raw text string
- `label`: integer class label (binary or multi-class)

Datasets:
- Kaggle: `vkrahul/twitter-hate-speech`, `mrmorj/hate-speech-and-offensive-language-dataset`, YouTube comments dataset (place after local preprocessing),
- Hugging Face: `PKU-Alignment/SafeSora-Label` (adult-content subset).

Use `download_datasets.py` as a scaffold to fetch and normalize into the required format, or manually place files.

## Setup
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # PowerShell on Windows
pip install -r requirements.txt
```

## Run Experiments
Train all models on each dataset separately and produce metrics/plots:
```bash
python run_experiments.py --datasets twitter reddit youtube adult --models lr svm lgbm xgb
```
Outputs:
- Trained models: `models/{dataset}/`
- Reports (CSV/JSON): `results/performance_reports/`
- Confusion matrices: `results/confusion_matrices/`
- Comparison plots: `results/performance_reports/`

## Test a Trained Model
```bash
python test_model.py --dataset twitter --model svm --texts "I hate you" "Have a great day"
```

## Reproducibility
- Fixed random seeds where applicable
- Consistent TF-IDF vectorizer configuration across models/datasets

## Notes
- LightGBM/XGBoost require numeric features; TF-IDF provides sparse input. LightGBM/XGBoost will be trained on dense arrays; ensure memory is sufficient or downscale features via `max_features` in `feature_engineering.py`.
- If datasets have different label spaces, metrics are computed per dataset independently (macro F1 recommended).