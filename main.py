# main.py
import argparse
import os
from download_datasets import download_all_datasets
from train_and_evaluate import train_all_datasets, train_single_dataset
from test_model import test_model_cli

DATASETS = {
    "twitter": "data/twitter_hate.csv",
    "reddit": "data/reddit_toxic.csv",
    "youtube": "data/youtube_comments.csv",
    "adult": "data/adult_content.csv"
}

def main():
    parser = argparse.ArgumentParser(description="HateSpeech & AdultContent Classification CLI")
    parser.add_argument("action", choices=["download", "train", "train-one", "test"], help="Action to perform")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Dataset key (twitter, reddit, youtube, adult)")
    parser.add_argument("--model", help="Model name when testing (LogisticRegression, SVM, XGBoost, LightGBM)")
    parser.add_argument("--fold", type=int, help="Fold number when testing")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds for training (default 5)")
    args = parser.parse_args()

    if args.action == "download":
        download_all_datasets()
    elif args.action == "train":
        train_all_datasets(n_splits=args.n_splits)
    elif args.action == "train-one":
        if not args.dataset:
            parser.error("--dataset is required for train-one")
        train_single_dataset(args.dataset, n_splits=args.n_splits)
    elif args.action == "test":
        if not (args.dataset and args.model and args.fold is not None):
            parser.error("--dataset, --model and --fold are required for test")
        test_model_cli(dataset_key=args.dataset, model_name=args.model, fold=args.fold)
    else:
        parser.error("unknown action")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/vectorizers", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    main()
