# main.py
import argparse
import os
import subprocess
import sys
from download_datasets import download_all_datasets
from train_and_evaluate import train_all_datasets, train_single_dataset
from test_model import test_model_cli

DATASETS = {
    "twitter": "data/twitter_hate.csv",
    "reddit": "data/reddit_toxic.csv",
    "youtube": "data/youtube_comments.csv",
    "adult": "data/adult_content.csv"
}

def main() -> None:
	parser = argparse.ArgumentParser(description="End-to-end: run experiments and comparison plots")
	parser.add_argument("--datasets", nargs="*", default=["twitter", "reddit", "youtube", "adult"]) 
	parser.add_argument("--models", nargs="*", default=["lr", "svm", "lgbm", "xgb"]) 
	args = parser.parse_args()

	cmd = [sys.executable, "run_experiments.py", "--datasets", *args.datasets, "--models", *args.models]
	subprocess.check_call(cmd)
	# Build comparison plot
	subprocess.check_call([sys.executable, "-m", "src.model_comparison"]) 


if __name__ == "__main__":
	main()
