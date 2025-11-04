# src/feature_engineering.py
from typing import Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def build_vectorizer(
	ngram_range: Tuple[int, int] = (1, 2),
	max_features: int = 100000,
	min_df: int = 2,
	max_df: float = 0.95,
) -> TfidfVectorizer:
	return TfidfVectorizer(
		ngram_range=ngram_range,
		max_features=max_features,
		min_df=min_df,
		max_df=max_df,
		norm="l2",
		sublinear_tf=True,
		analyzer="word",
	)


def to_dense_if_needed(X: Union[sparse.spmatrix, np.ndarray], max_dense_size_mb: int = 1024) -> Union[sparse.spmatrix, np.ndarray]:
	if isinstance(X, np.ndarray):
		return X
	# Estimate worst-case size if dense
	bytes_estimate = (X.shape[0] * X.shape[1]) * 8  # float64
	mb_estimate = bytes_estimate / (1024 * 1024)
	if mb_estimate <= max_dense_size_mb:
		return X.toarray()
	return X


def build_tfidf(train_texts, test_texts, max_features=5000, ngram_range=(1,2)):
    """
    Fit a TF-IDF vectorizer on train_texts and transform both train and test texts.
    Returns: X_train, X_test, vectorizer
    """
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec

def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)

def load_vectorizer(path):
    return joblib.load(path)
