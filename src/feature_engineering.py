# src/feature_engineering.py
from typing import Tuple, Iterable
import numpy as np
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(ngram_range=(1,2), max_features=100000, min_df=2, max_df=0.95):
    vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=min_df, max_df=max_df, norm="l2", sublinear_tf=True, analyzer="word")
    return vec

def to_dense_if_needed(X, max_dense_size_mb: int = 1024):
    # If already ndarray, return
    import numpy as np
    from scipy import sparse
    if isinstance(X, np.ndarray):
        return X
    # If sparse matrix, estimate size
    if sparse.issparse(X):
        bytes_est = X.shape[0] * X.shape[1] * 8
        mb = bytes_est / (1024*1024)
        if mb <= max_dense_size_mb:
            print(f"🧩 Converting sparse -> dense (est {mb:.2f} MB)")
            return X.toarray()
        else:
            print(f"⚠️ Keeping sparse matrix (est {mb:.2f} MB)")
            return X
    return X

