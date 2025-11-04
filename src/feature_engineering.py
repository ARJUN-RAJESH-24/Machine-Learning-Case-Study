"""
GPU-Accelerated Feature Engineering Module
------------------------------------------
Uses RAPIDS cuML's TF-IDF Vectorizer when available, otherwise falls back to scikit-learn.
Fully compatible with Linux, WSL2, or native CUDA setups.
"""

from typing import Tuple, Union
import numpy as np
import joblib
from scipy import sparse
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# ✅ GPU Detection and Backend Setup
# ================================================================
GPU_ENABLED = False
try:
    import cupy
    gpu_count = cupy.cuda.runtime.getDeviceCount()
    if gpu_count > 0:
        props = cupy.cuda.runtime.getDeviceProperties(0)
        gpu_name = props["name"].decode()
        print(f"⚡ Detected GPU: {gpu_name} — enabling cuML acceleration")
        from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
        GPU_ENABLED = True
except Exception as e:
    print(f"⚠️ GPU libraries unavailable ({e}) — using CPU backend")

# CPU fallback
if not GPU_ENABLED:
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("🧠 Using scikit-learn TF-IDF (CPU mode)")


# ================================================================
# ✅ TF-IDF Builder
# ================================================================
def build_vectorizer(
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 100000,
    min_df: int = 2,
    max_df: float = 0.95,
):
    """
    Returns a GPU or CPU TF-IDF vectorizer depending on availability.
    """
    if GPU_ENABLED:
        return cuTfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            norm="l2",
            sublinear_tf=True,
            analyzer="word",
        )
    else:
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            norm="l2",
            sublinear_tf=True,
            analyzer="word",
        )


# ================================================================
# ✅ Sparse → Dense Converter
# ================================================================
def to_dense_if_needed(
    X: Union[sparse.spmatrix, np.ndarray],
    max_dense_size_mb: int = 1024
) -> Union[np.ndarray, sparse.spmatrix]:
    """
    Converts a sparse matrix to dense only if the resulting array fits in memory.
    Works for both CPU and GPU arrays.
    """
    if isinstance(X, np.ndarray):
        return X

    try:
        import cupy
        if isinstance(X, cupy.ndarray):
            return X  # Already on GPU
    except ImportError:
        pass

    # Estimate dense size
    bytes_estimate = X.shape[0] * X.shape[1] * 8  # float64
    mb_estimate = bytes_estimate / (1024 * 1024)
    if mb_estimate <= max_dense_size_mb:
        print(f"🧩 Converting to dense (estimated size: {mb_estimate:.2f} MB)")
        return X.toarray()
    else:
        print(f"⚠️ Keeping sparse matrix (too large: {mb_estimate:.2f} MB)")
    return X


# ================================================================
# ✅ Build & Transform TF-IDF
# ================================================================
def build_tfidf(train_texts, test_texts, max_features=5000, ngram_range=(1, 2)):
    """
    Fit a TF-IDF vectorizer on train_texts and transform both train and test.
    Uses GPU if available (via cuML).
    """
    backend = "GPU (cuML)" if GPU_ENABLED else "CPU (scikit-learn)"
    print(f"🔧 Building TF-IDF using {backend} backend...")

    if GPU_ENABLED:
        vec = cuTfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    else:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    print(f"✅ TF-IDF vectors built: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, vec


# ================================================================
# ✅ Save / Load Utilities
# ================================================================
def save_vectorizer(vectorizer, path: str):
    """
    Save the vectorizer model to disk.
    """
    joblib.dump(vectorizer, path)
    print(f"💾 Saved TF-IDF vectorizer → {path}")


def load_vectorizer(path: str):
    """
    Load a previously saved TF-IDF vectorizer.
    """
    print(f"📂 Loading TF-IDF vectorizer from {path}")
    return joblib.load(path)
