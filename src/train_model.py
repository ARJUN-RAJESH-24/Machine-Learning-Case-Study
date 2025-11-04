from typing import Any
import warnings
import sys

warnings.filterwarnings("ignore")

# ==================================================
# ✅ GPU Detection (via CuPy)
# ==================================================
def detect_gpu() -> bool:
    """Check if a CUDA-capable GPU is available."""
    try:
        import cupy
        gpu_count = cupy.cuda.runtime.getDeviceCount()
        if gpu_count > 0:
            props = cupy.cuda.runtime.getDeviceProperties(0)
            name = props["name"].decode()
            print(f"⚡ GPU detected: {name} (CUDA {cupy.cuda.runtime.runtimeGetVersion() / 1000:.1f})")
            return True
    except Exception:
        pass
    print("⚠️ No GPU detected — running on CPU.")
    return False


# ==================================================
# ✅ GPU / CPU Imports
# ==================================================
GPU_AVAILABLE = detect_gpu()

# Try importing cuML models if GPU available
if GPU_AVAILABLE:
    try:
        from cuml.linear_model import LogisticRegression as cuLogisticRegression
        from cuml.svm import SVC as cuSVC
    except Exception as e:
        print(f"⚠️ cuML import failed: {e}")
        cuLogisticRegression = None
        cuSVC = None
        GPU_AVAILABLE = False
else:
    cuLogisticRegression = None
    cuSVC = None

# CPU fallbacks
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Optional GPU-capable frameworks
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ==================================================
# ✅ Model Factory Function
# ==================================================
def get_model(model_key: str, random_state: int = 42) -> Any:
    """
    Dynamically returns the best available model:
    - GPU version if CUDA/cuML/LightGBM/XGBoost GPU available
    - CPU fallback otherwise
    """
    mk = model_key.lower()

    # --- Logistic Regression ---
    if mk in ("lr", "logreg", "logistic"):
        if GPU_AVAILABLE and cuLogisticRegression is not None:
            print("⚡ Using GPU Logistic Regression (cuML)")
            return cuLogisticRegression(
                max_iter=1000,
                C=1.0,
                fit_intercept=True,
                tol=1e-4,
                verbose=0,
            )
        else:
            print("🧠 Using CPU Logistic Regression (sklearn)")
            return LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                solver="saga",
                class_weight="balanced",
                random_state=random_state,
            )

    # --- Support Vector Machine ---
    if mk in ("svm", "linearsvm", "linsvm"):
        if GPU_AVAILABLE and cuSVC is not None:
            print("⚡ Using GPU SVM (cuML)")
            return cuSVC(
                kernel="rbf",
                C=1.0,
                probability=True,
            )
        else:
            print("🧠 Using CPU LinearSVC (sklearn)")
            return LinearSVC(
                class_weight="balanced",
                random_state=random_state,
            )

    # --- LightGBM ---
    if mk in ("lgbm", "lightgbm"):
        if LGBMClassifier is None:
            raise ImportError("LightGBM not installed. Please install via `pip install lightgbm`.")
        device_type = "gpu" if GPU_AVAILABLE else "cpu"
        print(f"⚡ Using LightGBM on {device_type.upper()}")
        return LGBMClassifier(
            device_type=device_type,
            boosting_type="gbdt",
            n_estimators=500,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=random_state,
        )

    # --- XGBoost ---
    if mk in ("xgb", "xgboost"):
        if XGBClassifier is None:
            raise ImportError("XGBoost not installed. Please install via `pip install xgboost`.")
        tree_method = "gpu_hist" if GPU_AVAILABLE else "hist"
        predictor = "gpu_predictor" if GPU_AVAILABLE else "cpu_predictor"
        print(f"⚡ Using XGBoost ({'GPU' if GPU_AVAILABLE else 'CPU'})")
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            tree_method=tree_method,
            predictor=predictor,
            random_state=random_state,
        )

    # --- Unknown Model ---
    raise ValueError(f"❌ Unknown model key: {model_key}")
