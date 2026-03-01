# src/train_model.py
import warnings
import multiprocessing as mp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

N_JOBS = max(1, mp.cpu_count() - 1)

def get_model(model_key: str, random_state: int = 42, **kwargs):
    mk = model_key.lower()
    if mk == "lr":
        return LogisticRegression(max_iter=1000, solver="saga", n_jobs=N_JOBS, random_state=random_state, class_weight="balanced", **kwargs)
    if mk == "svm":
        return LinearSVC(max_iter=2000, random_state=random_state, class_weight="balanced", **kwargs)
    if mk == "lgbm":
        return LGBMClassifier(n_estimators=kwargs.get("n_estimators", 200), learning_rate=kwargs.get("learning_rate", 0.1), num_leaves=kwargs.get("num_leaves", 31), max_depth=kwargs.get("max_depth", -1), n_jobs=N_JOBS, random_state=random_state, class_weight="balanced", verbosity=-1, force_col_wise=True)
    if mk == "xgb":
        return XGBClassifier(n_estimators=kwargs.get("n_estimators", 200), learning_rate=kwargs.get("learning_rate", 0.1), max_depth=kwargs.get("max_depth", 6), n_jobs=N_JOBS, random_state=random_state, verbosity=0, tree_method="hist")
    raise ValueError("Unknown model key")

