from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

try:
	from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
	LGBMClassifier = None  # type: ignore

try:
	from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
	XGBClassifier = None  # type: ignore


def get_model(model_key: str, random_state: int = 42) -> Any:
	mk = model_key.lower()
	if mk in ("lr", "logreg", "logistic"):
		return LogisticRegression(
			max_iter=1000,
			n_jobs=-1,
			solver="saga",
			class_weight="balanced",
			random_state=random_state,
		)
	if mk in ("svm", "linearsvm", "linsvm"):
		return LinearSVC(
			class_weight="balanced",
			random_state=random_state,
		)
	if mk in ("lgbm", "lightgbm"):
		if LGBMClassifier is None:
			raise ImportError("LightGBM is not installed")
		return LGBMClassifier(
			n_estimators=500,
			learning_rate=0.05,
			n_jobs=-1,
			random_state=random_state,
		)
	if mk in ("xgb", "xgboost"):
		if XGBClassifier is None:
			raise ImportError("XGBoost is not installed")
		return XGBClassifier(
			n_estimators=500,
			learning_rate=0.05,
			max_depth=8,
			subsample=0.9,
			colsample_bytree=0.9,
			reg_lambda=1.0,
		
eval_metric="logloss",
			tree_method="hist",
			random_state=random_state,
		)
	raise ValueError(f"Unknown model_key: {model_key}")
