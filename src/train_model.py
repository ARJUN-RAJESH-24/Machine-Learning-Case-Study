from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear', probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier()
    }

def train_and_save_models(X_train, y_train, model_dir='models/'):
    models = get_models()
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f"{model_dir}{name.replace(' ', '_').lower()}.pkl")
