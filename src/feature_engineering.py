# src/feature_engineering.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

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
