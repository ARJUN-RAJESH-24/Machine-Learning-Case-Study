# src/preprocess.py
import re
import pandas as pd
from typing import Iterable, Union
from text_unidecode import unidecode

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_REGEX = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_REGEX = re.compile(r"#[\w_]+")
NUM_REGEX = re.compile(r"\b\d+\b")
WHITESPACE_REGEX = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = unidecode(str(text)).lower()
    t = URL_REGEX.sub(" ", t)
    t = MENTION_REGEX.sub(" ", t)
    t = HASHTAG_REGEX.sub(" ", t)
    t = NUM_REGEX.sub(" <num> ", t)
    t = re.sub(r"[^a-z0-9\s<>()\[\]{}!?%/\\:'\-]", " ", t)
    t = WHITESPACE_REGEX.sub(" ", t).strip()
    return t

def normalize_corpus(texts: Union[Iterable[str], pd.Series]) -> pd.Series:
    """
    Accepts list, iterable, or pd.Series; returns pd.Series of cleaned strings.
    Ensures downstream `.str` and `.astype` usage are safe.
    """
    if isinstance(texts, pd.Series):
        seq = texts.fillna("").astype(str).tolist()
    else:
        seq = [("" if t is None else str(t)) for t in texts]

    cleaned = [normalize_text(x) for x in seq]
    return pd.Series(cleaned, dtype="string")

