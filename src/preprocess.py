# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Iterable, List

from text_unidecode import unidecode


# Download required NLTK data (silently)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_REGEX = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_REGEX = re.compile(r"#[\w_]+")
NUM_REGEX = re.compile(r"\b\d+\b")
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - remove URLs, mentions, hashtags
      - keep only letters and spaces
      - lowercase, tokenize, remove stopwords, lemmatize
    """
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r'http\S+', ' ', text)           # urls
    text = re.sub(r'@\w+', ' ', text)              # mentions
    text = re.sub(r'#\w+', ' ', text)              # hashtags
    text = re.sub(r'&amp;', ' and ', text)         # HTML encoded ampersand
    text = re.sub(r'[^A-Za-z\s]', ' ', text)       # non-alpha chars
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.lower().split() if t not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def normalize_text(text: str) -> str:
	if text is None:
		return ""
	# Normalize unicode and lower-case
	t = unidecode(str(text)).lower()
	# Remove URLs, mentions, hashtags as tokens (keep hashtag text without '#')
	t = URL_REGEX.sub(" ", t)
	t = MENTION_REGEX.sub(" ", t)
	t = HASHTAG_REGEX.sub(" ", t)
	# Replace numbers with a special token
	t = NUM_REGEX.sub(" <num> ", t)
	# Remove non-word punctuation/symbols conservatively
	t = re.sub(r"[^a-z0-9\s<>()\[\]{}!?%/\\:'\-]", " ", t)
	# Collapse whitespace
	t = WHITESPACE_REGEX.sub(" ", t).strip()
	return t


def normalize_corpus(texts: Iterable[str]) -> List[str]:
	return [normalize_text(t) for t in texts]
