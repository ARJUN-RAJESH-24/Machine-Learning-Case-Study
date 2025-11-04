# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (silently)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

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
