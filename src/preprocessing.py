"""
NLP Preprocessing Module for Fake News Detection.

Performs text normalization, regex cleaning, publisher watermark stripping
to prevent data leakage, and stopword removal with preservation of negation words.
"""

import re
import html
import unicodedata
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
try:
    _STOPWORDS_SET = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _STOPWORDS_SET = set(stopwords.words('english'))

# Crucial negation and contrastive terms that modify meaning and should NOT be removed
NEGATION_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody', 'nothing',
    'nowhere', 'hardly', 'scarcely', 'barely', 'without', 'against',
    'cannot', "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
    "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't"
}

# Effective stopwords: standard english minus negation words
EFFECTIVE_STOPWORDS = _STOPWORDS_SET - NEGATION_WORDS

# Regex patterns for cleaning
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_PATTERN = re.compile(r'<.*?>')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
# Remove common wire service/publisher watermarks that cause dataset leakage
# e.g., "WASHINGTON (Reuters) - " or "NEW YORK (AP) — "
PUBLISHER_LEAKAGE_PATTERN = re.compile(r'^[A-Z\s,./-]+?\((?:reuters|ap|afp|bloomberg)\)\s*[-—–:]\s*', flags=re.IGNORECASE)
REUTERS_PATTERN = re.compile(r'\b(?:reuters|associated press)\b', flags=re.IGNORECASE)
PUNCT_SPECIAL_PATTERN = re.compile(r'[^a-zA-Z0-9\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_text(text: str) -> str:
    """
    Clean and normalize input text for NLP and TF-IDF feature extraction.
    
    Steps:
      1. Handle null or non-string input.
      2. Normalize unicode characters (NFKD).
      3. Unescape HTML entities.
      4. Remove wire-service publisher headers (avoiding data leakage).
      5. Remove URLs, emails, and HTML tags.
      6. Convert to lowercase.
      7. Remove publisher brand artifacts.
      8. Strip unwanted special characters, keeping alphanumeric tokens and whitespace.
      9. Remove non-negation stopwords.
      10. Normalize multiple whitespaces into a single space.
    
    Returns:
      Cleaned text string.
    """
    if text is None:
        return ""
    
    text = str(text).strip()
    if not text:
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)

    # Unescape HTML entities
    text = html.unescape(text)

    # Remove leading wire service tags to prevent leakage
    text = PUBLISHER_LEAKAGE_PATTERN.sub('', text)

    # Remove URLs, emails, and HTML tags
    text = URL_PATTERN.sub(' ', text)
    text = EMAIL_PATTERN.sub(' ', text)
    text = HTML_TAG_PATTERN.sub(' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove lingering publisher watermark names
    text = REUTERS_PATTERN.sub(' ', text)

    # Remove unwanted special characters while preserving alphanumeric tokens
    text = PUNCT_SPECIAL_PATTERN.sub(' ', text)

    # Tokenize and filter out non-negation stopwords
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in EFFECTIVE_STOPWORDS and len(word) > 1]

    # Join and strip excess whitespace
    cleaned = ' '.join(filtered_tokens).strip()
    return cleaned


if __name__ == "__main__":
    sample = "WASHINGTON (Reuters) - The president said he will NOT approve the new spending bill! Check https://example.com"
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))
