"""
utils/text_cleaner.py
─────────────────────
Reusable text normalization and NLP preprocessing utilities for ResCheck.
Provides unified text cleaning, tokenization, and n-gram extraction.
"""

import re
import string
from typing import Optional, Set, List


# ──────────────────────────────────────────────────────────────────────────────
# Common English Stopwords (Curated & Fast, zero-dependency)
# ──────────────────────────────────────────────────────────────────────────────

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "also", "just"
}


def normalize_whitespace(text: str) -> str:
    """Replace multiple spaces, tabs, and newlines with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_for_nlp(text: str) -> str:
    """
    Unified text cleaning pipeline for both resumes and job descriptions.
    - Decodes ASCII artifacts safely
    - Strips URLs, emails, phone numbers
    - Normalizes special characters and punctuation
    - Preserves meaningful technical symbols (+, #, -, ., /)
    - Normalizes whitespace and drops noisy short lines
    """
    if not text:
        return ""

    # Decode unicode artifacts to nearest ascii equivalent
    cleaned = text.encode("ascii", errors="ignore").decode("ascii")

    # Replace invisible and non-standard spacing
    cleaned = re.sub(r"[\xa0\u200b\t\r]", " ", cleaned)

    # Strip URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Strip emails
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", cleaned)

    # Strip phone numbers
    cleaned = re.sub(r"(\+?\d[\d\s\-().]{7,}\d)", " ", cleaned)

    # Retain characters common in skills: +, #, -, ., /, (), &, comma
    cleaned = re.sub(r"[^\w\s,./\-+#()&]", " ", cleaned)

    # Consolidate multiple spaces
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)

    # Consolidate excessive newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Filter out empty or ultra-short noisy lines
    lines = [line.strip() for line in cleaned.split("\n") if len(line.strip()) > 2]
    return "\n".join(lines).strip()


def tokenize(text: str, remove_stop: bool = True) -> List[str]:
    """
    Extract lowercased alphanumeric tokens from text.
    Optionally filters common stopwords.
    """
    text_lower = text.lower()
    # Extract words or technical tokens (e.g. c++, python, ci/cd)
    raw_tokens = re.findall(r"\b[a-z0-9][a-z0-9+#./-]*\b", text_lower)
    if remove_stop:
        return [t for t in raw_tokens if len(t) > 1 and t not in STOPWORDS]
    return [t for t in raw_tokens if len(t) > 1]


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generate n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """Truncate text safely at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return (truncated[:last_space].strip() + "...") if last_space > 0 else truncated.strip()
