"""
utils/text_cleaner.py
─────────────────────
Reusable text normalization and utility functions.
These are standalone helpers that can be imported anywhere in the project.
"""

import re
import string
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Common English stopwords (lightweight, no NLTK dependency)
# ──────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "it", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "what", "which", "who", "whom",
    "when", "where", "why", "how", "not", "no", "so", "if", "as", "such",
    "also", "more", "most", "other", "some", "any", "both", "each",
    "few", "more", "than", "too", "very", "just", "only",
}


def normalize_whitespace(text: str) -> str:
    """Replace multiple spaces/tabs/newlines with single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords from text."""
    words = text.lower().split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered)


def remove_punctuation(text: str, keep: str = "") -> str:
    """
    Remove punctuation from text.

    Args:
        text: Input text.
        keep: String of punctuation characters to preserve. E.g., keep="+-#"

    Returns:
        Text with punctuation removed (except kept chars).
    """
    to_remove = "".join(c for c in string.punctuation if c not in keep)
    translator = str.maketrans("", "", to_remove)
    return text.translate(translator)


def tokenize(text: str) -> list[str]:
    """
    Simple word tokenizer.

    Args:
        text: Input text.

    Returns:
        List of lowercase tokens (no punctuation, no stopwords).
    """
    text = text.lower()
    text = remove_punctuation(text, keep="-")
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def extract_bigrams(tokens: list[str]) -> list[str]:
    """
    Generate bigrams from a token list.

    Args:
        tokens: List of word tokens.

    Returns:
        List of 'word1 word2' bigram strings.
    """
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """
    Truncate text to a maximum number of characters, ending at a word boundary.

    Args:
        text: Input text.
        max_chars: Maximum character limit.

    Returns:
        Truncated text.
    """
    if len(text) <= max_chars:
        return text

    # Find the last space before the limit
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space].strip() + "…" if last_space > 0 else truncated.strip()


def clean_for_display(text: str, max_lines: int = 50) -> str:
    """
    Prepare text for display in UI (trim long texts, clean visual noise).

    Args:
        text: Input text.
        max_lines: Maximum number of lines to return.

    Returns:
        Display-ready text string.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines[:max_lines])


def highlight_keywords(text: str, keywords: list[str], tag: str = "**") -> str:
    """
    Wrap occurrences of keywords in text with a given tag (e.g., Markdown bold).

    Args:
        text: Input text.
        keywords: List of keywords to highlight.
        tag: Wrapping tag (default: Markdown bold '**').

    Returns:
        Text with keywords highlighted.
    """
    for kw in sorted(keywords, key=len, reverse=True):  # Longer first to avoid partial wrapping
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f"{tag}{kw}{tag}", text)
    return text


def count_words(text: str) -> int:
    """Return word count of text."""
    return len(text.split())


def extract_email(text: str) -> Optional[str]:
    """Extract the first email address found in text."""
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    """Extract the first phone number found in text."""
    match = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text)
    return match.group(0).strip() if match else None


def split_into_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter using punctuation.

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]