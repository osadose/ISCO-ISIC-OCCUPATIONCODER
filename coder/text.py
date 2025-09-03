import re
import unicodedata
from typing import List, Set

_punct_re = re.compile(r"[^0-9a-zA-Z\s]+")
_ws_re = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize text to ASCII lowercase, remove punctuation, collapse whitespace."""
    if text is None:
        return ""
    t = str(text)
    # normalize unicode -> ascii where possible
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    t = _punct_re.sub(" ", t)
    t = _ws_re.sub(" ", t).strip()
    return t


def tokens(text: str) -> List[str]:
    t = normalize(text)
    return t.split() if t else []


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def token_overlap_score(a: str, b: str) -> float:
    A = token_set(a)
    B = token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = max(len(A), len(B))
    return inter / denom


def simple_ratio(a: str, b: str) -> float:
    """
    Lightweight similarity metric that returns 0..1.
    Uses token overlap and length ratio to approximate similarity.
    """
    a_n = normalize(a)
    b_n = normalize(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    overlap = token_overlap_score(a_n, b_n)
    # length ratio on normalized strings
    len_ratio = min(len(a_n), len(b_n)) / max(len(a_n), len(b_n))
    return 0.6 * overlap + 0.4 * len_ratio