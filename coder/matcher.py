from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from rapidfuzz import process, fuzz
from .text import normalize, tokens, simple_ratio

@dataclass
class CatalogEntry:
    code: str
    title: str
    description: str = ""


class Catalog:
    """
    Catalog of code entries (ISCO or ISIC). Provides fast exact title lookup,
    inverted token index for candidate selection, and a best_match scorer.
    """

    def __init__(self, entries: List[CatalogEntry]):
        self.entries = entries
        # normalized forms
        self.norm_titles = [normalize(e.title) for e in entries]
        self.norm_descs = [normalize(e.description) for e in entries]
        self.codes = [e.code for e in entries]

        # exact normalized title -> index
        self.title_index: Dict[str, int] = {}
        for i, t in enumerate(self.norm_titles):
            if t:
                self.title_index[t] = i

        # inverted index: token -> set(indices)
        self.token_index: Dict[str, set] = {}
        for i, t in enumerate(self.norm_titles):
            for tok in set(t.split()):
                if tok:
                    self.token_index.setdefault(tok, set()).add(i)

    def get_by_title_norm(self, title_norm: str) -> Optional[CatalogEntry]:
        i = self.title_index.get(title_norm)
        return self.entries[i] if i is not None else None

    def candidates(self, query: str, max_candidates: int = 200) -> List[int]:
        q_tokens = set(normalize(query).split())
        idxs = set()
        for tok in q_tokens:
            if tok in self.token_index:
                idxs |= self.token_index[tok]
        if not idxs:
            idxs = set(range(len(self.entries)))
        # return up to max_candidates indexes
        return list(list(idxs)[:max_candidates])

    def best_match(self, query: str, max_candidates: int = 200) -> Tuple[Optional[CatalogEntry], float]:
        if not query or not query.strip():
            return None, 0.0
        qn = normalize(query)

        # 1) exact normalized title
        if qn in self.title_index:
            return self.entries[self.title_index[qn]], 1.0

        # 2) containment heuristic (title within query or query within title)
        best_i = None
        best_s = 0.0
        for i, t in enumerate(self.norm_titles):
            if t and (t in qn or qn in t):
                s = min(len(qn), len(t)) / max(len(qn), len(t))
                if s > best_s:
                    best_s, best_i = s, i

        # 3) token candidate scoring
        cands = self.candidates(query, max_candidates=max_candidates)
        for i in cands:
            s_title = simple_ratio(query, self.norm_titles[i])
            s_desc = simple_ratio(query, self.norm_descs[i]) if self.norm_descs[i] else 0.0
            s = max(s_title, 0.7 * s_title + 0.3 * s_desc)
            if s > best_s:
                best_s, best_i = s, i

        if best_i is None:
            return None, 0.0
        return self.entries[best_i], float(best_s)