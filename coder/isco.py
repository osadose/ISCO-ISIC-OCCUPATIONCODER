import pandas as pd
from typing import Tuple, Optional
from .matcher import Catalog, CatalogEntry
from .text import normalize


def build_query(row: pd.Series, title_fields, description_fields, synonyms: dict) -> str:
    parts = []
    for f in title_fields + description_fields:
        if f in row and pd.notna(row[f]):
            parts.append(str(row[f]))
    q = " ".join(parts)
    qn = normalize(q)
    # apply synonyms
    for src, dst in synonyms.items():
        qn = qn.replace(normalize(src), normalize(dst))
    return qn


def match_isco(row: pd.Series, catalog: Catalog, title_fields, description_fields, synonyms: dict, max_candidates: int) -> Tuple[Optional[CatalogEntry], float]:
    # Try exact normalized title using title fields with synonyms applied
    for f in title_fields:
        if f in row and pd.notna(row[f]):
            raw = str(row[f])
            tn = normalize(raw)
            for src, dst in synonyms.items():
                tn = tn.replace(normalize(src), normalize(dst))
            exact = catalog.get_by_title_norm(tn) if hasattr(catalog, "get_by_title_norm") else None
            if exact:
                return exact, 1.0

    # Else, fuzzy on the combined fields
    query = build_query(row, title_fields, description_fields, synonyms)
    entry, score = catalog.best_match(query, max_candidates=max_candidates)
    return entry, float(score)
