"""
Semantic matcher using sentence-transformers.
- Precompute embeddings for a catalog (list of titles).
- Cache embeddings to disk for fast reuse.
- Provide best-match among catalog (cosine similarity).
"""

import os
import json
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

CACHE_DIR = "coder/data/embeddings_cache"  # adjust if you prefer

class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Instantiate model. device: 'cpu' or 'cuda' or None (auto).
        """
        kwargs = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model_name = model_name
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _cache_path(self, name: str):
        # name: a short identifier for the catalog e.g. "isco_08"
        safe = name.replace(" ", "_").lower()
        return os.path.join(CACHE_DIR, f"{safe}__{self.model_name.replace('/', '_')}.npz")

    def build_and_cache(self, catalog_titles: List[str], catalog_name: str):
        """Encode catalog titles and save embeddings + metadata to disk."""
        if not catalog_titles:
            raise ValueError("catalog_titles must be non-empty")
        cache_p = self._cache_path(catalog_name)
        # encode in batches (sentence-transformers handles batching)
        embeddings = self.model.encode(catalog_titles, convert_to_tensor=True, show_progress_bar=True)
        # convert to numpy for saving
        emb_np = embeddings.cpu().numpy()
        # save titles & embeddings
        np.savez_compressed(cache_p, titles=np.array(catalog_titles, dtype=object), embeddings=emb_np)
        return cache_p

    def load_cached(self, catalog_name: str):
        p = self._cache_path(catalog_name)
        if not os.path.exists(p):
            return None, None
        data = np.load(p, allow_pickle=True)
        titles = data["titles"].tolist()
        embeddings = data["embeddings"]
        return titles, embeddings

    def best_semantic(self, query: str, cached_titles: List[str], cached_embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Return top_k (index, similarity_score) pairs (score in 0..1) for query vs cached_embeddings.
        Uses cosine similarity via sentence-transformers util.cos_sim (works with numpy/torch).
        """
        if not query or not cached_titles or cached_embeddings is None:
            return []
        q_emb = self.model.encode([query], convert_to_tensor=True)
        # cached_embeddings -> convert to tensor inside util.cos_sim
        scores = util.cos_sim(q_emb, cached_embeddings)[0]  # shape (n,)
        # get top_k indices
        values, indices = scores.topk(k=min(top_k, scores.shape[0]))
        results = []
        for v, idx in zip(values, indices):
            results.append((int(idx.cpu().item()), float(v.cpu().item())))
        return results