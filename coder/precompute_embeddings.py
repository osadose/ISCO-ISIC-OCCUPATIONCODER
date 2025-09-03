"""
Utility CLI to precompute embeddings for ISCO/ISIC catalogs.
Run this once (or whenever catalogs change) to cache embeddings for faster processing.
"""

import argparse
import os
from coder.loader import load_isco_catalog, load_isic_catalog
from coder.ml_matcher import SemanticMatcher

def main():
    parser = argparse.ArgumentParser(description="Precompute catalog embeddings")
    parser.add_argument("--isco", required=False, help="Path to ISCO workbook")
    parser.add_argument("--isic", required=False, help="Path to ISIC workbook")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    args = parser.parse_args()

    sm = SemanticMatcher(model_name=args.model)

    if args.isco:
        print("Loading ISCO catalog...")
        isco_catalog = load_isco_catalog(args.isco)
        titles = [e.title for e in isco_catalog.entries]
        print(f"Encoding {len(titles)} ISCO titles...")
        p = sm.build_and_cache(titles, "isco_catalog")
        print("Cached ISCO embeddings to:", p)

    if args.isic:
        print("Loading ISIC catalog...")
        isic_catalog = load_isic_catalog(args.isic)
        titles = [e.title for e in isic_catalog.entries]
        print(f"Encoding {len(titles)} ISIC titles...")
        p = sm.build_and_cache(titles, "isic_catalog")
        print("Cached ISIC embeddings to:", p)

if __name__ == "__main__":
    main()