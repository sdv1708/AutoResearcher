#!/usr/bin/env python
"""
CLI utility:

python -m autoresearcher.scripts.build_index \
    --pmc_dir data/pmc_xml \
    --cord_dir data/cord_json \
    --out_path indexes/medical.faiss
"""

import argparse
from pathlib import Path

from autoresearcher.data.loaders.cord19_loader import CORD19Loader
from autoresearcher.data.loaders.pmc_loader import PMCXMLLoader
from autoresearcher.embeddings.embedding_pipeline import EmbeddingPipeline


def iter_docs(pmc_dir: Path, cord_dir: Path):
    if pmc_dir:
        for p in pmc_dir.glob("*.xml"):
            yield PMCXMLLoader(p).load()
    if cord_dir:
        for p in cord_dir.glob("*.json"):
            yield CORD19Loader(p).load()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmc_dir", type=Path, default=None)
    parser.add_argument("--cord_dir", type=Path, default=None)
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    pipeline = EmbeddingPipeline()
    pipeline.ingest(iter_docs(args.pmc_dir, args.cord_dir))
    pipeline.save(args.out_path)
    print(f"✅  Saved index to {args.out_path}")

# chmod +x src/autoresearcher/scripts/build_index.py to make it executable

if __name__ == "__main__":
    main()
