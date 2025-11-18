"""
Compute OpenAI embeddings for the Doshi/Hauser writers stories dataset and save
them to a CSV with metadata + emb_0..emb_(d-1) columns.

Usage (from repo root):
  export OPENAI_API_KEY=...
  python3 doshi_embed.py \
    --csv doshi_hauser_writers_stories.csv \
    --text-col writer_story \
    --out doshi_embeddings.csv \
    --model text-embedding-3-large
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import shutil


DATA_URL_ID = "1EBN6UD5hYwVVwWUJ9xY3l6bkru43zQwA"
DEFAULT_IN = "doshi_hauser_writers_stories.csv"
DEFAULT_OUT = "doshi_embeddings.csv"


def download_dataset(path: Path) -> None:
    if path.exists():
        return
    if not shutil.which("gdown"):
        raise FileNotFoundError(
            f"{path} not found. Install gdown or place the CSV manually."
        )
    cmd = ["gdown", DATA_URL_ID, "-O", str(path)]
    subprocess.check_call(cmd)


def get_openai_embeddings(texts: List[str], model: str = "text-embedding-3-large") -> np.ndarray:
    """Compute embeddings via OpenAI (requires OPENAI_API_KEY)."""
    import openai

    client_cls = getattr(openai, "OpenAI", None)
    if client_cls:
        client = client_cls()

        def _embed(batch):
            resp = client.embeddings.create(model=model, input=batch)
            return [item.embedding for item in resp.data]
    else:

        def _embed(batch):
            resp = openai.Embeddings.create(model=model, input=batch)
            return [item["embedding"] for item in resp["data"]]

    embeddings: List[List[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        embeddings.extend(_embed(texts[i : i + batch_size]))
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_IN, help="Input dataset CSV path")
    parser.add_argument("--text-col", default="writer_story", help="Text column name")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output embeddings CSV path")
    parser.add_argument(
        "--model", default="text-embedding-3-large", help="OpenAI embedding model"
    )
    args = parser.parse_args()

    data_path = Path(args.csv)
    if not data_path.exists():
        download_dataset(data_path)

    df = pd.read_csv(data_path)
    if args.text_col != "text_to_embed" and args.text_col in df.columns:
        df = df.rename(columns={args.text_col: "text_to_embed"})
        text_col = "text_to_embed"
    else:
        text_col = args.text_col

    texts = df[text_col].astype(str).tolist()
    embeddings = get_openai_embeddings(texts, model=args.model)
    dim = embeddings.shape[1]
    emb_cols = {f"emb_{i}": embeddings[:, i] for i in range(dim)}
    out_df = df.copy()
    for k, v in emb_cols.items():
        out_df[k] = v

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote embeddings CSV to {out_path} (dim={dim}, rows={len(out_df)})")


if __name__ == "__main__":
    main()
