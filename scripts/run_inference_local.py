"""
Run sentiment inference on Parquet data without Spark (pandas + ModelHandler).

Use this when PySpark has environment issues (e.g., Python 3.14, Java 21/22).
Processes data in chunks to control memory usage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.inference.model_handler import ModelHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone sentiment inference (no Spark).")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input Parquet file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output Parquet file.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="review_body",
        help="Column containing text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows per chunk (memory control).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device (-1 for CPU). Default: auto.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    n = len(df)
    print(f"Loaded {n} rows. Column for text: {args.text_col}")

    handler = ModelHandler(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        batch_size=args.batch_size,
        device=args.device,
    )

    t0 = time.perf_counter()
    sentiments = []
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        chunk = df.iloc[start:end]
        texts = chunk[args.text_col].astype(str).fillna("").tolist()
        preds = handler.predict_batch(texts)
        sentiments.extend(preds)
        print(f"  Processed {end}/{n} rows...")

    df = df.copy()
    df["sentiment"] = sentiments

    df.to_parquet(output_path, index=False)
    elapsed = time.perf_counter() - t0
    print(f"Wrote {n} rows to {output_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
