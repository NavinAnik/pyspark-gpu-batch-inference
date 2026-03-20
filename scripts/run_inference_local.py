"""
Run inference on Parquet data without Spark (pandas + ModelHandler).

Supports image-classification (image paths) or sentiment-analysis (text).
Use when PySpark has environment issues (e.g., Python 3.14, Java 21/22).
Processes data in chunks to control memory usage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.inference.model_handler import ModelHandler


def _parse_device(value: str) -> Union[int, str]:
    """Parse --device: integer GPU ID, or 'mps'/'cpu' as string."""
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone inference (no Spark): image classification or sentiment."
    )
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
        "--input-col",
        type=str,
        default="image_path",
        help="Column containing image path or text (default: image_path).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("image-classification", "sentiment-analysis"),
        default="image-classification",
        help="Inference task (default: image-classification).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/vit-base-patch16-224",
        help="Hugging Face model (ViT for images, DistilBERT for sentiment).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (default 64 for M4 Pro / 24GB).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Rows per chunk (memory control; default 2048 for 24GB RAM).",
    )
    parser.add_argument(
        "--device",
        type=_parse_device,
        default=None,
        help="Device: integer GPU ID, 'mps' for Apple Silicon, -1 for CPU. Default: auto-detect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only first N rows (for quick demo). Omit to process all.",
    )
    args = parser.parse_args()

    if args.task == "sentiment-analysis" and args.model_name == "google/vit-base-patch16-224":
        args.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    if args.limit is not None:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} rows (--limit)")
    n = len(df)
    print(f"Loaded {n} rows. Column: {args.input_col}, task: {args.task}")

    handler = ModelHandler(
        model_name=args.model_name,
        task=args.task,
        batch_size=args.batch_size,
        device=args.device,
    )

    t0 = time.perf_counter()
    predictions = []
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        chunk = df.iloc[start:end]
        if args.task == "sentiment-analysis":
            inputs = chunk[args.input_col].astype(str).fillna("").tolist()
        else:
            inputs = chunk[args.input_col].astype(str).fillna("").tolist()
        preds = handler.predict_batch(inputs)
        predictions.extend(preds)
        print(f"  Processed {end}/{n} rows...")

    df = df.copy()
    df["prediction"] = predictions

    df.to_parquet(output_path, index=False)
    elapsed = time.perf_counter() - t0
    print(f"Wrote {n} rows to {output_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
