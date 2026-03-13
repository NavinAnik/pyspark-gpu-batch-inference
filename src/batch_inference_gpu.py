"""
PySpark batch inference script for GPU-accelerated sentiment analysis.

Reads Parquet from a data lake, applies the sentiment UDF, and writes enriched
Parquet. Supports --generate-sample-data for local testing without a data lake.
"""

from __future__ import annotations

import argparse
from typing import Optional
import os
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.config import SparkConfig, SparkSessionBuilder
from src.inference.udf import predict_sentiment_udf
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Default column name for review text (Amazon Product Reviews schema)
DEFAULT_TEXT_COL = "review_body"

# Sample reviews for --generate-sample-data
SAMPLE_REVIEWS = [
    "This product exceeded my expectations. Highly recommend!",
    "Terrible quality, would not buy again.",
    "It's okay for the price, nothing special.",
    "Absolutely love it! Best purchase I've made.",
    "Broke after one week. Very disappointed.",
    "Works as described. Fast shipping.",
    "Not worth the money. Save your cash.",
    "Great value and quality. Will buy again.",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated batch sentiment inference on Parquet data."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=False,
        help="Path to input Parquet (local, s3, gs, or abfs).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=False,
        help="Path to output Parquet.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=DEFAULT_TEXT_COL,
        help="Column containing review text.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Hugging Face model for sentiment analysis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size per Arrow batch.",
    )
    parser.add_argument(
        "--max-records-per-batch",
        type=int,
        default=512,
        help="Arrow maxRecordsPerBatch (OOM prevention).",
    )
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Create synthetic Parquet and run inference (for local testing).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device ID (-1 for CPU). Default: auto (cuda if available).",
    )
    return parser.parse_args()


def _generate_sample_data(spark: SparkSession, output_dir: str) -> str:
    """Create a synthetic Parquet dataset with ~10k review-like rows."""
    import random

    rows = []
    for i in range(10_000):
        text = random.choice(SAMPLE_REVIEWS)
        rows.append({"review_id": str(i), DEFAULT_TEXT_COL: text})

    df = spark.createDataFrame(rows)
    path = os.path.join(output_dir, "sample_input")
    df.write.mode("overwrite").parquet(path)
    logger.info("Generated sample Parquet at %s with %s rows", path, len(rows))
    return path


def run(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    text_col: str = DEFAULT_TEXT_COL,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 32,
    max_records_per_batch: int = 512,
    device: Optional[int] = None,
) -> None:
    """Execute the batch inference pipeline."""
    t0 = time.perf_counter()

    df = spark.read.parquet(input_path)
    n = df.count()
    logger.info("Read %s rows from %s", n, input_path)

    spark.conf.set(
        "spark.sql.execution.arrow.maxRecordsPerBatch", str(max_records_per_batch)
    )

    udf_fn = predict_sentiment_udf(
        model_name=model_name, batch_size=batch_size, device=device
    )
    result = df.withColumn("sentiment", udf_fn(F.col(text_col)))

    t1 = time.perf_counter()
    result.write.mode("overwrite").parquet(output_path)
    t2 = time.perf_counter()

    logger.info(
        "Wrote %s rows to %s (inference ~%.1fs, write ~%.1fs)",
        n,
        output_path,
        t1 - t0,
        t2 - t1,
    )


def main() -> None:
    args = _parse_args()

    config = SparkConfig(
        arrow_max_records_per_batch=args.max_records_per_batch,
    )
    builder = SparkSessionBuilder(config=config)
    spark = builder.build()

    if args.generate_sample_data:
        base = args.output_path or "./data"
        input_path = _generate_sample_data(spark, base)
        output_path = os.path.join(base, "sample_output")
    else:
        if not args.input_path or not args.output_path:
            raise SystemExit("--input-path and --output-path required (or use --generate-sample-data)")
        input_path = args.input_path
        output_path = args.output_path

    run(
        spark=spark,
        input_path=input_path,
        output_path=output_path,
        text_col=args.text_col,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_records_per_batch=args.max_records_per_batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
