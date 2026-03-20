"""
PySpark batch inference script for GPU-accelerated inference (image classification or sentiment).

Reads Parquet from a data lake, applies the inference UDF, and writes enriched
Parquet. Supports --generate-sample-data for local testing (sentiment only; for
image classification use real data from scripts/download_and_prepare.py).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Union

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

try:
    from pyspark.errors import UnsupportedOperationException as PySparkUnsupportedOperation
except ImportError:
    PySparkUnsupportedOperation = Exception  # older PySpark

from src.config import SparkConfig, SparkSessionBuilder
from src.inference.udf import predict_udf
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Default column name for image path or text (schema from download_and_prepare / product reviews)
DEFAULT_INPUT_COL = "image_path"

# Sample reviews for --generate-sample-data (sentiment task only)
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


def _parse_device(value: str) -> Union[int, str]:
    """Parse --device: integer GPU ID, or 'mps'/'cpu' as string."""
    try:
        return int(value)
    except ValueError:
        return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated batch inference on Parquet data (image classification or sentiment)."
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
        "--input-col",
        type=str,
        default=DEFAULT_INPUT_COL,
        help="Column containing image path or review text (default: image_path).",
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
        help="Inference batch size per Arrow batch (default 64 for M4 Pro / 24GB).",
    )
    parser.add_argument(
        "--max-records-per-batch",
        type=int,
        default=1024,
        help="Arrow maxRecordsPerBatch (default 1024 for 24GB RAM).",
    )
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Create synthetic Parquet and run inference (sentiment task only; for image use real data).",
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
    return parser.parse_args()


def _generate_sample_data(spark: SparkSession, output_dir: str, task: str) -> str:
    """Create synthetic Parquet: text rows for sentiment, or dummy image paths for image task."""
    import random

    if task == "sentiment-analysis":
        rows = []
        for i in range(10_000):
            text = random.choice(SAMPLE_REVIEWS)
            rows.append({"review_id": str(i), "review_body": text})
        df = spark.createDataFrame(rows)
    else:
        # Dummy image paths (inference will return ERROR unless files exist)
        rows = [{"image_id": str(i), "image_path": f"/nonexistent/demo_{i}.jpg", "label": ""} for i in range(100)]
        df = spark.createDataFrame(rows)

    path = os.path.join(output_dir, "sample_input")
    df.write.mode("overwrite").parquet(path)
    logger.info("Generated sample Parquet at %s with %s rows", path, len(rows))
    return path


def run(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    input_col: str = DEFAULT_INPUT_COL,
    task: str = "image-classification",
    model_name: str = "google/vit-base-patch16-224",
    batch_size: int = 32,
    max_records_per_batch: int = 512,
    device: Optional[Union[int, str]] = None,
    limit: Optional[int] = None,
) -> None:
    """Execute the batch inference pipeline."""
    t0 = time.perf_counter()

    df = spark.read.parquet(input_path)
    if limit is not None:
        df = df.limit(limit)
        logger.info("Limiting to first %s rows (--limit)", limit)
    n = df.count()
    logger.info("Read %s rows from %s", n, input_path)

    spark.conf.set(
        "spark.sql.execution.arrow.maxRecordsPerBatch", str(max_records_per_batch)
    )

    udf_fn = predict_udf(
        model_name=model_name,
        task=task,
        batch_size=batch_size,
        device=device,
    )
    result = df.withColumn("prediction", udf_fn(F.col(input_col)))

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


_JAVA_INCOMPATIBLE_MSG = """
PySpark failed due to Java compatibility. Spark 3.5 supports Java 11/17 only, not Java 21/22+.

Options:
  1. Use Java 11 or 17: set JAVA_HOME and rerun this script
  2. Run standalone inference (no Spark): python scripts/run_inference_local.py --input-path %s --output-path %s --input-col %s --task %s
"""

_JAVA_GETSUBJECT_MSG = """
PySpark failed: getSubject is not supported. This happens with Java 21+ and Hadoop.

Options:
  1. Use Java 11 or 17: set JAVA_HOME to a Java 11/17 installation, then rerun
  2. Run without Spark (same inference, no JVM):
     python scripts/run_inference_local.py --input-path %s --output-path %s --input-col %s --task %s
"""


def main() -> None:
    args = _parse_args()

    # Default model per task when user does not pass --model-name
    if args.task == "sentiment-analysis" and args.model_name == "google/vit-base-patch16-224":
        args.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    if args.generate_sample_data and args.task == "image-classification":
        logger.warning(
            "generate-sample-data with image-classification creates dummy paths (predictions will be ERROR). "
            "For real image demo run: python scripts/download_and_prepare.py"
        )

    config = SparkConfig(
        arrow_max_records_per_batch=args.max_records_per_batch,
    )
    builder = SparkSessionBuilder(config=config)

    try:
        spark = builder.build()
    except TypeError as e:
        if "JavaPackage" in str(e) and "not callable" in str(e):
            inp = args.input_path or "<input-path>"
            out = args.output_path or "<output-path>"
            raise SystemExit(_JAVA_INCOMPATIBLE_MSG % (inp, out, args.input_col, args.task))
        raise

    if args.generate_sample_data:
        base = args.output_path or "./data"
        input_path = _generate_sample_data(spark, base, args.task)
        output_path = os.path.join(base, "sample_output")
    else:
        if not args.input_path or not args.output_path:
            raise SystemExit("--input-path and --output-path required (or use --generate-sample-data)")
        input_path = args.input_path
        output_path = args.output_path

    try:
        run(
            spark=spark,
            input_path=input_path,
            output_path=output_path,
            input_col=args.input_col,
            task=args.task,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_records_per_batch=args.max_records_per_batch,
            device=args.device,
            limit=args.limit,
        )
    except PySparkUnsupportedOperation as e:
        err_str = str(e).lower()
        if "getsubject" in err_str or "get_subject" in err_str:
            raise SystemExit(
                _JAVA_GETSUBJECT_MSG % (input_path, output_path, args.input_col, args.task)
            ) from e
        raise


if __name__ == "__main__":
    main()
