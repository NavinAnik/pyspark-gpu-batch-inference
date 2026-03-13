"""
Local test for the sentiment UDF using CPU fallback (no GPU required).
"""

import json
import sys

import pytest

# PySpark/Py4J support Python 3.8-3.12. Python 3.14+ causes 'JavaPackage' object is not callable.
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="PySpark requires Python 3.8-3.12. Use pyenv or a venv with Python 3.11/3.12.",
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Add project root to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.udf import predict_sentiment_udf


@pytest.fixture(scope="module")
def spark():
    try:
        return (
            SparkSession.builder
            .master("local[2]")
            .appName("test-udf")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "64")
            .getOrCreate()
        )
    except TypeError as e:
        if "JavaPackage" in str(e) and "not callable" in str(e):
            pytest.skip(
                "PySpark fails with Java 21/22. Use Java 11 or 17: "
                "https://adoptium.net or set JAVA_HOME to a Java 11/17 installation."
            )
        raise


def test_sentiment_udf_schema(spark):
    """UDF returns StringType with valid JSON."""
    udf_fn = predict_sentiment_udf(device=-1, batch_size=4)  # CPU fallback
    df = spark.createDataFrame(
        [("Great product!",), ("Terrible experience.",), ("It's okay.",)],
        ["text"],
    )
    result = df.withColumn("sentiment", udf_fn(F.col("text")))

    rows = result.collect()
    assert len(rows) == 3

    for row in rows:
        assert "sentiment" in row.asDict()
        parsed = json.loads(row["sentiment"])
        assert "label" in parsed
        assert "score" in parsed
        assert parsed["label"] in ("POSITIVE", "NEGATIVE")
        assert 0 <= parsed["score"] <= 1


def test_sentiment_udf_empty(spark):
    """Empty input produces empty output."""
    from pyspark.sql.types import StringType, StructField, StructType

    udf_fn = predict_sentiment_udf(device=-1, batch_size=4)
    schema = StructType([StructField("text", StringType())])
    df = spark.createDataFrame([], schema)
    result = df.withColumn("sentiment", udf_fn(F.col("text")))
    assert result.count() == 0
