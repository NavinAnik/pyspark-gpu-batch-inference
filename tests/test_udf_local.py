"""
Local test for the inference UDF (sentiment and image classification) using CPU fallback (no GPU required).
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# PySpark/Py4J support Python 3.8-3.12. Python 3.14+ causes 'JavaPackage' object is not callable.
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="PySpark requires Python 3.8-3.12. Use pyenv or a venv with Python 3.11/3.12.",
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.inference.model_handler import _resolve_device
from src.inference.udf import predict_sentiment_udf, predict_udf


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
    """UDF returns StringType with valid JSON (sentiment task)."""
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


def test_image_classification_udf_schema(spark):
    """Image classification UDF returns StringType with valid JSON (label, score)."""
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (224, 224), color="red")
        img.save(f.name)
        image_path = f.name

    try:
        udf_fn = predict_udf(
            task="image-classification",
            model_name="google/vit-base-patch16-224",
            device=-1,
            batch_size=2,
        )
        df = spark.createDataFrame([(image_path,)], ["image_path"])
        result = df.withColumn("prediction", udf_fn(F.col("image_path")))

        rows = result.collect()
        assert len(rows) == 1
        parsed = json.loads(rows[0]["prediction"])
        assert "label" in parsed
        assert "score" in parsed
        assert 0 <= parsed["score"] <= 1
    finally:
        Path(image_path).unlink(missing_ok=True)


def test_resolve_device_mps():
    """_resolve_device returns 'mps' when CUDA is unavailable and MPS is available."""
    import torch

    fake_mps = type("FakeMPS", (), {"is_available": lambda self: True})()
    with patch("src.inference.model_handler.torch.cuda.is_available", return_value=False), patch.object(
        torch.backends, "mps", fake_mps
    ):
        assert _resolve_device(None) == "mps"
