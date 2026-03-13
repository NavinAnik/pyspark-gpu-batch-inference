"""
Pandas UDF for GPU-accelerated sentiment inference.

Uses the Iterator[Series] -> Iterator[Series] pattern so the model is loaded
once per Python worker (at first batch), then reused for all subsequent batches.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Union

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

from src.inference.model_handler import ModelHandler

logger = logging.getLogger(__name__)


def predict_sentiment_udf(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 32,
    device: Optional[Union[int, str]] = None,
):
    """
    Return a Pandas UDF that predicts sentiment for text batches.

    The UDF uses Iterator[Series] -> Iterator[Series] so the ModelHandler
    (and Hugging Face pipeline) is created once per partition and reused
    across Arrow batches. device=None means auto (cuda if available).
    """

    @pandas_udf(StringType())
    def _udf(text_series: Iterator[pd.Series]) -> Iterator[pd.Series]:
        handler: Optional[ModelHandler] = None
        for batch in text_series:
            if handler is None:
                handler = ModelHandler(
                    model_name=model_name,
                    batch_size=batch_size,
                    device=device,
                )
            texts = batch.astype(str).fillna("").tolist()
            predictions = handler.predict_batch(texts)
            yield pd.Series(predictions)

    return _udf
