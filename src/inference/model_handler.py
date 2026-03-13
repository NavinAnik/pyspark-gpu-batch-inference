"""
Model lifecycle management for Hugging Face sentiment-analysis pipeline.

The ModelHandler loads the pipeline exactly once per Python worker, provides
predict_batch with CUDA OOM recovery, and uses torch.no_grad() for inference.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class ModelHandler:
    """
    Loads the Hugging Face sentiment pipeline once and reuses it for batched inference.

    CUDA OOM is handled by halving batch size and retrying. Supports CPU fallback
    when CUDA is unavailable (e.g., local tests).
    """

    _pipeline: Optional[Any] = None

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[Union[int, str]] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)
        self.batch_size = batch_size

    def _load_pipeline(self) -> Any:
        """Load and cache the Hugging Face sentiment pipeline once per instance."""
        if ModelHandler._pipeline is not None:
            return ModelHandler._pipeline

        from transformers import pipeline

        ModelHandler._pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            truncation=True,
        )
        logger.info(
            "Loaded sentiment pipeline model=%s device=%s batch_size=%s",
            self.model_name,
            self.device,
            self.batch_size,
        )
        return ModelHandler._pipeline

    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Run sentiment analysis on a list of texts.

        Returns a list of JSON strings like {"label": "POSITIVE", "score": 0.9987}.
        Catches CUDA OOM, clears cache, retries with halved batch size.
        """
        if not texts:
            return []

        pipe = self._load_pipeline()
        results: List[str] = []
        current_batch_size = self.batch_size
        i = 0

        while i < len(texts):
            batch = texts[i : i + current_batch_size]
            try:
                with torch.no_grad():
                    raw = pipe(batch)
                for r in raw:
                    results.append(json.dumps({"label": r["label"], "score": r["score"]}))
                i += len(batch)
                current_batch_size = self.batch_size
            except RuntimeError as e:
                err_msg = str(e).lower()
                # Handles both torch.cuda.OutOfMemoryError and generic CUDA OOM messages
                if "out of memory" in err_msg or "cuda" in err_msg:
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(
                        "CUDA OOM, retrying with batch_size=%s", current_batch_size
                    )
                else:
                    raise

        return results
