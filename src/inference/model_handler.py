"""
Model lifecycle management for Hugging Face pipelines (sentiment-analysis, image-classification).

The ModelHandler loads the pipeline exactly once per Python worker, provides
predict_batch with CUDA/MPS OOM recovery, and uses torch.no_grad() for inference.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


def _resolve_device(device: Optional[Union[int, str]]) -> Union[int, str]:
    """Resolve device: explicit value, else CUDA > MPS > CPU."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return -1


class ModelHandler:
    """
    Loads a Hugging Face pipeline once and reuses it for batched inference.

    Supports task="sentiment-analysis" (text) or task="image-classification" (image paths).
    CUDA/MPS OOM is handled by halving batch size and retrying. Supports CPU fallback
    when CUDA/MPS is unavailable (e.g., local tests).
    """

    _pipelines: Dict[tuple, Any] = {}  # (task, model_name) -> pipeline

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        task: str = "image-classification",
        device: Optional[Union[int, str]] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.task = task
        self.device = _resolve_device(device)
        self.batch_size = batch_size

    def _load_pipeline(self) -> Any:
        """Load and cache the Hugging Face pipeline once per (task, model_name)."""
        cache_key = (self.task, self.model_name)
        if cache_key in ModelHandler._pipelines:
            return ModelHandler._pipelines[cache_key]

        from transformers import pipeline

        pipeline_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
        }
        if self.task == "sentiment-analysis":
            pipeline_kwargs["truncation"] = True

        pipe = pipeline(
            self.task,
            **pipeline_kwargs,
        )
        ModelHandler._pipelines[cache_key] = pipe
        logger.info(
            "Loaded %s pipeline model=%s device=%s batch_size=%s",
            self.task,
            self.model_name,
            self.device,
            self.batch_size,
        )
        return pipe

    def predict_batch(self, inputs: List[str]) -> List[str]:
        """
        Run inference on a list of inputs (text strings or image file paths).

        Returns a list of JSON strings like {"label": "...", "score": 0.99}.
        For image-classification: top-1 prediction per image. Catches CUDA/MPS OOM,
        clears cache, retries with halved batch size. Corrupted/missing images
        yield {"label": "ERROR", "score": 0.0} with a log warning.
        """
        if not inputs:
            return []

        pipe = self._load_pipeline()
        results: List[str] = []
        current_batch_size = self.batch_size
        i = 0

        while i < len(inputs):
            batch = inputs[i : i + current_batch_size]
            try:
                with torch.no_grad():
                    raw = pipe(batch)
                for r in raw:
                    if self.task == "image-classification":
                        # raw is list of lists (one per image); take top-1
                        if isinstance(r, list) and len(r) > 0:
                            top = r[0]
                            results.append(json.dumps({"label": top["label"], "score": top["score"]}))
                        else:
                            results.append(json.dumps({"label": "ERROR", "score": 0.0}))
                    else:
                        results.append(json.dumps({"label": r["label"], "score": r["score"]}))
                i += len(batch)
                current_batch_size = self.batch_size
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "cuda" in err_msg or "mps" in err_msg:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                        if hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(
                        "GPU OOM, retrying with batch_size=%s", current_batch_size
                    )
                else:
                    raise
            except (OSError, Exception) as e:
                err_msg = str(e).lower()
                # Per-item image load errors: try one-by-one and mark failures
                if self.task == "image-classification" and (
                    "cannot identify image" in err_msg
                    or "no such file" in err_msg
                    or "decode" in err_msg
                ):
                    for inp in batch:
                        try:
                            with torch.no_grad():
                                out = pipe([inp])
                            if out and isinstance(out[0], list) and len(out[0]) > 0:
                                top = out[0][0]
                                results.append(json.dumps({"label": top["label"], "score": top["score"]}))
                            else:
                                results.append(json.dumps({"label": "ERROR", "score": 0.0}))
                                logger.warning("Image inference failed for path: %s", inp)
                        except Exception:
                            results.append(json.dumps({"label": "ERROR", "score": 0.0}))
                            logger.warning("Image load/inference failed for path: %s", inp)
                    i += len(batch)
                    current_batch_size = self.batch_size
                else:
                    raise

        return results
