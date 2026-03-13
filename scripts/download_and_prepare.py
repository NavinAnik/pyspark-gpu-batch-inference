"""
Download IMDB 50K dataset from Kaggle, convert to Parquet, and pre-download the model.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json.
Create token at: https://www.kaggle.com/settings (Account -> Create New Token)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KAGGLE_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    # Also check env vars
    import os
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def download_dataset() -> Path:
    """Download IMDB dataset from Kaggle and return path to extracted CSV."""
    if not check_kaggle_credentials():
        print(
            "Kaggle API not configured. Please:\n"
            "  1. Create token at https://www.kaggle.com/settings\n"
            "  2. Place kaggle.json at ~/.kaggle/kaggle.json\n"
            "     (or set KAGGLE_USERNAME and KAGGLE_KEY env vars)"
        )
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(RAW_DIR),
        unzip=True,
    )
    # Dataset extracts to IMDB Dataset.csv
    csv_path = RAW_DIR / "IMDB Dataset.csv"
    if not csv_path.exists():
        # Try alternative name
        csvs = list(RAW_DIR.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {RAW_DIR} after download")
        csv_path = csvs[0]
    return csv_path


def convert_to_parquet(csv_path: Path) -> Path:
    """Convert CSV to Parquet, renaming 'review' to 'review_body'."""
    import pandas as pd

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARQUET_DIR / "imdb_reviews.parquet"

    df = pd.read_csv(csv_path)
    if "review" in df.columns:
        df = df.rename(columns={"review": "review_body"})
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def download_model() -> None:
    """Pre-download and cache the sentiment model, run sanity check."""
    from transformers import pipeline

    print("Downloading model (this may take a minute)...")
    pipe = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        truncation=True,
    )
    out = pipe("This movie was fantastic!")
    print(f"Model loaded. Sanity check: {out}")
    print("Model cached and ready for inference.")


def main() -> None:
    print("Step 1: Downloading IMDB dataset from Kaggle...")
    csv_path = download_dataset()
    print(f"Downloaded to {csv_path}")

    print("\nStep 2: Converting to Parquet...")
    parquet_path = convert_to_parquet(csv_path)
    print(f"Parquet ready at {parquet_path}")

    print("\nStep 3: Pre-downloading model...")
    download_model()

    print("\nDone! Run inference with:")
    print(
        f"  python -m src.batch_inference_gpu "
        f"--input-path {parquet_path} "
        f"--output-path data/output/imdb_sentiment "
        f"--text-col review_body"
    )


if __name__ == "__main__":
    main()
