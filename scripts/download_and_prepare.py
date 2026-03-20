"""
Download Sea Turtles in Drone Imagery dataset from Kaggle, build Parquet of image paths, and pre-download the ViT model.

Credentials: set in project .env (KAGGLE_USERNAME, KAGGLE_KEY) or ~/.kaggle/kaggle.json.
Create token at: https://www.kaggle.com/settings (Account -> Create New Token)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load project .env so KAGGLE_USERNAME and KAGGLE_KEY are set when running this script
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # Fallback: simple key=value parse (no quotes or comments)
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

KAGGLE_DATASET = "kmldas/sea-turtles-in-drone-imagery"
# Fallback if primary dataset returns 403 (e.g. rules not accepted); smaller, often no rules
KAGGLE_FALLBACK_DATASET = "dansbecker/hot-dog-not-hot-dog"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sea_turtles"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
MODEL_NAME = "google/vit-base-patch16-224"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _ensure_kaggle_json() -> None:
    """Write ~/.kaggle/kaggle.json from KAGGLE_USERNAME and KAGGLE_KEY so the Kaggle API can authenticate."""
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not username or not key or username == "your_kaggle_username":
        return
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": key}, f, indent=2)
    try:
        kaggle_json.chmod(0o600)
    except OSError:
        pass


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API is configured (from .env or ~/.kaggle/kaggle.json)."""
    _ensure_kaggle_json()
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            with open(kaggle_json) as f:
                data = json.load(f)
            if data.get("username") and data.get("key") and data.get("username") != "your_kaggle_username":
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _require_valid_username() -> None:
    """Exit with clear instructions if KAGGLE_USERNAME is still the placeholder."""
    username = (os.environ.get("KAGGLE_USERNAME") or "").strip()
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not username and kaggle_json.exists():
        try:
            with open(kaggle_json) as f:
                username = (json.load(f).get("username") or "").strip()
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    if username in ("", "your_kaggle_username"):
        print(
            "KAGGLE_USERNAME must be your real Kaggle username, not the placeholder.\n"
            "  1. Open https://www.kaggle.com/settings (logged in)\n"
            "  2. Your username is shown there, or in the 'Account' section\n"
            "  3. Edit project .env and set: KAGGLE_USERNAME=your_actual_username\n"
            "  4. Keep KAGGLE_KEY as your API key (the KGAT_... or key from 'Create New Token')"
        )
        sys.exit(1)


def download_dataset() -> Path:
    """Download Sea Turtles in Drone Imagery dataset from Kaggle and return path to extracted directory."""
    if not check_kaggle_credentials():
        print(
            "Kaggle API not configured. Please:\n"
            "  1. Create token at https://www.kaggle.com/settings\n"
            "  2. Place kaggle.json at ~/.kaggle/kaggle.json\n"
            "     (or set KAGGLE_USERNAME and KAGGLE_KEY in .env)"
        )
        sys.exit(1)

    _require_valid_username()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi
    from requests.exceptions import HTTPError

    api = KaggleApi()
    api.authenticate()

    def _download(dataset: str) -> None:
        api.dataset_download_files(
            dataset,
            path=str(RAW_DIR.parent),
            unzip=True,
        )

    try:
        _download(KAGGLE_DATASET)
    except HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            print(
                "\n403 Forbidden: This usually means you must accept the dataset's rules in a browser first.\n"
                "  1. Open this link (logged in to Kaggle):\n"
                f"     https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
                "  2. Go to the 'Data' tab and click 'Download' (or 'I Understand and Accept' if shown).\n"
                "  3. Run this script again.\n"
                "Trying fallback dataset for demo..."
            )
            try:
                _download(KAGGLE_FALLBACK_DATASET)
                print(f"Using fallback dataset for demo: {KAGGLE_FALLBACK_DATASET}")
            except HTTPError as e2:
                if e2.response is not None and e2.response.status_code == 403:
                    print(
                        "\nFallback dataset also returned 403. Accept the primary dataset rules above, or regenerate your token at https://www.kaggle.com/settings"
                    )
                raise
        else:
            raise
    # Dataset extracts under data/raw
    return RAW_DIR.parent


def collect_image_paths(root: Path) -> list[tuple[str, str, str]]:
    """
    Walk directory tree; return list of (image_id, image_path, label).
    Assumes structure: root / split (train|valid|test) / species_name / *.jpg
    """
    rows = []
    seen = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS:
            # Use parent folder as label (species name)
            label = path.parent.name
            image_path = str(path.resolve())
            image_id = path.stem + "_" + path.parent.name
            if image_id in seen:
                image_id = f"{path.stem}_{len(rows)}"
            seen.add(image_id)
            rows.append((image_id, image_path, label))
    return rows


def build_parquet(extracted_root: Path) -> Path:
    """Build Parquet with columns: image_id, image_path, label."""
    import pandas as pd

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARQUET_DIR / "sea_turtles.parquet"

    rows = collect_image_paths(extracted_root)
    if not rows:
        raise FileNotFoundError(
            f"No image files found under {extracted_root}. "
            "Check that the dataset extracted correctly (train/valid/test with species subdirs)."
        )

    df = pd.DataFrame(rows, columns=["image_id", "image_path", "label"])
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def download_model() -> None:
    """Pre-download and cache the ViT image-classification model, run sanity check."""
    from transformers import pipeline

    print("Downloading model (this may take a minute)...")
    pipe = pipeline(
        "image-classification",
        model=MODEL_NAME,
    )
    # Sanity check: run on a tiny placeholder (HF can accept a URL or we create a small image)
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (224, 224), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        out = pipe(Image.open(buf))
    except Exception:
        out = pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrot.jpg")
    print(f"Model loaded. Sanity check: {out[:1] if isinstance(out, list) else out}")
    print("Model cached and ready for inference.")


def main() -> None:
    print("Step 1: Downloading Sea Turtles in Drone Imagery dataset from Kaggle...")
    extracted = download_dataset()
    print(f"Downloaded to {extracted}")

    print("\nStep 2: Building Parquet of image paths...")
    parquet_path = build_parquet(extracted)
    print(f"Parquet ready at {parquet_path}")

    print("\nStep 3: Pre-downloading ViT model...")
    download_model()

    print("\nDone! Run inference with:")
    print(
        "  python scripts/run_inference_local.py \\"
    )
    print(
        f"    --input-path {parquet_path} \\"
    )
    print(
        "    --output-path data/output/sea_turtles_predictions.parquet"
    )
    print("Or with PySpark:")
    print(
        f"  python -m src.batch_inference_gpu \\"
    )
    print(
        f"    --input-path {parquet_path} \\"
    )
    print(
        "    --output-path data/output/sea_turtles_predictions"
    )


if __name__ == "__main__":
    main()
