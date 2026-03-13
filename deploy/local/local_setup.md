# Local GPU Setup

## Prerequisites

### 1. Java 11 or 17 (required for PySpark)

PySpark does not support Java 21/22; you'll get `TypeError: 'JavaPackage' object is not callable`. Use **Java 11 or 17**:

- [Eclipse Temurin (Adoptium)](https://adoptium.net) — download JDK 11 or 17
- Set `JAVA_HOME` to the JDK install path (e.g. `C:\Program Files\Eclipse Adoptium\jdk-17.x.x-hotspot`)
- Verify: `java -version`

### 2. NVIDIA Driver

Install a driver compatible with your CUDA version. Verify:

```bash
nvidia-smi
```

### 3. CUDA Toolkit

PyTorch bundles its own CUDA runtime; you typically do **not** need a system-wide CUDA install. Use a PyTorch build that matches your driver, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Python Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Verify GPU in Python

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Running Locally

```bash
./deploy/local/run_local.sh
```

This uses `--generate-sample-data` and writes to `./data/sample_output`.

### With Your Own Data

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
spark-submit \
  --master 'local[*]' \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=512 \
  src/batch_inference_gpu.py \
  --input-path ./data/input \
  --output-path ./data/output
```

## Consumer GPUs vs Data Center GPUs

| GPU              | VRAM  | Suggested batch_size | maxRecordsPerBatch |
|------------------|-------|----------------------|--------------------|
| RTX 3060/4060    | 8–12GB| 8–16                 | 128–256            |
| RTX 3090/4090    | 24GB  | 32–64                | 512                |
| T4 / A10         | 16–24GB| 32                  | 512                |
| A100             | 80GB  | 64–128               | 1024               |

If you hit OOM, reduce `--batch-size` and `--max-records-per-batch`.

## Troubleshooting

- **"JavaPackage object is not callable"**: Use Java 11 or 17 (not 21/22). Set `JAVA_HOME` and run `java -version` to confirm.
- **"CUDA not available"**: Install the CUDA-enabled PyTorch build (`pip install torch` with the correct `cu*` index).
- **"Out of memory"**: Lower `--batch-size` (e.g. 8) and `--max-records-per-batch` (e.g. 128).
- **"No such file: discover_gpu.sh"**: Run from the project root or use an absolute path for the discovery script.
