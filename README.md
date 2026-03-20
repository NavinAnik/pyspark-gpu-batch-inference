# Scaling ML Batch Inference: PySpark + CUDA / MPS GPUs

Reference implementation for GPU-accelerated batch inference using PySpark, Pandas UDFs (Arrow), and Hugging Face transformers. Processes Amazon Product Reviews in Parquet with a DistilBERT sentiment model across AWS EMR, Azure Databricks, GCP Dataproc, and local GPU (NVIDIA CUDA or Apple Silicon MPS).

## Architecture

```mermaid
flowchart LR
  dataLake["Cloud Data Lake (Parquet)"]
  driver["Spark Driver"]
  subgraph executors ["Spark Executors with GPUs"]
    e1["Executor 1"]
    e2["Executor 2"]
    eN["Executor N"]
  end
  hf["HF Sentiment Pipeline (PyTorch + CUDA/MPS)"]
  output["Output Parquet"]

  dataLake --> driver --> executors
  executors --> hf --> output
```

## Quick Start (Local GPU)

1. **Prerequisites**: **Python 3.8–3.12**, **Java 11 or 17** (not 21/22). For GPU: NVIDIA GPU with `nvidia-smi` (CUDA) or Apple Silicon Mac (MPS).

   ```bash
   pip install -r requirements.txt
   python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'MPS:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"
   ```

2. **Run with sample data** (no external dataset):

   ```bash
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   ./deploy/local/run_local.sh
   ```

   Output: `./data/sample_output/`

3. **Run with your own Parquet** (use `--device mps` on Apple Silicon, `--device -1` for CPU):

   ```bash
   spark-submit src/batch_inference_gpu.py \
     --input-path /path/to/input \
     --output-path /path/to/output \
     --text-col review_body
   ```

## Deployment

| Platform | Guide |
|----------|-------|
| Local GPU | [deploy/local/local_setup.md](deploy/local/local_setup.md) |
| AWS EMR | [deploy/aws/](deploy/aws/) – `emr_create_cluster.sh`, `emr_submit_job.sh` |
| Azure Databricks | [deploy/azure/databricks_setup.md](deploy/azure/databricks_setup.md) |
| GCP Dataproc | [deploy/gcp/](deploy/gcp/) – `dataproc_create_cluster.sh`, `dataproc_submit_job.sh` |

## Article

See [article/ARTICLE_OUTLINE.md](article/ARTICLE_OUTLINE.md) for the full technical article outline and deep dive.

## Tests

```bash
pytest tests/ -v
```

Uses CPU fallback; no GPU required. Requires **Python 3.8–3.12** (PySpark/Py4J do not support 3.14+).
