# Azure Databricks GPU Setup

## Prerequisites

- Azure subscription
- Databricks workspace
- Storage account (ADLS Gen2 or blob) for input/output Parquet

## 1. Create a GPU Cluster

### Via UI

1. Compute → Create Cluster
2. Choose **Databricks runtime for ML** (GPU)
3. Select `15.4.x-gpu-ml-scala2.12` or later GPU ML runtime
4. Node type: `Standard_NC4as_T4_v3` (T4 GPU)
5. Worker count: 1–4 (use autoscaling for production)
6. **Disable Photon** (not supported with GPU instances)
7. Advanced options → Spark → add:

   ```
   spark.executor.resource.gpu.amount 1
   spark.task.resource.gpu.amount 1
   spark.sql.execution.arrow.pyspark.enabled true
   spark.sql.execution.arrow.maxRecordsPerBatch 512
   spark.python.worker.reuse true
   ```

8. Create cluster

### Via CLI

```bash
databricks clusters create --json @deploy/azure/databricks_cluster.json
```

## 2. Install Libraries

Cluster → Libraries → Install New:

- PyPI: `torch`
- PyPI: `transformers`
- PyPI: `pyarrow`

(Databricks ML runtimes usually include `pyspark` and `pandas`.)

## 3. Run the Job

### Option A: Job

1. Workflows → Jobs → Create Job
2. Task: PySpark script
3. Source: upload `src/batch_inference_gpu.py` or point to workspace path
4. Cluster: select your GPU cluster
5. Parameters: `--input-path abfs://container@account.dfs.core.windows.net/reviews/input --output-path abfs://container@account.dfs.core.windows.net/reviews/output`

### Option B: Notebook

```python
%pip install torch transformers

from src.batch_inference_gpu import run
from src.config import SparkSessionBuilder, SparkConfig

spark = SparkSessionBuilder(config=SparkConfig()).build()

run(
    spark=spark,
    input_path="/mnt/reviews/input",
    output_path="/mnt/reviews/output",
    text_col="review_body",
)
```

Mount your storage to `/mnt/reviews` first.

## Instance Types

| Node Type              | GPU              | VRAM | Use Case      |
|------------------------|------------------|------|---------------|
| Standard_NC4as_T4_v3   | 1x T4            | 16GB | Default       |
| Standard_NC24ads_A100_v4 | 1x A100        | 80GB | Large batches |
| NCads_H100_v5          | 1–2x H100        | 94GB | Heavy models  |
