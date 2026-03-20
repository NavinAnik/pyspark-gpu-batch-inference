#!/bin/bash
# Run PySpark batch inference locally with GPU (spark-submit --master local[*]).
# Supports image classification (default) or sentiment-analysis. Uses NVIDIA CUDA or Apple Silicon MPS.
# Tuned for M4 Pro / 24GB RAM: higher driver memory, larger Arrow batches, larger inference batch.
# For image demo: run scripts/download_and_prepare.py first, then pass --input-path and --output-path.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

# Prevent Spark from binding to a LAN IP that the executor can't reach back on
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"

SCRIPT="src/batch_inference_gpu.py"
DISCOVER_SCRIPT="${ROOT}/deploy/local/discover_gpu.sh"

# Memory: 24GB RAM → ~10g driver, ~10g executor (local mode shares JVM; leave room for OS + Python)
SPARK_DRIVER_MEMORY="${SPARK_DRIVER_MEMORY:-10g}"
SPARK_EXECUTOR_MEMORY="${SPARK_EXECUTOR_MEMORY:-10g}"
# Larger batches = fewer callbacks, better GPU utilization (M4 Pro / 24GB)
ARROW_BATCH="${ARROW_MAX_RECORDS_PER_BATCH:-1024}"
INFERENCE_BATCH="${INFERENCE_BATCH_SIZE:-64}"

spark-submit \
  --master 'local[*]' \
  --conf spark.driver.host=127.0.0.1 \
  --conf spark.driver.bindAddress=127.0.0.1 \
  --conf spark.driver.memory="${SPARK_DRIVER_MEMORY}" \
  --conf spark.executor.memory="${SPARK_EXECUTOR_MEMORY}" \
  --conf spark.driver.resource.gpu.amount=1 \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.driver.resource.gpu.discoveryScript="${DISCOVER_SCRIPT}" \
  --conf spark.executor.resource.gpu.discoveryScript="${DISCOVER_SCRIPT}" \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch="${ARROW_BATCH}" \
  --conf spark.python.worker.reuse=true \
  "${SCRIPT}" \
  --task image-classification \
  --batch-size "${INFERENCE_BATCH}" \
  --max-records-per-batch "${ARROW_BATCH}" \
  "$@"
