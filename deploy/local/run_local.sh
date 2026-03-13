#!/bin/bash
# Run PySpark batch inference locally with GPU (spark-submit --master local[*]).
# Set PYTHONPATH so `src` is importable. Use --generate-sample-data for quick testing.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

SCRIPT="src/batch_inference_gpu.py"
DISCOVER_SCRIPT="${ROOT}/deploy/local/discover_gpu.sh"

spark-submit \
  --master 'local[*]' \
  --conf spark.driver.resource.gpu.amount=1 \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=1 \
  --conf spark.driver.resource.gpu.discoveryScript="${DISCOVER_SCRIPT}" \
  --conf spark.executor.resource.gpu.discoveryScript="${DISCOVER_SCRIPT}" \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=512 \
  --conf spark.python.worker.reuse=true \
  "${SCRIPT}" \
  --generate-sample-data \
  --output-path ./data \
  "$@"
