#!/bin/bash
# Submit the batch inference job to an existing Dataproc cluster.
set -e

REGION="${GCP_REGION:-us-central1}"
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
CLUSTER_NAME="${CLUSTER_NAME:-pyspark-gpu-inference}"
INPUT_PATH="${INPUT_PATH:-gs://your-bucket/reviews/input}"
OUTPUT_PATH="${OUTPUT_PATH:-gs://your-bucket/reviews/output}"

# Path to the main script in GCS (upload first)
SCRIPT_GCS="${SCRIPT_GCS:-gs://your-bucket/scripts/batch_inference_gpu.py}"

gcloud dataproc jobs submit pyspark "${SCRIPT_GCS}" \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT}" \
  --properties="spark.executor.resource.gpu.amount=1,spark.task.resource.gpu.amount=1,spark.sql.execution.arrow.pyspark.enabled=true,spark.sql.execution.arrow.maxRecordsPerBatch=512,spark.python.worker.reuse=true" \
  -- \
  --input-path "${INPUT_PATH}" \
  --output-path "${OUTPUT_PATH}"
