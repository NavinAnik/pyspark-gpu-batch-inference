#!/bin/bash
# Create a Dataproc cluster with T4 GPUs and CUDA drivers.
# Replace REGION and INIT_ACTIONS bucket with your project values.
set -e

REGION="${GCP_REGION:-us-central1}"
PROJECT="${GCP_PROJECT:?Set GCP_PROJECT}"
CLUSTER_NAME="${CLUSTER_NAME:-pyspark-gpu-inference}"

INIT_ACTIONS="gs://goog-dataproc-initialization-actions-${REGION}/gpu/install_gpu_driver.sh"

gcloud dataproc clusters create "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT}" \
  --image-version=2.2-debian12 \
  --master-machine-type=n1-standard-4 \
  --worker-machine-type=n1-standard-4 \
  --worker-accelerator type=nvidia-tesla-t4,count=1 \
  --num-workers=2 \
  --initialization-actions="${INIT_ACTIONS}" \
  --initialization-action-timeout=20m \
  --metadata=cuda-version=12.4 \
  --properties="spark:spark.executor.resource.gpu.amount=1,spark:spark.task.resource.gpu.amount=1,spark:spark.sql.execution.arrow.pyspark.enabled=true,spark:spark.sql.execution.arrow.maxRecordsPerBatch=512"
