#!/bin/bash
# Submit the batch inference job to an existing EMR cluster.
set -e

CLUSTER_ID="${EMR_CLUSTER_ID:?Set EMR_CLUSTER_ID}"
BUCKET="${S3_BUCKET:?Set S3_BUCKET}"
INPUT_PATH="${INPUT_PATH:-s3://${BUCKET}/data/reviews/input}"
OUTPUT_PATH="${OUTPUT_PATH:-s3://${BUCKET}/data/reviews/output}"
SCRIPT_PATH="s3://${BUCKET}/scripts/batch_inference_gpu.py"

aws emr add-steps \
  --cluster-id "${CLUSTER_ID}" \
  --steps '[
    {
      "Name": "GPU Batch Inference",
      "ActionOnFailure": "CONTINUE",
      "HadoopJarStep": {
        "Jar": "command-runner.jar",
        "Args": [
          "spark-submit",
          "--master", "yarn",
          "--deploy-mode", "cluster",
          "--conf", "spark.executor.resource.gpu.amount=1",
          "--conf", "spark.task.resource.gpu.amount=1",
          "--conf", "spark.sql.execution.arrow.pyspark.enabled=true",
          "--conf", "spark.sql.execution.arrow.maxRecordsPerBatch=512",
          "--conf", "spark.python.worker.reuse=true",
          "--py-files", "s3://'${BUCKET}'/scripts/src.zip",
          "'"${SCRIPT_PATH}"'",
          "--input-path", "'"${INPUT_PATH}"'",
          "--output-path", "'"${OUTPUT_PATH}"'"
        ]
      }
    }
  ]'
