#!/bin/bash
# Create an EMR cluster with GPU instances for PySpark batch inference.
# Uses g4dn.xlarge (1x T4 GPU) for cores. RAPIDS is disabled since we use custom Pandas UDFs.
set -e

REGION="${AWS_REGION:-us-east-1}"
BUCKET="${S3_BUCKET:-your-bucket-name}"
RELEASE="emr-7.2.0"
BOOTSTRAP_SCRIPT="s3://${BUCKET}/scripts/emr_bootstrap.sh"

aws emr create-cluster \
  --name "pyspark-gpu-batch-inference" \
  --release-label "${RELEASE}" \
  --applications Name=Spark Name=Hadoop \
  --ec2-attributes KeyName=your-key,InstanceProfile=EMR_EC2_DefaultRole \
  --service-role EMR_DefaultRole \
  --instance-groups \
    InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
    InstanceGroupType=CORE,InstanceCount=2,InstanceType=g4dn.xlarge \
  --configurations '[
    {"Classification":"spark","Properties":{"enableSparkRapids":"false"},"Configurations":[]},
    {"Classification":"yarn-site","Properties":{"yarn.nodemanager.resource-plugins":"yarn.io/gpu"},"Configurations":[]},
    {"Classification":"capacity-scheduler","Properties":{"yarn.scheduler.capacity.resource-calculator":"org.apache.hadoop.yarn.util.resource.DominantResourceCalculator"},"Configurations":[]}
  ]' \
  --bootstrap-actions Path="${BOOTSTRAP_SCRIPT}" \
  --region "${REGION}" \
  --log-uri "s3://${BUCKET}/logs/emr/"

# Upload emr_bootstrap.sh to s3://${BUCKET}/scripts/ before running.
