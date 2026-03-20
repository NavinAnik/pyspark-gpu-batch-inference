#!/bin/bash
# GPU discovery script for Spark. Outputs JSON: {"name":"gpu","addresses":["0","1",...]}
# See: https://spark.apache.org/docs/latest/configuration.html#configuring-gpus
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | \
    awk 'BEGIN{printf "{\"name\":\"gpu\",\"addresses\":["} {printf "%s\"%s\"", (NR>1?",":""), $1} END{print "]}"}'
elif [[ "$(uname)" == "Darwin" ]]; then
  # Apple Silicon MPS: report one "gpu" so Spark sees it
  echo '{"name":"gpu","addresses":["0"]}'
else
  echo '{"name":"gpu","addresses":[]}'
fi
