## Scaling Machine Learning Batch Inference: Utilizing PySpark and CUDA GPUs Across AWS, Azure, and GCP

### Abstract

- **Problem**: Traditional CPU-bound batch inference pipelines struggle to keep up with the scale and latency requirements of modern machine learning workloads, especially for NLP over billions of documents.
- **Solution**: Combine PySpark 3.x GPU-aware scheduling, Pandas UDFs (Arrow), and PyTorch/CUDA (via Hugging Face transformers) to build a scalable, cloud-agnostic batch inference architecture.
- **Deliverable**: A reference implementation that processes the Amazon Product Reviews dataset stored in a cloud data lake (Parquet) using a GPU-accelerated sentiment analysis model (`distilbert-base-uncased-finetuned-sst-2-english`) across AWS EMR, Azure Databricks, GCP Dataproc, and a single local GPU machine.

### 1. Introduction: Why Scale Batch Inference?

- **1.1. From model training to model serving at scale**
  - Contrast online inference (REST/gRPC) vs. offline/batch inference.
  - Use cases for batch inference on text: customer sentiment tracking, compliance scans, topic tagging.
- **1.2. The CPU bottleneck**
  - Explain why Python + CPUs quickly become the bottleneck for transformer-based NLP.
  - Limitations of single-node pandas/NumPy pipelines.
- **1.3. Objectives of this article**
  - What the reader will build: a PySpark + GPU batch inference pipeline for Amazon Product Reviews.
  - Technologies used: PySpark 3.x, Pandas UDFs, Arrow, PyTorch, Hugging Face, CUDA-capable GPUs.

### 2. Architecture Overview

- **2.1. High-level data flow**
  - Describe the end-to-end pipeline: ingest Parquet from data lake → distribute rows across Spark executors → Arrow-serialize into columnar batches → run GPU-accelerated sentiment model via Pandas UDF → write Parquet back to the lake.
- **2.2. Logical architecture diagram**

```mermaid
flowchart LR
  dataLake["Cloud Data Lake (Parquet)"]
  driver["Spark Driver"]
  subgraph gpuCluster ["Spark Executors with GPUs"]
    executor1["Executor 1\nGPU 0"]
    executor2["Executor 2\nGPU 0"]
    executorN["Executor N\nGPU 0"]
  end
  hfPipeline["HF Sentiment Pipeline\n(Pytorch + CUDA)"]
  outputLake["Curated Parquet\n(With Sentiment Columns)"]

  dataLake --> driver --> gpuCluster
  gpuCluster --> hfPipeline --> outputLake
```

- **2.3. Physical deployment variants**
  - Local workstation with a single NVIDIA GPU (developer workflow).
  - AWS EMR GPU cluster (e.g., `g4dn.xlarge`).
  - Azure Databricks GPU cluster (NC-series).
  - GCP Dataproc with T4 GPUs.

### 3. PySpark 3.x GPU-Aware Scheduling Deep Dive

- **3.1. How Spark 3.x understands GPUs**
  - Resource profiles and the `spark.*.resource.gpu.*` configuration family.
  - Difference between driver, executor, and task level GPU configs.
- **3.2. Core GPU configuration parameters**
  - `spark.executor.resource.gpu.amount`
  - `spark.task.resource.gpu.amount`
  - `spark.executor.resource.gpu.discoveryScript`
  - Discussion of fractional GPU amounts (e.g., `0.25`) and when you would or would not use them for inference.
- **3.3. Scheduling semantics across cluster managers**
  - Briefly cover YARN, Kubernetes, and standalone/Databricks/Dataproc abstractions.
  - How resource discovery scripts map logical Spark GPUs to physical devices.
- **3.4. Why we choose one task per GPU for this workload**
  - DistilBERT memory footprint and batch-size trade-offs.
  - Avoiding GPU thrash and OOM by not oversubscribing.

### 4. Pandas UDFs + PyTorch + CUDA: The Execution Engine

- **4.1. Why Pandas UDFs instead of row UDFs**
  - Arrow-based vectorization eliminates per-row serialization overhead.
  - JVM ↔ Python boundary crossed once per batch instead of once per row.
- **4.2. Iterator-style Pandas UDFs**
  - Explain `Iterator[pandas.Series] -> Iterator[pandas.Series]` pattern.
  - How this allows us to load the Hugging Face pipeline once per Python worker (per partition) instead of once per row or batch.
- **4.3. Integrating the Hugging Face pipeline**
  - How the `pipeline("sentiment-analysis", model=..., device=0)` abstraction wraps tokenization, model forward pass, and decoding.
  - Why we rely on device placement at model load time and do not move tensors manually inside the UDF.
- **4.4. CUDA memory management and batching strategy**
  - How Arrow batch size (`spark.sql.execution.arrow.maxRecordsPerBatch`) interacts with model batch size.
  - Implemented strategy: start with a safe `batch_size`, catch `torch.cuda.OutOfMemoryError`, clear cache, and retry with a smaller batch.
  - Discuss truncation and maximum sequence length for long reviews.
- **4.5. Output schema and downstream consumption**
  - Returning JSON-encoded sentiment (label + score) vs. a full nested `StructType`.
  - Trade-offs for BI tools and downstream ETL.

### 5. Implementation Walkthrough

- **5.1. Repository layout**
  - Brief tour of `src/config.py`, `src/inference/model_handler.py`, `src/inference/udf.py`, `src/batch_inference_gpu.py`, and deployment scripts.
- **5.2. Spark configuration module (`config.py`)**
  - Walk through `SparkSessionBuilder` and key configs:
    - Arrow enablement and `maxRecordsPerBatch`.
    - GPU resource settings.
    - Sensible defaults with environment variable overrides.
- **5.3. Model lifecycle (`model_handler.py`)**
  - Show how the Hugging Face pipeline is created lazily and cached.
  - Explain batch prediction helper, OOM handling, and CPU fallback for local/testing.
- **5.4. The Pandas UDF (`udf.py`)**
  - Detailed look at the iterator-based UDF, including:
    - Model initialization on first batch.
    - Iterating over Arrow batches, inferring sentiment, and emitting Pandas Series.
    - Logging at the partition / batch level.
- **5.5. Orchestration script (`batch_inference_gpu.py`)**
  - CLI arguments: input/output paths, model name, batch size, Arrow batch size, and `--generate-sample-data`.
  - Reading the Amazon Reviews Parquet (schema assumptions and customization points).
  - Applying the UDF and writing enriched Parquet with partitioning strategy.
  - Logging of record counts and wall-clock timings.

### 6. Deployment Across Local GPU and Clouds

- **6.1. Local GPU (developer workflow)**
  - Running `spark-submit --master local[*]` on a workstation with a single NVIDIA GPU.
  - Validating CUDA installation and PyTorch GPU visibility.
  - Using `--generate-sample-data` for dry runs without a real data lake.
- **6.2. AWS EMR**
  - Choosing GPU instance types (`g4dn`, `p3`) and EMR release.
  - Enabling GPU support and configuring Spark for GPUs.
  - Example `aws emr create-cluster` and `aws emr add-steps` commands.
- **6.3. Azure Databricks**
  - Selecting NC-series or similar GPU SKUs.
  - Using Databricks Runtime for Machine Learning (GPU) and Spark configs.
  - Submitting the job via Jobs UI or Databricks CLI with cluster JSON spec.
- **6.4. GCP Dataproc**
  - Attaching T4 GPUs to master/worker nodes.
  - Installing CUDA drivers via Dataproc initialization actions or ML images.
  - Example `gcloud dataproc clusters create` and `gcloud dataproc jobs submit pyspark` commands.

### 7. Performance Benchmarks & Tuning

- **7.1. Baseline CPU vs. GPU comparison**
  - Methodology: fixed number of reviews, same model, same Spark cluster size.
  - Throughput metrics (rows/second) and cost considerations.
- **7.2. Key tuning levers**
  - Spark partition count vs. number of GPUs.
  - Arrow batch size vs. model batch size.
  - Choosing `spark.task.resource.gpu.amount` and concurrency.
- **7.3. Observability and troubleshooting**
  - Monitoring GPU utilization with `nvidia-smi`.
  - Tracking Spark stage metrics and skew.
  - Common failure modes (OOM, skewed partitions, driver bottlenecks) and mitigations.

### 8. Production Considerations and Next Steps

- **8.1. Model and data versioning**
  - Using model registries / artifacts with explicit version tags.
  - Partitioning strategies for reruns and backfills.
- **8.2. Reliability and cost optimization**
  - Retry policies, idempotent writes, and checkpointing.
  - Using spot/preemptible instances and autoscaling.
- **8.3. Extending beyond sentiment analysis**
  - Applying the same pattern to topic modeling, NER, embedding generation.
  - Integrating with vector databases and downstream search systems.

