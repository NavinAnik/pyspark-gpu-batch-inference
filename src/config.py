from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

from pyspark.sql import SparkSession

from src.utils.logging_utils import setup_logger


logger = setup_logger(__name__)


@dataclass
class SparkConfig:
    """
    Container for Spark configuration values.

    Values are exposed as a dataclass to make it explicit which settings are
    tuned for GPU-aware execution and easier to document in the article.
    """

    app_name: str = "pyspark-gpu-batch-inference"
    master: Optional[str] = None

    # Arrow + Pandas UDF tuning (1024 for 24GB RAM / M4 Pro)
    arrow_enabled: bool = True
    arrow_max_records_per_batch: int = 1024

    # GPU resource configuration (one task per GPU by default)
    executor_gpu_amount: int = 1
    task_gpu_amount: float = 1.0
    driver_gpu_amount: Optional[int] = None

    # Path to resource discovery script when running on YARN / standalone
    gpu_discovery_script: Optional[str] = None

    # Additional Spark conf overrides
    extra_confs: Optional[Dict[str, str]] = None


class SparkSessionBuilder:
    """
    Build a SparkSession with GPU-aware configuration.

    All GPU and Arrow-related configuration is centralized here so that the
    article can clearly show which knobs matter for performance and stability.
    """

    def __init__(self, config: Optional[SparkConfig] = None) -> None:
        self._config = config or self._from_env()

    def _from_env(self) -> SparkConfig:
        """
        Create a SparkConfig using sensible defaults, allowing overrides from
        environment variables to make cloud jobs configurable without code changes.
        """
        arrow_max = int(os.getenv("ARROW_MAX_RECORDS_PER_BATCH", "512"))
        executor_gpu = int(os.getenv("SPARK_EXECUTOR_GPU_AMOUNT", "1"))
        task_gpu = float(os.getenv("SPARK_TASK_GPU_AMOUNT", "1.0"))
        driver_gpu_env = os.getenv("SPARK_DRIVER_GPU_AMOUNT")
        driver_gpu = int(driver_gpu_env) if driver_gpu_env is not None else None
        discovery_script = os.getenv("SPARK_GPU_DISCOVERY_SCRIPT")

        return SparkConfig(
            app_name=os.getenv("SPARK_APP_NAME", "pyspark-gpu-batch-inference"),
            master=os.getenv("SPARK_MASTER"),
            arrow_enabled=True,
            arrow_max_records_per_batch=arrow_max,
            executor_gpu_amount=executor_gpu,
            task_gpu_amount=task_gpu,
            driver_gpu_amount=driver_gpu,
            gpu_discovery_script=discovery_script,
            extra_confs=None,
        )

    def build(self) -> SparkSession:
        """
        Instantiate a SparkSession with all GPU and Arrow tuning applied.
        """
        logger.info("Building SparkSession with GPU-aware configuration.")
        builder = SparkSession.builder.appName(self._config.app_name)

        if self._config.master:
            builder = builder.master(self._config.master)

        # Arrow configuration for efficient Pandas UDFs.
        if self._config.arrow_enabled:
            builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
        builder = builder.config(
            "spark.sql.execution.arrow.maxRecordsPerBatch",
            str(self._config.arrow_max_records_per_batch),
        )

        # Encourage worker reuse so that model weights stay hot in memory.
        builder = builder.config("spark.python.worker.reuse", "true")

        # Bind driver to localhost to prevent heartbeat failures on local mode
        builder = builder.config("spark.driver.host", "127.0.0.1")
        builder = builder.config("spark.driver.bindAddress", "127.0.0.1")

        # GPU resource discovery and allocation.
        builder = builder.config(
            "spark.executor.resource.gpu.amount",
            str(self._config.executor_gpu_amount),
        ).config(
            "spark.task.resource.gpu.amount",
            str(self._config.task_gpu_amount),
        )

        if self._config.driver_gpu_amount is not None:
            builder = builder.config(
                "spark.driver.resource.gpu.amount",
                str(self._config.driver_gpu_amount),
            )

        if self._config.gpu_discovery_script:
            builder = builder.config(
                "spark.executor.resource.gpu.discoveryScript",
                self._config.gpu_discovery_script,
            ).config(
                "spark.driver.resource.gpu.discoveryScript",
                self._config.gpu_discovery_script,
            )

        if self._config.extra_confs:
            for key, value in self._config.extra_confs.items():
                builder = builder.config(key, value)

        spark = builder.getOrCreate()
        logger.info("SparkSession created with master=%s", spark.sparkContext.master)
        return spark

