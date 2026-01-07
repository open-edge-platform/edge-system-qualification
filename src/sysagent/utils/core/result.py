# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Result module for handling test results in a structured way.

The Result class now includes support for KPI (Key Performance Indicator)
configurations and validation results, enabling comprehensive test reporting
that includes both measured metrics and their validation status against
configured thresholds.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    unit: Optional[str] = None
    value: Any = None
    is_key_metric: bool = False


@dataclass
class Result:
    """
    Structured test result container.

    The Result class automatically includes default metadata:
    - created_at: ISO timestamp of result creation
    - updated_at: ISO timestamp of result completion (updated via update_timestamps())
    - total_duration_seconds: Total test duration calculated from created_at to updated_at
    - kpi_validation_status: Overall KPI validation status ("skipped", "passed", or "failed")

    Additional metadata can be added for test-specific information:
    - model_export_duration_seconds: Duration of model export operation (if applicable)

    The Result class supports automatic metadata merging from profile and test configurations.
    Use Result.from_test_config() or result.apply_config_metadata() to apply configuration metadata.

    Attributes:
        name: Test identifier
        parameters: Test configuration parameters
        metrics: Measured test metrics with units
        metadata: Additional test metadata and status information (auto-populated with defaults and config metadata)
                  Should contain simple property-value pairs for human-readable reporting
        extended_metadata: Structured metadata for storing complex objects and data structures
                          Used for data analysis, visualization, and programmatic access
                          Not constrained to simple key-value pairs like metadata
        kpis: KPI configurations and validation results, structured as:
              {kpi_name: {"config": {...}, "validation": {...}, "mode": "all|any|skip"}}
    """

    name: str = "T0000"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Metrics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extended_metadata: Dict[str, Any] = field(default_factory=dict)
    kpis: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Add created_at to metadata if not already present
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().astimezone().isoformat()

        # Add updated_at to metadata if not already present
        if "updated_at" not in self.metadata:
            self.metadata["updated_at"] = datetime.now().astimezone().isoformat()

        # Add default KPI validation status if not already present
        if "kpi_validation_status" not in self.metadata:
            self.metadata["kpi_validation_status"] = "skipped"

    def to_dict(self):
        # Convert nested dataclasses to dict for JSON serialization
        return {
            "name": self.name,
            "parameters": self.parameters,
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
            "metadata": self.metadata,
            "extended_metadata": self.extended_metadata,
            "kpis": self.kpis,
        }

    def update_timestamps(self):
        """
        Update the updated_at timestamp and calculate total duration.

        This should be called when the test completes to capture the final timestamp
        and calculate total test duration in seconds.
        """
        self.metadata["updated_at"] = datetime.now().astimezone().isoformat()

        # Calculate duration if both timestamps exist
        if "created_at" in self.metadata and "updated_at" in self.metadata:
            try:
                created = datetime.fromisoformat(self.metadata["created_at"])
                updated = datetime.fromisoformat(self.metadata["updated_at"])
                duration_seconds = (updated - created).total_seconds()
                self.metadata["total_duration_seconds"] = round(duration_seconds, 2)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to calculate duration: {e}")

    def update_kpi_validation_status(self, validation_results: Dict[str, Any], mode: str = "all"):
        """
        Update the KPI validation status in metadata based on validation results and mode.

        Args:
            validation_results: Dictionary containing validation results with 'passed' and 'skipped' keys
            mode: Validation mode ('all', 'any', or 'skip')
        """
        # If mode is "skip", force status to "skipped" regardless of validation results
        if mode == "skip":
            self.metadata["kpi_validation_status"] = "skipped"
            self.metadata["kpi_validation_mode"] = "skip"
        elif validation_results.get("skipped", False):
            self.metadata["kpi_validation_status"] = "skipped"
            # When validation is skipped, set mode to "skip"
            self.metadata["kpi_validation_mode"] = "skip"
        elif validation_results.get("passed", False):
            self.metadata["kpi_validation_status"] = "passed"
            # Store the actual mode used for validation
            self.metadata["kpi_validation_mode"] = mode
        else:
            self.metadata["kpi_validation_status"] = "failed"
            # Store the actual mode used for validation
            self.metadata["kpi_validation_mode"] = mode

    def get_final_validation_mode(self, validation_results: Dict[str, Any], original_mode: str = "all") -> str:
        """
        Get the final validation mode based on validation results.

        Args:
            validation_results: Dictionary containing validation results with 'skipped' key
            original_mode: Original validation mode from configuration ('all' or 'any')

        Returns:
            Final validation mode ('skip' if validation was skipped, otherwise original_mode)
        """
        if validation_results.get("skipped", False):
            return "skip"
        return original_mode

    def set_key_metric(self, metric_name: str):
        """
        Set a specific metric as the key metric. Only one metric can be key at a time.

        Args:
            metric_name: Name of the metric to mark as key metric
        """
        # First, clear any existing key metrics
        for name, metric in self.metrics.items():
            metric.is_key_metric = False

        # Set the specified metric as key metric if it exists
        if metric_name in self.metrics:
            self.metrics[metric_name].is_key_metric = True
        else:
            raise ValueError(
                f"Metric '{metric_name}' not found in results. Available metrics: {list(self.metrics.keys())}"
            )

    def get_key_metric(self) -> Optional[str]:
        """
        Get the name of the current key metric.

        Returns:
            Name of the key metric, or None if no key metric is set
        """
        for name, metric in self.metrics.items():
            if metric.is_key_metric:
                return name
        return None

    def _is_device_specific_metric(self, metric_name: str) -> bool:
        """
        Check if a metric name represents a device-specific metric.

        Supports all device naming patterns:
        - CPU: metric_cpu
        - iGPU: metric_igpu
        - Single dGPU: metric_dgpu (no index)
        - Indexed dGPUs: metric_dgpu0, metric_dgpu1, metric_dgpu2, etc.
        - HETERO devices: metric_hetero_dgpu
        - NPU: metric_npu

        Args:
            metric_name: Name of the metric to check

        Returns:
            True if the metric is device-specific, False if it's an aggregate metric
        """
        import re

        # Pattern for device-specific suffixes:
        # - _cpu, _igpu, _npu (simple suffixes)
        # - _dgpu (single dGPU without index)
        # - _dgpu0, _dgpu1, _dgpu2, etc. (indexed dGPUs)
        # - _hetero_dgpu (HETERO devices)
        device_pattern = r"_(cpu|igpu|npu|dgpu\d*|hetero_dgpu)$"

        return bool(re.search(device_pattern, metric_name))

    def auto_set_key_metric(
        self,
        validation_results: Optional[Dict[str, Any]] = None,
        kpi_validation_mode: str = "all",
        device_count: int = 1,
    ):
        """
        Automatically determine and set the key metric based on validation results and context.
        Enhanced to handle multi-device scenarios with aggregate and individual metrics.

        Supports all device naming patterns:
        - CPU: metric_cpu
        - iGPU: metric_igpu
        - Single dGPU: metric_dgpu (no index)
        - Indexed dGPUs: metric_dgpu0, metric_dgpu1, etc.
        - HETERO devices: metric_hetero_dgpu
        - NPU: metric_npu

        Args:
            validation_results: KPI validation results
            kpi_validation_mode: Validation mode ('all', 'any', 'skip')
            device_count: Number of devices being tested
        """
        if not self.metrics:
            logger.debug("No metrics available for key metric selection")
            return

        logger.debug(f"Auto-determining key metric from available metrics: {list(self.metrics.keys())}")
        logger.debug(f"KPI validation mode: {kpi_validation_mode}, Device count: {device_count}")

        # Strategy 1: KPI validation mode "any" - use the first passed metric (highest priority)
        if kpi_validation_mode == "any" and validation_results and validation_results.get("validations"):
            for metric_name in self.metrics.keys():
                validation_data = validation_results["validations"].get(metric_name)
                if validation_data and validation_data.get("passed", False):
                    self.set_key_metric(metric_name)
                    logger.debug(f"Set key metric to '{metric_name}' based on KPI validation mode 'any'")
                    return

        # Strategy 2: Multi-device scenario - prioritize aggregate metrics over individual device metrics
        if device_count > 1:
            aggregate_metrics = []
            device_specific_metrics = []

            for metric_name in self.metrics.keys():
                # Check if it's a device-specific metric using comprehensive pattern matching
                is_device_specific = self._is_device_specific_metric(metric_name)

                if is_device_specific:
                    device_specific_metrics.append(metric_name)
                else:
                    aggregate_metrics.append(metric_name)

            logger.debug(
                f"Multi-device test - Aggregate: {aggregate_metrics}, Device-specific: {device_specific_metrics}"
            )

            # For multi-device tests, prefer aggregate metrics for KPI validation
            # Prioritize aggregate metrics even if they have error values (0 or -1)
            if aggregate_metrics:
                # Prioritize certain metric types for key metric selection
                priority_metrics = ["streams_max", "throughput", "performance_score", "fps_total"]
                for priority_metric in priority_metrics:
                    for metric_name in aggregate_metrics:
                        if priority_metric in metric_name.lower():
                            self.set_key_metric(metric_name)
                            logger.debug(
                                f"Set key metric to aggregate metric '{metric_name}' for multi-device test "
                                f"(value: {self.metrics[metric_name].value})"
                            )
                            return

                # If no priority metrics found, use the first aggregate metric
                self.set_key_metric(aggregate_metrics[0])
                logger.debug(
                    f"Set key metric to first aggregate metric '{aggregate_metrics[0]}' for multi-device test "
                    f"(value: {self.metrics[aggregate_metrics[0]].value})"
                )
                return

            # If no aggregate metrics but have device-specific metrics, choose the highest value one
            elif device_specific_metrics:
                best_metric = None
                best_value = -1
                for metric_name in device_specific_metrics:
                    metric_value = getattr(self.metrics[metric_name], "value", 0)
                    if isinstance(metric_value, (int, float)) and metric_value > best_value:
                        best_value = metric_value
                        best_metric = metric_name

                if best_metric:
                    self.set_key_metric(best_metric)
                    logger.debug(f"Set key metric to highest-value device metric '{best_metric}' = {best_value}")
                    return

        # Strategy 3: Single device scenario - prioritize main performance metrics
        else:
            single_device_priorities = ["speed", "streams_max", "throughput", "performance", "fps", "latency"]
            for priority_pattern in single_device_priorities:
                for metric_name in self.metrics.keys():
                    if priority_pattern in metric_name.lower():
                        self.set_key_metric(metric_name)
                        logger.debug(
                            f"Set key metric to '{metric_name}' based on single-device pattern '{priority_pattern}'"
                        )
                        return

        # Strategy 4: Fallback - use the first metric with highest value
        metric_names = list(self.metrics.keys())
        if metric_names:
            best_metric = None
            best_value = -1

            # Try to find metric with numeric value > 0
            for metric_name in metric_names:
                metric_value = getattr(self.metrics[metric_name], "value", 0)
                if isinstance(metric_value, (int, float)) and metric_value > best_value:
                    best_value = metric_value
                    best_metric = metric_name

            # If we found a good metric, use it; otherwise use the first one
            selected_metric = best_metric if best_metric else metric_names[0]
            self.set_key_metric(selected_metric)
            logger.debug(f"Set key metric to fallback selection '{selected_metric}'")

    def apply_config_metadata(self, configs: Dict[str, Any]):
        """
        Apply metadata from configuration (profile and test level) to the Result instance.

        This method merges metadata from both profile-level and test-level configurations,
        with test-level metadata taking precedence over profile-level metadata.

        Args:
            configs: Test configuration dictionary that may contain metadata
        """
        merged_metadata = self._merge_metadata_from_configs(configs)

        # Apply merged metadata to the Result instance
        for key, value in merged_metadata.items():
            # Don't override existing metadata that was explicitly set
            if key not in self.metadata:
                self.metadata[key] = value

    def _merge_metadata_from_configs(self, configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge metadata from profile and test configurations.

        Args:
            configs: Test configuration dictionary

        Returns:
            Dict[str, Any]: Merged metadata dictionary
        """
        merged_metadata = {}

        # Get profile-level metadata from active profile configurations
        try:
            from sysagent.utils.config import get_active_profile_configs

            profile_configs = get_active_profile_configs()

            if profile_configs and "params" in profile_configs and "metadata" in profile_configs["params"]:
                profile_metadata = profile_configs["params"]["metadata"]

                # Handle metadata as list of key-value pairs (as seen in metro.yml)
                if isinstance(profile_metadata, list):
                    for item in profile_metadata:
                        if isinstance(item, dict) and "key" in item and "value" in item:
                            merged_metadata[item["key"]] = item["value"]
                # Handle metadata as dict
                elif isinstance(profile_metadata, dict):
                    merged_metadata.update(profile_metadata)

        except Exception:
            # Silently ignore if profile configs can't be loaded
            pass

        # Apply test-level metadata (takes precedence over profile-level)
        if "metadata" in configs:
            test_metadata = configs["metadata"]

            # Handle metadata as list of key-value pairs
            if isinstance(test_metadata, list):
                for item in test_metadata:
                    if isinstance(item, dict) and "key" in item and "value" in item:
                        merged_metadata[item["key"]] = item["value"]
            # Handle metadata as dict
            elif isinstance(test_metadata, dict):
                merged_metadata.update(test_metadata)

        return merged_metadata

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Result":
        """
        Create a Result instance from a dictionary.

        Args:
            data: Dictionary containing result data

        Returns:
            Result instance
        """
        metrics = {}
        for k, v in data.get("metrics", {}).items():
            metrics[k] = Metrics(**v) if isinstance(v, dict) else v

        return cls(
            name=data.get("name", "T0000"),
            parameters=data.get("parameters", {}),
            metrics=metrics,
            metadata=data.get("metadata", {}),
            extended_metadata=data.get("extended_metadata", {}),
            kpis=data.get("kpis", {}),
        )

    @classmethod
    def from_test_config(cls, configs: Dict[str, Any], **kwargs) -> "Result":
        """
        Create a Result instance with automatic metadata application from test configs.

        This is a convenience method that creates a Result instance and automatically
        applies metadata from both profile-level and test-level configurations.

        Args:
            configs: Test configuration dictionary containing metadata and other settings
            **kwargs: Additional arguments to pass to the Result constructor

        Returns:
            Result instance with applied configuration metadata
        """
        # Extract display name for the result name if available
        if "display_name" in configs and "name" not in kwargs:
            test_id = configs.get("test_id", "T0000")
            display_name = configs["display_name"]
            kwargs["name"] = f"{test_id} - {display_name}"

        # Create the Result instance
        result = cls(**kwargs)

        # Apply configuration metadata
        result.apply_config_metadata(configs)

        return result


def get_metric_name_for_device(device_id, device_type=None, prefix="metric"):
    """
    Returns a metric name with the given prefix, customized for device_id and device_type.

    Supports indexed dGPU metrics (dgpu1, dgpu2, etc.) and HETERO device metrics.

    Args:
        device_id: OpenVINO device ID (e.g., "CPU", "GPU.0", "GPU.1", "HETERO:GPU.0,GPU.1")
        device_type: Device type string (optional, will be detected if not provided)
        prefix: Metric name prefix (default: "metric")

    Returns:
        Metric name string (e.g., "throughput_cpu", "throughput_dgpu1", "throughput_hetero_dgpu")
    """
    device_id_lower = device_id.lower()

    # Handle HETERO devices
    if device_id.upper().startswith("HETERO:"):
        return f"{prefix}_hetero_dgpu"

    # Handle standard devices
    if device_id_lower == "cpu":
        return f"{prefix}_cpu"
    elif device_id_lower == "npu":
        return f"{prefix}_npu"
    elif device_id.upper().startswith("GPU"):
        # If device_type is not provided, try to detect it
        if device_type is None:
            from sysagent.utils.system.ov_helper import get_openvino_device_type

            device_type = get_openvino_device_type(device_id)

        # Handle integrated GPU
        if device_type == "Type.INTEGRATED":
            return f"{prefix}_igpu"

        # Handle discrete GPU with indexing
        elif device_type == "Type.DISCRETE":
            # Extract GPU index from device_id (e.g., "GPU.0" -> "0", "GPU.1" -> "1")
            if "." in device_id:
                gpu_index = device_id.split(".")[1]
                return f"{prefix}_dgpu{gpu_index}"
            else:
                # Fallback for GPU without index
                return f"{prefix}_dgpu"

    return prefix
