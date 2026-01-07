# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Caching system for test results to avoid redundant test execution.
"""

import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .result import Result

logger = logging.getLogger(__name__)


class TestResultCache:
    """
    Cache system for storing and retrieving test results.

    This cache enables test result reuse for different tests with the same configuration,
    reducing redundant test executions.
    """

    # Excluded fields for cache key generation because they do not affect test execution logic
    excluded_fields = [
        "configs",  # List of all configurations, not relevant for a single test run
        "description",  # Test description, not relevant for result caching
        "name",  # Test instance name, does not affect test logic
        "display_name",  # Human-readable label, not relevant for result caching
        "kpi_refs",  # KPI references, may change for reporting but not for execution
        "kpi_override",  # KPI override, not part of test logic
        "profile",  # Profile, not part of test logic
        "profile_name",  # Profile name, not part of test logic
        "requirements",  # Requirements, not part of test logic
        "severity",  # Severity, not part of test logic
        "test_id",  # Unique test identifier, not relevant for result caching
        "tiers",  # Test tier, not part of test logic
    ]

    def __init__(self, cache_dir: str = None):
        """
        Initialize the test result cache.

        Args:
            cache_dir: Directory to store cache files. If None, uses default data directory.
        """
        from sysagent.utils.config import setup_data_dir

        # Determine cache directory
        if cache_dir is None:
            data_dir = setup_data_dir()
            cache_dir = os.path.join(os.getcwd(), data_dir, "cache")

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_cache_key(self, test_config: Dict[str, Any]) -> str:
        """
        Generate a cache key for a test configuration.

        Args:
            test_config: Test configuration dictionary

        Returns:
            str: Cache key (SHA256 hash)
        """
        # Create a copy of the config and remove excluded fields
        config_for_key = test_config.copy()
        for field in self.excluded_fields:
            config_for_key.pop(field, None)

        # Serialize config to create consistent key
        config_json = json.dumps(config_for_key, sort_keys=True)
        cache_key = hashlib.sha256(config_json.encode()).hexdigest()

        logger.debug(f"Generated cache key: {cache_key}")
        return cache_key

    def _generate_cache_key(
        self, test_name: str, test_configs: Dict[str, Any], cache_configs: Dict[str, Any] = None
    ) -> tuple:
        """
        Generate a unique cache key for a test and its configuration.

        Args:
            test_name: Name of the test
            test_configs: Test configuration parameters
            cache_configs: Specific configurations for caching

        Returns:
            tuple: (cache_key, filtered_configs)

        Note:
            Excludes some fields from the cache key
            calculation to allow reuse of test results even when these fields change
        """
        # Create a copy of the configs to avoid modifying the original
        test_configs_copy = test_configs.copy()

        for key in self.excluded_fields:
            if key in test_configs_copy:
                test_configs_copy.pop(key)

        configs = cache_configs if cache_configs is not None else test_configs_copy
        logger.debug(f"Generating cache key for {test_name}")
        # Canonicalize the config by sorting keys
        config_str = json.dumps(configs, sort_keys=True)
        key_str = f"{test_name}:{config_str}"
        cache_key = hashlib.sha256(key_str.encode()).hexdigest()
        logger.debug(f"Generated cache key: {cache_key} for test {test_name}")
        return cache_key, configs

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def get_cache_key_and_config(
        self, test_name: str, test_configs: Dict[str, Any], cache_configs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract the cache key and the filtered configuration (with excluded fields removed)
        used to generate the cache key, as well as the list of excluded fields.

        Args:
            test_name: Name of the test
            test_configs: Test configuration parameters
            cache_configs: Specific configurations for caching

        Returns:
            Dict[str, Any]: {
                "cache_key": <cache_key>,
                "cache_configs": <filtered_config_dict>,
                "excluded_fields": <excluded_fields_list>
            }
        """
        logger.debug(f"Getting cache key and config for {test_name}")
        logger.debug(f"Cache configs: {cache_configs}")
        cache_key, key_cache_config = self._generate_cache_key(test_name, test_configs, cache_configs)
        return {"cache_key": cache_key, "cache_configs": key_cache_config, "excluded_fields": self.excluded_fields}

    def store(
        self, test_name: str, test_configs: Dict[str, Any], test_result, cache_configs: Dict[str, Any] = None
    ) -> str:
        """
        Store a test result in the cache.

        Args:
            test_name: Name of the test
            test_configs: Test configuration parameters
            test_result: Test result metrics to cache
            cache_configs: Specific configurations for caching

        Returns:
            str: The cache key for the stored result
        """
        cache_key, key_cache_configs = self._generate_cache_key(test_name, test_configs, cache_configs)
        cache_path = self._get_cache_path(cache_key)

        if hasattr(test_result, "to_dict"):
            test_result_to_store = test_result.to_dict()
        else:
            test_result_to_store = test_result

        with open(cache_path, "w") as f:
            json.dump(
                {
                    "test_name": test_name,
                    "test_configs": test_configs,
                    "cache_configs": key_cache_configs,
                    "test_result": test_result_to_store,
                    "cache_key": cache_key,
                    "timestamp": os.path.getmtime(cache_path) if os.path.exists(cache_path) else None,
                },
                f,
                indent=2,
            )

        logger.debug(f"Cached test result for {test_name} with key {cache_key}")
        return cache_key

    def retrieve(self, test_name: str, test_configs: Dict[str, Any], cache_configs: Dict[str, Any] = None):
        """
        Retrieve a test result from the cache.

        Args:
            test_name: Name of the test
            test_configs: Test configuration parameters
            cache_configs: Specific configurations for caching

        Returns:
            Optional[Result]: The cached test result or None if not found
        """
        cache_key, key_cache_config = self._generate_cache_key(test_name, test_configs, cache_configs)
        logger.debug(f"Retrieving cache for {test_name} [{cache_key}]")
        cache_path = self._get_cache_path(cache_key)

        if not os.path.exists(cache_path):
            logger.info(f"No cached result found for {test_name}")
            return None

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
                logger.debug(f"Retrieved cached result for {test_name} with key {cache_key}")
                result_dict = cache_data["test_result"]
                # Reconstruct Result object
                from .result import Metrics, Result

                parameters = result_dict.get("parameters", {})
                metrics_dict = result_dict.get("metrics", {})
                metrics = {k: Metrics(**v) for k, v in metrics_dict.items()}
                metadata = result_dict.get("metadata", {})
                extended_metadata = result_dict.get("extended_metadata", {})
                kpis = result_dict.get("kpis", {})
                return Result(
                    parameters=parameters,
                    metrics=metrics,
                    metadata=metadata,
                    extended_metadata=extended_metadata,
                    kpis=kpis,
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Error reading cache for {test_name}: {e}")
            return None

    def invalidate(self, test_name: str, test_configs: Dict[str, Any], cache_configs: Dict[str, Any] = None) -> bool:
        """
        Invalidate a cached test result.

        Args:
            test_name: Name of the test
            test_configs: Test configuration parameters
            cache_configs: Specific configurations for caching

        Returns:
            bool: True if cache was invalidated, False otherwise
        """
        cache_key, key_cache_config = self._generate_cache_key(test_name, test_configs, cache_configs)
        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.debug(f"Invalidating cache for {test_name} [{cache_key}] - config {key_cache_config}")
            return True

        logger.info(f"No cache to invalidate for {test_name}")
        return False

    def get_cache_file_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.

        Args:
            cache_key: Cache key

        Returns:
            str: Full path to cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def cache_exists(self, cache_key: str) -> bool:
        """
        Check if a cache entry exists.

        Args:
            cache_key: Cache key to check

        Returns:
            bool: True if cache exists, False otherwise
        """
        cache_file = self.get_cache_file_path(cache_key)
        exists = os.path.exists(cache_file)

        if exists:
            logger.debug(f"Cache hit for key: {cache_key}")
        else:
            logger.debug(f"Cache miss for key: {cache_key}")

        return exists

    def store_result(self, test_config: Dict[str, Any], result: "Result") -> None:
        """
        Store a test result in the cache.

        Args:
            test_config: Test configuration that was executed
            result: Test result to cache
        """

        cache_key = self.generate_cache_key(test_config)
        cache_file = self.get_cache_file_path(cache_key)

        try:
            # Convert result to dictionary for serialization
            result_dict = result.to_dict()

            cache_data = {"cache_key": cache_key, "test_config": test_config, "result": result_dict}

            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Stored result in cache: {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to store result in cache: {e}")

    def get_result(self, test_config: Dict[str, Any]) -> Optional["Result"]:
        """
        Retrieve a cached test result.

        Args:
            test_config: Test configuration to look up

        Returns:
            Result: Cached result if found, None otherwise
        """
        from .result import Result

        cache_key = self.generate_cache_key(test_config)

        if not self.cache_exists(cache_key):
            return None

        cache_file = self.get_cache_file_path(cache_key)

        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Reconstruct result from cached data
            result_dict = cache_data["result"]
            result = Result.from_dict(result_dict)

            logger.info(f"Retrieved cached result: {cache_key}")
            return result

        except Exception as e:
            logger.warning(f"Failed to retrieve cached result {cache_key}: {e}")
            return None

    def clear_cache(self) -> None:
        """
        Clear all cached results.
        """
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def clear_all(self) -> int:
        """
        Clear all cached results.

        Returns:
            int: Number of cache entries cleared
        """
        count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    cache_path = os.path.join(self.cache_dir, filename)
                    os.remove(cache_path)
                    count += 1
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

        logger.info(f"Cleared {count} cache entries")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict containing cache statistics
        """
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".json")]
            total_files = len(cache_files)

            total_size = 0
            for file in cache_files:
                file_path = os.path.join(self.cache_dir, file)
                total_size += os.path.getsize(file_path)

            return {"total_entries": total_files, "total_size_bytes": total_size, "cache_directory": self.cache_dir}
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"total_entries": 0, "total_size_bytes": 0, "cache_directory": self.cache_dir}
