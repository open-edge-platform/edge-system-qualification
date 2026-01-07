# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Execution utilities for core testing framework.

This module provides fixtures and functions for executing tests with
standardized logging and error handling.
"""

import json
import logging
from typing import Any, Callable, Dict, Optional, Union

import allure
import pytest

from sysagent.utils.core import Result
from sysagent.utils.reporting import update_allure_title_with_metrics

logger = logging.getLogger(__name__)


def _make_serializable(obj):
    """
    Recursively convert objects to JSON-serializable types.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Convert objects with __dict__ to string representation
        return str(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Convert any other type to string
        return str(obj)


# Global container log collector for consolidation during test execution
_CONTAINER_LOGS_COLLECTOR = []


def add_container_log_for_consolidation(container_name: str, log_text: str, exit_code: int = None):
    """Add container logs to the global collector for consolidation."""
    global _CONTAINER_LOGS_COLLECTOR
    _CONTAINER_LOGS_COLLECTOR.append((container_name, log_text, exit_code))


def get_and_clear_container_logs_for_consolidation():
    """Get and clear all collected container logs."""
    global _CONTAINER_LOGS_COLLECTOR
    logs = _CONTAINER_LOGS_COLLECTOR.copy()
    _CONTAINER_LOGS_COLLECTOR.clear()
    return logs


@pytest.fixture(scope="function")
def execute_test_with_cache():
    """
    Fixture that provides a function to execute tests with caching support.

    This fixture standardizes the way tests are executed with result caching.
    It handles the caching logic, logging, and error handling.

    Usage in tests:
        results = execute_test_with_cache(
            cached_result=cached_result,
            cache_result=cache_result,
            run_test_func=run_test_function,
            test_name="test_name",
            configs=configs,
            cache_configs=cache_configs
        )

    Returns:
        A function that executes a test with caching support
    """

    def _execute_test_with_cache(
        cached_result: Optional[Union[Result, Dict[str, Any]]],
        cache_result: Callable[[Union[Result, Dict[str, Any]]], None],
        run_test_func: Callable[[], Union[Result, Dict[str, Any]]],
        test_name: str,
        configs: Dict[str, Any],
        cache_configs: Dict[str, Any] = None,
        update_title_with_metrics: bool = False,
        name: str = "Analysis",
    ) -> Union[Result, Dict[str, Any]]:
        """
        Execute a test with caching support.

        Args:
            cached_result: Cached test result if available
            cache_result: Function to cache test result
            run_test_func: Function to run the test
            test_name: Name of the test for logging
            configs: Test configurations
            cache_configs: Specific configurations for caching

        Returns:
            Result: The result of the test execution
        """
        update_title = update_allure_title_with_metrics()

        # Check if cached result is available
        cached_result_data = cached_result(cache_configs=cache_configs)
        if cached_result_data:
            logger.debug(f"Using cached '{name}' result for test: {test_name}")

            from sysagent.utils.core import TestResultCache

            test_cache = TestResultCache()

            with allure.step(f"Execute test - {name} ⚪"):
                cache_key_config = test_cache.get_cache_key_and_config(test_name, configs, cache_configs)
                allure.attach(
                    json.dumps(cache_key_config, indent=2),
                    name=f"Cache Key Config - {name}",
                    attachment_type=allure.attachment_type.JSON,
                )
                if isinstance(cached_result_data, Result):
                    test_id = configs.get("test_id", "T0000")
                    display_name = configs.get("display_name", test_name)
                    cached_result_data.name = f"{test_id} - {display_name}"

                if isinstance(cached_result_data, Result):
                    cached_result_json = cached_result_data.to_dict()
                else:
                    cached_result_json = cached_result_data
                allure.attach(
                    json.dumps(cached_result_json, indent=2),
                    name=f"Results (Cached) - {name}",
                    attachment_type=allure.attachment_type.JSON,
                )
            if update_title_with_metrics:
                update_title(configs, cached_result_data)
            return cached_result_data

        # Execute the test
        logger.info(f"Executing '{name}' for test: {test_name}")
        with allure.step(f"Execute test - {name}"):
            try:
                # Set environment flag to prevent automatic container log attachments
                import os

                previous_flag = os.environ.get("CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS", "")
                os.environ["CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS"] = "1"

                test_exception = None
                try:
                    results = run_test_func()
                    logger.info(f"Test {test_name} executed successfully")
                except Exception as e:
                    test_exception = e
                    results = None
                finally:
                    # Restore previous flag
                    if previous_flag:
                        os.environ["CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS"] = previous_flag
                    else:
                        os.environ.pop("CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS", None)

                    # Always collect & attach container logs, even if test failed
                    try:
                        collected_logs = get_and_clear_container_logs_for_consolidation()
                        if collected_logs:
                            container_logs_consolidated = []
                            for item in collected_logs:
                                if len(item) == 2:
                                    container_name, log_text = item
                                    exit_code = None
                                else:
                                    container_name, log_text, exit_code = item

                                if log_text:
                                    exit_code_info = f" (Exit Code: {exit_code})" if exit_code is not None else ""
                                    container_logs_consolidated.append(
                                        f"=== Container: {container_name}{exit_code_info} ===\n{log_text}\n"
                                    )

                            if container_logs_consolidated:
                                consolidated_logs = "\n".join(container_logs_consolidated)
                                allure.attach(
                                    consolidated_logs,
                                    name=f"Container Logs - {name}",
                                    attachment_type=allure.attachment_type.TEXT,
                                )
                                logger.debug(
                                    f"Attached consolidated container logs for "
                                    f"{len(container_logs_consolidated)} containers"
                                )
                    except Exception as e:
                        logger.debug(f"Could not collect container logs for consolidation: {e}")

                # Re-raise the test exception if one occurred
                if test_exception:
                    raise test_exception

                if isinstance(results, Result):
                    test_id = configs.get("test_id", "T0000")
                    display_name = configs.get("display_name", test_name)
                    results.name = f"{test_id} - {display_name}"

                # Attach the executed result for reporting
                if isinstance(results, Result):
                    results_json = results.to_dict()
                else:
                    results_json = results

                # Safe JSON serialization with fallback
                try:
                    json_data = json.dumps(results_json, indent=2)
                except TypeError as e:
                    logger.warning(f"JSON serialization failed for {name}: {e}")
                    safe_results = _make_serializable(results_json)
                    json_data = json.dumps(safe_results, indent=2)

                allure.attach(
                    json_data,
                    name=f"Results - {name}",
                    attachment_type=allure.attachment_type.JSON,
                )

                # Check for results status
                status = results.metadata.get("status", True)
                if status is False:
                    logger.error(f"Test {test_name} failed or is broken")
                    # pytest.fail(f"{results.parameters.get('error', 'Unknown error')}")
                else:
                    # Cache the result if status is pass
                    cache_result(results, cache_configs=cache_configs)
                    logger.debug(f"Cached '{name}' result for {test_name}")

                if update_title_with_metrics:
                    update_title(configs, results)
                return results
            except KeyboardInterrupt:
                error_message = "Test execution interrupted by user (KeyboardInterrupt)"
                logger.error(error_message)
                raise
            except Exception as e:
                logger.error(f"Error executing '{name}' for {test_name}: {e}")
                raise

    return _execute_test_with_cache


@pytest.fixture(scope="function")
def prepare_test():
    """
    Fixture that provides a function to prepare test execution.

    This fixture standardizes the way tests are prepared, including
    logging, error handling, and reporting.

    Usage in tests:
        prepare_data = prepare_test(
            test_name="test_name",
            configs=configs,
            prepare_func=prepare_function,
            preparation_params={...}
            cached_result=cached_result  # Optional
        )

    Returns:
        A function that prepares a test for execution
    """

    def _prepare_test(
        prepare_func: Callable[[], Dict[str, Any]],
        test_name: str,
        configs: Dict[str, Any],
        cached_result: Optional[Union[Result, Dict[str, Any]]] = None,
        cache_result: Callable[[Union[Result, Dict[str, Any]]], None] = None,
        cache_configs: Dict[str, Any] = None,
        name: str = "Configuration",
    ) -> Union[Result, Dict[str, Any]]:
        """
        Prepare a test for execution.

        Args:
            test_name: Name of the preparation test
            configs: Test configurations
            prepare_func: Function to prepare the test
            cached_result: Cached result if available
            name: Name for the preparation step

        Returns:
            Dict[str, Any]: Preparation results
        """
        # Check if cached result is available
        if cached_result and cache_result:
            cached_result_data = cached_result(cache_configs=cache_configs)
            if cached_result_data:
                logger.debug(f"Using cached '{name}' for test: {test_name}")
                from sysagent.utils.core import TestResultCache

                test_cache = TestResultCache()

                with allure.step(f"Prepare test - {name} ⚪"):
                    cache_key_config = test_cache.get_cache_key_and_config(test_name, configs, cache_configs)
                    allure.attach(
                        json.dumps(cache_key_config, indent=2),
                        name=f"Cache Key Config - {name}",
                        attachment_type=allure.attachment_type.JSON,
                    )
                    if isinstance(cached_result_data, Result):
                        test_id = configs.get("test_id", "T0000")
                        display_name = configs.get("display_name", test_name)
                        cached_result_data.name = f"{test_id} - {display_name}"

                    if isinstance(cached_result_data, Result):
                        cached_result_json = cached_result_data.to_dict()
                    else:
                        cached_result_json = cached_result_data
                    allure.attach(
                        json.dumps(cached_result_json, indent=2),
                        name=f"Preparation Data (Cached) - {name}",
                        attachment_type=allure.attachment_type.JSON,
                    )
                return cached_result_data

        with allure.step(f"Prepare test - {name}"):
            try:
                results = prepare_func()

                # Attach the executed result for reporting
                if isinstance(results, Result):
                    results_json = results.to_dict()
                else:
                    results_json = results

                # Safe JSON serialization with fallback
                try:
                    json_data = json.dumps(results_json, indent=2)
                except TypeError as e:
                    logger.warning(f"JSON serialization failed for {name}: {e}")
                    # Create a safe version by converting problematic objects to strings
                    safe_results = _make_serializable(results_json)
                    json_data = json.dumps(safe_results, indent=2)

                allure.attach(
                    json_data,
                    name=f"Preparation Data - {name}",
                    attachment_type=allure.attachment_type.JSON,
                )
                # Check for results status
                status = results.metadata.get("status", True)
                if status is False:
                    logger.error(f"Test {test_name} failed or is broken")
                    # Optionally attach the failed result for reporting
                    # Safe JSON serialization with fallback
                    try:
                        json_data = json.dumps(results_json, indent=2)
                    except TypeError as e:
                        logger.warning(f"JSON serialization failed for failed result: {e}")
                        safe_results = _make_serializable(results_json)
                        json_data = json.dumps(safe_results, indent=2)

                    allure.attach(
                        json_data,
                        name="Failed Preparation Result",
                        attachment_type=allure.attachment_type.JSON,
                    )
                    # pytest.fail(f"{results.metadata.get('error', 'Unknown error')}")

                # Cache the result only if it's not an error state
                # Don't cache results with status=False (failed/error states)
                should_cache = True
                if cache_result:
                    if isinstance(results, Result):
                        status = results.metadata.get("status", True)
                        if status is False:
                            should_cache = False
                            logger.debug(f"Skipping cache for '{name}' due to failed status (status={status})")

                    if should_cache:
                        cache_result(results, cache_configs=cache_configs)
                        logger.debug(f"Cached '{name}' for test: {test_name}")

                return results
            except KeyboardInterrupt:
                error_message = "Test preparation interrupted by user"
                logger.error(error_message)
                raise
            except Exception as e:
                logger.error(f"Error preparing '{name}' for test {test_name}: {e}")
                raise

    return _prepare_test
