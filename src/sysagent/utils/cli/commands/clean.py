# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Clean command implementation.

Handles cleaning various data directories including logs, test results,
Allure reports, cache, and third-party installations.
"""
import os
import shutil
import logging

from sysagent.utils.config import setup_data_dir
from sysagent.utils.logging import setup_command_logging
from sysagent.utils.reporting import ALLURE_DIR_NAME

logger = logging.getLogger(__name__)


def clean_data_dir(
    clean_cache: bool = False,
    clean_thirdparty: bool = False,
    clean_data: bool = False,
    clean_all: bool = False,
    cache_only: bool = False,
    thirdparty_only: bool = False,
    data_only: bool = False,
    verbose: bool = False,
    debug: bool = False
) -> int:
    """
    Clean the data directory by removing logs, test results, Allure reports, and Allure history.
    
    Args:
        clean_cache: Whether to clean the cache directory as well (in addition to results)
        clean_thirdparty: Whether to clean the thirdparty directory as well (in addition to results)
        clean_data: Whether to clean the entire data directory as well (in addition to results)
        clean_all: Whether to clean all directories (equivalent to setting all other clean flags to True)
        cache_only: Whether to clean only the cache directory (skip results/logs cleaning)
        thirdparty_only: Whether to clean only the thirdparty directory (skip results/logs cleaning) 
        data_only: Whether to clean only the data directory (skip results/logs cleaning)
        verbose: Whether to show more detailed output
        debug: Whether to show debug level logs
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        data_dir = setup_data_dir()
        
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory does not exist: {data_dir}")
            return 0
        
        # Set up logging for this operation
        setup_command_logging("clean", verbose=verbose, debug=debug)
        
        # Check if any "only" options are specified
        only_options = cache_only or thirdparty_only or data_only
        cleaned_items = []
        
        if only_options:
            # Only clean specific directories, skip default cleaning
            if cache_only:
                logger.info("Cleaning only cache directory")
                _clean_cache_directory(data_dir)
                cleaned_items.append("cache")
            
            if thirdparty_only:
                logger.info("Cleaning only thirdparty directory")
                _clean_thirdparty_directory(data_dir)
                cleaned_items.append("thirdparty")
            
            if data_only:
                logger.info("Cleaning only data directory")
                _clean_application_data_directory(data_dir)
                cleaned_items.append("application data")
                
        else:
            # Default behavior: clean results/logs + any additional specified directories
            
            # If clean_all is True, set all other clean flags to True
            if clean_all:
                clean_cache = True
                clean_thirdparty = True
                clean_data = True
                logger.info("Cleaning all directories (cache, thirdparty, and data)")
            
            # Clean standard directories (results, logs, reports)
            _clean_logs_directory(data_dir)
            _clean_allure_results_directory(data_dir)
            _clean_core_results_directory(data_dir)
            _clean_allure_reports_directory(data_dir)
            cleaned_items.extend(["logs", "results", "reports"])
            
            # Clean optional directories based on flags
            if clean_cache:
                _clean_cache_directory(data_dir)
                cleaned_items.append("cache")
            
            if clean_thirdparty:
                _clean_thirdparty_directory(data_dir)
                cleaned_items.append("thirdparty")
            else:
                _clean_allure_history(data_dir)
                cleaned_items.append("Allure history")
            
            if clean_data:
                _clean_application_data_directory(data_dir)
                cleaned_items.append("application data")
        
        # Show specific success message based on what was cleaned
        if len(cleaned_items) == 1:
            logger.info(f"Successfully cleaned {cleaned_items[0]} directory")
        elif len(cleaned_items) > 1:
            items_str = ", ".join(cleaned_items[:-1]) + f" and {cleaned_items[-1]}"
            logger.info(f"Successfully cleaned {items_str} directories")
        else:
            logger.info("No directories were cleaned")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error cleaning data directory: {e}")
        return 1


def _clean_logs_directory(data_dir: str):
    """Clean the logs directory."""
    logs_dir = os.path.join(data_dir, "logs")
    if os.path.exists(logs_dir):
        logger.info(f"Cleaning logs directory: {logs_dir}")
        for filename in os.listdir(logs_dir):
            file_path = os.path.join(logs_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Removed file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")


def _clean_allure_results_directory(data_dir: str):
    """Clean the Allure results directory."""
    allure_dir = os.path.join(data_dir, "results", "allure")
    if os.path.exists(allure_dir):
        logger.info(f"Cleaning allure results directory: {allure_dir}")
        for filename in os.listdir(allure_dir):
            file_path = os.path.join(allure_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.debug(f"Removed directory: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file/directory {file_path}: {e}")


def _clean_core_results_directory(data_dir: str):
    """Clean the core results directory."""
    core_dir = os.path.join(data_dir, "results", "core")
    if os.path.exists(core_dir):
        logger.info(f"Cleaning core results directory: {core_dir}")
        for filename in os.listdir(core_dir):
            file_path = os.path.join(core_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.debug(f"Removed directory: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file/directory {file_path}: {e}")


def _clean_allure_reports_directory(data_dir: str):
    """Clean the Allure reports directory."""
    allure_reports_dir = os.path.join(data_dir, "reports", "allure")
    if os.path.exists(allure_reports_dir):
        logger.info(f"Cleaning allure reports directory: {allure_reports_dir}")
        try:
            # Use shutil.rmtree to remove the entire directory structure
            shutil.rmtree(allure_reports_dir)
            logger.debug(f"Removed allure reports directory: {allure_reports_dir}")
            # Recreate the empty directory
            os.makedirs(allure_reports_dir, exist_ok=True)
            logger.debug(f"Recreated empty allure reports directory")
        except Exception as e:
            logger.error(f"Error removing allure reports directory {allure_reports_dir}: {e}")


def _clean_cache_directory(data_dir: str):
    """Clean the cache directory."""
    cache_dir = os.path.join(data_dir, "cache")
    if os.path.exists(cache_dir):
        logger.info(f"Cleaning cache directory: {cache_dir}")
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Removed file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {e}")


def _clean_thirdparty_directory(data_dir: str):
    """Clean the entire thirdparty directory."""
    thirdparty_dir = os.path.join(data_dir, "thirdparty")
    if os.path.exists(thirdparty_dir):
        logger.info(f"Cleaning thirdparty directory: {thirdparty_dir}")
        try:
            # Use shutil.rmtree to remove the entire directory
            shutil.rmtree(thirdparty_dir)
            logger.debug(f"Removed thirdparty directory: {thirdparty_dir}")
            # Recreate the empty directory
            os.makedirs(thirdparty_dir, exist_ok=True)
            logger.debug(f"Recreated empty thirdparty directory")
        except Exception as e:
            logger.error(f"Error removing thirdparty directory {thirdparty_dir}: {e}")


def _clean_allure_history(data_dir: str):
    """Clean only Allure history while preserving other thirdparty files."""
    thirdparty_dir = os.path.join(data_dir, "thirdparty")
    if os.path.exists(thirdparty_dir):
        logger.info("Cleaning Allure history while preserving other thirdparty files")
        
        # Clean Allure .allure directory which contains history
        allure_dot_dir = os.path.join(thirdparty_dir, ALLURE_DIR_NAME, ".allure")
        if os.path.exists(allure_dot_dir):
            try:
                logger.info(f"Cleaning Allure history directory: {allure_dot_dir}")
                shutil.rmtree(allure_dot_dir)
                logger.debug(f"Removed Allure history directory: {allure_dot_dir}")
            except Exception as e:
                logger.error(f"Error removing Allure history directory {allure_dot_dir}: {e}")


def _clean_application_data_directory(data_dir: str):
    """Clean the entire application data directory."""
    app_data_dir = os.path.join(data_dir, "data")
    if os.path.exists(app_data_dir):
        logger.info(f"Cleaning application data directory: {app_data_dir}")
        try:
            # Use shutil.rmtree to remove the entire directory
            shutil.rmtree(app_data_dir)
            logger.debug(f"Removed data directory: {app_data_dir}")
            # Recreate the empty directory
            os.makedirs(app_data_dir, exist_ok=True)
            logger.debug(f"Recreated empty data directory")
        except Exception as e:
            logger.error(f"Error removing data directory {app_data_dir}: {e}")
