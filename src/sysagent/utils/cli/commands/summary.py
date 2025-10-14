# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Summary generation command implementation.

Handles generating JSON summaries from existing test results
with optional detailed tables and custom output files.
"""
import os
import json
import logging

from sysagent.utils.config import setup_data_dir
from sysagent.utils.logging import setup_command_logging
from sysagent.utils.reporting import CoreResultsSummaryGenerator, TestSummaryTableGenerator

logger = logging.getLogger(__name__)


def generate_summary(output_file: str = None, verbose: bool = False, debug: bool = False) -> int:
    """
    Generate JSON summary from existing test results.
    
    Args:
        output_file: Custom output filename for the summary JSON
        verbose: Whether to include detailed test information
        debug: Whether to show debug level logs
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set up logging for this operation
        setup_command_logging("summary", verbose=True, debug=debug)
        
        # Setup data directory
        data_dir = setup_data_dir()
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory does not exist: {data_dir}")
            return 1
        
        # Generate summary
        logger.info("Generating test results summary from existing Allure results")
        summary_generator = CoreResultsSummaryGenerator(data_dir)
        
        try:
            summary_filepath = summary_generator.generate_and_save_summary(
                verbose=verbose,
                filename=output_file
            )
            
            # Load summary data to check if there are actual test results
            with open(summary_filepath, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # Check if there are actual test results before displaying summary
            summary_info = summary_data.get("summary", {})
            total_tests = summary_info.get("total_tests", 0)
            
            if total_tests > 0:
                # Generate and display summary table only if there are test results
                table_generator = TestSummaryTableGenerator(summary_data)
                summary_table = table_generator.generate_summary_table()
                detailed_table = table_generator.generate_detailed_test_table()
                
                # Always log summary tables to both log files and console
                # Use different log levels based on verbose/debug flags for console output
                if verbose or debug:
                    # Show detailed summary in console for verbose/debug mode
                    logger.info("\n\n" + summary_table + "\n" + detailed_table)
                else:
                    # Show basic summary table only in console for non-verbose mode
                    logger.info("\n\n" + summary_table)

            return 0
            
        except Exception as e:
            logger.error(f"Failed to generate test results summary: {e}")
            return 1
    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return 1
