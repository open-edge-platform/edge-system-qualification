#!/usr/bin/env python3

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Sysbench results analysis script for Docker container performance testing.
Parses sysbench output and generates JSON results for example testing framework.
"""

import json
import os
import re
import sys
from datetime import datetime


def parse_sysbench_output(test_type):
    """Parse sysbench output files and extract performance metrics."""

    # Additional security validation: ensure test_type contains only allowed characters
    # This prevents any path manipulation attempts
    VALID_TEST_TYPES = {"cpu", "memory", "fileio", "mutex"}
    if test_type not in VALID_TEST_TYPES:
        print(f"Error: Invalid test type '{test_type}' in parse function", file=sys.stderr)
        return None

    # Construct file path using validated test_type (safe from path manipulation)
    output_file = f"/tmp/sysbench_{test_type}.txt"

    # Additional security check: ensure the constructed path is within expected directory
    # and doesn't contain path traversal sequences
    expected_path = os.path.abspath(output_file)
    if not expected_path.startswith("/tmp/sysbench_") or ".." in output_file or "/" in test_type:
        print("Error: Invalid file path construction detected", file=sys.stderr)
        return None

    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} not found", file=sys.stderr)
        return None

    try:
        with open(output_file, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading output file: {e}", file=sys.stderr)
        return None

    # Initialize result structure
    result = {
        "test_type": test_type,
        "cpu_speed": 0.0,
        "memory_throughput": 0.0,
        "operations_per_second": 0.0,
        "performance_score": 0.0,
        "events_per_second": 0.0,
        "total_time": 0.0,
        "total_events": 0,
        "latency_ms": {"min": 0.0, "avg": 0.0, "max": 0.0, "95th_percentile": 0.0},
        "threads": 1,
        "test_config": {},
        "unit": "events/sec",
        "container_hostname": os.uname().nodename,
        "test_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Parse common metrics
    try:
        # Extract total events
        events_match = re.search(r"total number of events:\s*(\d+)", content)
        if events_match:
            result["total_events"] = int(events_match.group(1))

        # Extract total time
        time_match = re.search(r"total time:\s*([0-9.]+)s", content)
        if time_match:
            result["total_time"] = float(time_match.group(1))

        # Calculate events per second
        if result["total_time"] > 0:
            result["events_per_second"] = result["total_events"] / result["total_time"]

        # Extract threads
        threads_match = re.search(r"Number of threads:\s*(\d+)", content)
        if threads_match:
            result["threads"] = int(threads_match.group(1))

        # Extract latency information
        latency_section = re.search(r"Latency \(ms\):(.*?)(?=\n\n|\nThroughput|\Z)", content, re.DOTALL)
        if latency_section:
            latency_text = latency_section.group(1)

            # Extract min latency
            min_match = re.search(r"min:\s*([0-9.]+)", latency_text)
            if min_match:
                result["latency_ms"]["min"] = float(min_match.group(1))

            # Extract avg latency
            avg_match = re.search(r"avg:\s*([0-9.]+)", latency_text)
            if avg_match:
                result["latency_ms"]["avg"] = float(avg_match.group(1))

            # Extract max latency
            max_match = re.search(r"max:\s*([0-9.]+)", latency_text)
            if max_match:
                result["latency_ms"]["max"] = float(max_match.group(1))

            # Extract 95th percentile
            p95_match = re.search(r"95th percentile:\s*([0-9.]+)", latency_text)
            if p95_match:
                result["latency_ms"]["95th_percentile"] = float(p95_match.group(1))

        # Test-specific parsing
        if test_type == "cpu":
            result["unit"] = "events/sec"
            result["cpu_speed"] = result["events_per_second"]

            # Extract CPU-specific metrics
            cpu_match = re.search(r"events per second:\s*([0-9.]+)", content)
            if cpu_match:
                result["events_per_second"] = float(cpu_match.group(1))
                result["cpu_speed"] = result["events_per_second"]

        elif test_type == "memory":
            result["unit"] = "MiB/sec"

            # Extract memory throughput
            throughput_match = re.search(r"([0-9.]+) MiB/sec", content)
            if throughput_match:
                result["performance_score"] = float(throughput_match.group(1))
                result["memory_throughput"] = result["performance_score"]

            # Extract operations per second
            ops_match = re.search(r"([0-9.]+) ops/sec", content)
            if ops_match:
                result["operations_per_second"] = float(ops_match.group(1))

        elif test_type == "fileio":
            result["unit"] = "MB/sec"

            # Extract file I/O throughput
            read_match = re.search(r"read, MiB/s:\s*([0-9.]+)", content)
            write_match = re.search(r"written, MiB/s:\s*([0-9.]+)", content)

            if read_match:
                result["read_throughput"] = round(float(read_match.group(1)), 2)
            if write_match:
                result["write_throughput"] = round(float(write_match.group(1)), 2)

            # Calculate combined throughput as performance score
            read_throughput = result.get("read_throughput", 0.0)
            write_throughput = result.get("write_throughput", 0.0)
            result["performance_score"] = round(read_throughput + write_throughput, 2)

            # Extract IOPS
            iops_match = re.search(r"([0-9.]+) IOPS", content)
            if iops_match:
                result["iops"] = float(iops_match.group(1))

        elif test_type == "mutex":
            result["unit"] = "events/sec"
            result["performance_score"] = result["events_per_second"]

            # Extract mutex-specific metrics
            mutex_match = re.search(r"events per second:\s*([0-9.]+)", content)
            if mutex_match:
                result["events_per_second"] = float(mutex_match.group(1))
                result["performance_score"] = result["events_per_second"]

        # Add test configuration
        result["test_config"] = {
            "test_type": test_type,
            "threads": result["threads"],
            "duration_s": result["total_time"],
        }

        # Add raw output for debugging
        result["raw_output"] = content

    except Exception as e:
        print(f"Error parsing sysbench output: {e}", file=sys.stderr)
        result["error"] = f"parsing_failed: {str(e)}"
        result["raw_output"] = content

    return result


def main():
    """Main function to parse sysbench results."""

    if len(sys.argv) != 2:
        print("Usage: sysbench_analysis.py <test_type>", file=sys.stderr)
        print("Test types: cpu, memory, fileio, mutex", file=sys.stderr)
        sys.exit(1)

    # Define allowed test types as a security measure against path manipulation
    ALLOWED_TEST_TYPES = {"cpu": "cpu", "memory": "memory", "fileio": "fileio", "mutex": "mutex"}

    # Sanitize input by using allow-list approach
    raw_test_type = sys.argv[1].lower().strip()

    # Validate against allowed test types using dictionary lookup for security
    if raw_test_type not in ALLOWED_TEST_TYPES:
        print(f"Error: Invalid test type '{raw_test_type}'", file=sys.stderr)
        print("Valid test types: cpu, memory, fileio, mutex", file=sys.stderr)
        sys.exit(1)

    # Use the sanitized value from the allow-list
    test_type = ALLOWED_TEST_TYPES[raw_test_type]

    result = parse_sysbench_output(test_type)

    if result is None:
        # Create fallback result
        result = {
            "test_type": test_type,
            "cpu_speed": 0.0,
            "memory_throughput": 0.0,
            "performance_score": 0.0,
            "events_per_second": 0.0,
            "total_time": 0.0,
            "total_events": 0,
            "latency_ms": {"min": 0.0, "avg": 0.0, "max": 0.0, "95th_percentile": 0.0},
            "threads": 1,
            "test_config": {"test_type": test_type},
            "unit": "events/sec",
            "error": "output_file_not_found",
            "container_hostname": os.uname().nodename,
            "test_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    # Output JSON result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
