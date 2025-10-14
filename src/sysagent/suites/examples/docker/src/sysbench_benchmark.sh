#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# sysbench_benchmark.sh - Script to measure system performance using sysbench
# This script will be executed inside the Docker container

set -e

# Default parameters
TEST_DURATION="${TEST_DURATION:-10}"
TEST_THREADS="${TEST_THREADS:-1}"
TEST_TYPE="${TEST_TYPE:-cpu}"
MEMORY_SIZE="${MEMORY_SIZE:-1G}"
FILE_SIZE="${FILE_SIZE:-1G}"

# Try to write to volume mount first, fallback to /app
if [ -d "/results" ]; then
    # Test if we can write to /results
    if touch "/results/test_write" 2>/dev/null; then
        rm -f "/results/test_write"
        OUTPUT_FILE="/results/results.json"
        echo "Using volume-mounted results directory: /results"
    else
        echo "WARNING: /results directory exists but is not writable, using /app"
        OUTPUT_FILE="/app/results.json"
    fi
else
    OUTPUT_FILE="/app/results.json"
    echo "Using local results directory: /app"
fi

echo "=== Docker Container Performance Test (Sysbench) ==="
echo "Test Type: $TEST_TYPE"
echo "Test Duration: $TEST_DURATION seconds"
echo "Test Threads: $TEST_THREADS"
echo "Memory Size: $MEMORY_SIZE"
echo "File Size: $FILE_SIZE"
echo "Output File: $OUTPUT_FILE"
echo "================================================"

# Container information
echo "Container Information:"
echo "- Hostname: $(hostname)"
echo "- IP Address: $(hostname -I | awk '{print $1}' || echo 'N/A')"
echo "- OS: $(grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"')"
echo "- Kernel: $(uname -r)"
echo "- CPU Info: $(nproc) cores, $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo

# Check sysbench version
echo "Sysbench version: $(sysbench --version 2>&1 || echo 'Unknown')"
echo

# Function to run CPU benchmark
run_cpu_benchmark() {
    echo "Running CPU benchmark..."
    echo "Testing CPU performance with prime number calculation"
    
    local output_file="/tmp/sysbench_cpu.txt"
    local max_prime=20000
    
    echo "Command: sysbench cpu --cpu-max-prime=$max_prime --threads=$TEST_THREADS --time=$TEST_DURATION run"
    
    sysbench cpu \
        --cpu-max-prime="$max_prime" \
        --threads="$TEST_THREADS" \
        --time="$TEST_DURATION" \
        run > "$output_file" 2>&1
    
    echo "CPU benchmark completed"
    cat "$output_file"
    echo
    
    return 0
}

# Function to run memory benchmark
run_memory_benchmark() {
    echo "Running memory benchmark..."
    echo "Testing memory performance with read/write operations"
    
    local output_file="/tmp/sysbench_memory.txt"
    
    echo "Command: sysbench memory --memory-block-size=1K --memory-total-size=$MEMORY_SIZE --threads=$TEST_THREADS --time=$TEST_DURATION run"
    
    sysbench memory \
        --memory-block-size=1K \
        --memory-total-size="$MEMORY_SIZE" \
        --threads="$TEST_THREADS" \
        --time="$TEST_DURATION" \
        run > "$output_file" 2>&1
    
    echo "Memory benchmark completed"
    cat "$output_file"
    echo
    
    return 0
}

# Function to run file I/O benchmark
run_fileio_benchmark() {
    echo "Running file I/O benchmark..."
    echo "Testing file system performance"
    
    local output_file="/tmp/sysbench_fileio.txt"
    local test_dir="/tmp/sysbench_fileio_test"
    
    # Create test directory
    mkdir -p "$test_dir"
    cd "$test_dir"
    
    echo "Preparing test files..."
    sysbench fileio \
        --file-total-size="$FILE_SIZE" \
        --file-test-mode=rndrw \
        --threads="$TEST_THREADS" \
        prepare > /dev/null 2>&1
    
    echo "Command: sysbench fileio --file-total-size=$FILE_SIZE --file-test-mode=rndrw --threads=$TEST_THREADS --time=$TEST_DURATION run"
    
    sysbench fileio \
        --file-total-size="$FILE_SIZE" \
        --file-test-mode=rndrw \
        --threads="$TEST_THREADS" \
        --time="$TEST_DURATION" \
        run > "$output_file" 2>&1
    
    echo "File I/O benchmark completed"
    cat "$output_file"
    echo
    
    # Cleanup
    echo "Cleaning up test files..."
    sysbench fileio \
        --file-total-size="$FILE_SIZE" \
        --file-test-mode=rndrw \
        --threads="$TEST_THREADS" \
        cleanup > /dev/null 2>&1
    
    cd /app
    rm -rf "$test_dir"
    
    return 0
}

# Function to run mutex benchmark
run_mutex_benchmark() {
    echo "Running mutex benchmark..."
    echo "Testing mutex performance"
    
    local output_file="/tmp/sysbench_mutex.txt"
    
    echo "Command: sysbench mutex --mutex-num=4096 --mutex-locks=10000 --mutex-loops=10000 --threads=$TEST_THREADS run"
    
    sysbench mutex \
        --mutex-num=4096 \
        --mutex-locks=10000 \
        --mutex-loops=10000 \
        --threads="$TEST_THREADS" \
        run > "$output_file" 2>&1
    
    echo "Mutex benchmark completed"
    cat "$output_file"
    echo
    
    return 0
}

# Function to analyze results and create JSON output
analyze_results() {
    echo "Analyzing results..."
    
    if python3 /app/sysbench_analysis.py "$TEST_TYPE" > $OUTPUT_FILE; then
        echo "Results analysis completed successfully"
        echo "Results saved to: $OUTPUT_FILE"
        
        # Display summary
        echo
        echo "Results Summary:"
        # Show summary without raw_output field
        if command -v jq >/dev/null 2>&1; then
            cat $OUTPUT_FILE | jq 'del(.raw_output)' 2>/dev/null || cat $OUTPUT_FILE
        else
            cat $OUTPUT_FILE
        fi
    else
        echo "ERROR: Failed to analyze sysbench results"
        # Create a fallback simple result
        cat > "$OUTPUT_FILE" << EOF
{
    "test_type": "$TEST_TYPE",
    "performance_score": 0.0,
    "events_per_second": 0.0,
    "total_time": $TEST_DURATION,
    "total_events": 0,
    "latency_ms": {
        "min": 0.0,
        "avg": 0.0,
        "max": 0.0,
        "95th_percentile": 0.0
    },
    "threads": $TEST_THREADS,
    "test_config": {
        "duration_s": $TEST_DURATION,
        "threads": $TEST_THREADS,
        "test_type": "$TEST_TYPE"
    },
    "unit": "events/sec",
    "error": "analysis_failed",
    "container_hostname": "$(hostname)",
    "test_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    fi
}

# Function to perform system diagnostics
perform_system_diagnostics() {
    echo
    echo "=== System Diagnostics ==="
    
    # CPU information
    echo "CPU Information:"
    echo "  Cores: $(nproc)"
    echo "  Architecture: $(uname -m)"
    echo "  CPU MHz: $(grep 'cpu MHz' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs || echo 'N/A')"
    
    # Memory information
    echo "Memory Information:"
    free -h | sed 's/^/  /'
    
    # Disk space
    echo "Disk Space:"
    df -h / | sed 's/^/  /'
    
    # Load average
    echo "Load Average:"
    echo "  $(cat /proc/loadavg)"
    
    # Process count
    echo "Process Count:"
    echo "  Total: $(ps aux | wc -l)"
    echo "  Running: $(ps aux | awk '$8~/^R/ {count++} END {print count+0}')"
    
    echo "========================="
}

# Cleanup function
cleanup() {
    echo
    echo "Cleaning up..."
    
    # Remove temporary files
    rm -f /tmp/sysbench_*.txt
    
    echo "Cleanup completed"
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    # Validate parameters
    if ! [[ "$TEST_DURATION" =~ ^[0-9]+$ ]] || [ "$TEST_DURATION" -lt 1 ]; then
        echo "ERROR: TEST_DURATION must be a positive integer (seconds)"
        exit 1
    fi
    
    if ! [[ "$TEST_THREADS" =~ ^[0-9]+$ ]] || [ "$TEST_THREADS" -lt 1 ]; then
        echo "ERROR: TEST_THREADS must be a positive integer"
        exit 1
    fi
    
    # Validate test type
    case "$TEST_TYPE" in
        cpu|memory|fileio|mutex)
            echo "Test type '$TEST_TYPE' is valid"
            ;;
        *)
            echo "ERROR: TEST_TYPE must be one of: cpu, memory, fileio, mutex"
            exit 1
            ;;
    esac
    
    # Check if sysbench is available
    if ! command -v sysbench >/dev/null 2>&1; then
        echo "ERROR: sysbench is not available. Please install sysbench package."
        exit 1
    fi
    
    # Perform system diagnostics
    perform_system_diagnostics
    
    # Run the appropriate benchmark
    case "$TEST_TYPE" in
        cpu)
            run_cpu_benchmark
            ;;
        memory)
            run_memory_benchmark
            ;;
        fileio)
            run_fileio_benchmark
            ;;
        mutex)
            run_mutex_benchmark
            ;;
    esac
    
    # Analyze results
    analyze_results
    
    echo
    echo "Performance test completed successfully!"
    echo "Results are available in: $OUTPUT_FILE"
}

# Run main function
main "$@"
