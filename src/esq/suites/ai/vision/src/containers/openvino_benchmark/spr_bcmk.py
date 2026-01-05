# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess   #nosec B404
import re
from pathlib import Path
import logging

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = Path(f'{CURR_DIR}/output').resolve()
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR/'ov_execution.log'),
        #logging.StreamHandler()
    ]
)

def extract_value(line):
    m = re.search(r'\d+\.\d+', line)
    try:
        return m.group()
    except:
        return ""

def fetch_cpu_info():
    # Function to execute a command and return its output
    def execute_command(command):
        result = subprocess.run(command, shell=False, text=True, capture_output=True)
        return result.stdout.strip()

    # Execute lscpu and capture its output
    lscpu_output = execute_command(["lscpu"])

    # Function to find a value using a regex pattern from the lscpu output
    def find_value(pattern):
        match = re.search(pattern, lscpu_output)
        if match:
            return int(match.group(1))
        return None

    # Extracting required information using regex
    sockets_num = find_value(r'Socket\(s\):\s*(\d+)')
    cores_per_socket = find_value(r'Core\(s\) per socket:\s*(\d+)')
    numa_nodes_num = find_value(r'NUMA node\(s\):\s*(\d+)')

    # Calculating values
    physical_cores_num = sockets_num * cores_per_socket if sockets_num and cores_per_socket else None
    cores_per_node = physical_cores_num // numa_nodes_num if physical_cores_num and numa_nodes_num else None
    cores_per_instance = cores_per_node

    # Returning the calculated values
    return {
        'sockets_num': sockets_num,
        'cores_per_socket': cores_per_socket,
        'physical_cores_num': physical_cores_num,
        'numa_nodes_num': numa_nodes_num,
        'cores_per_node': cores_per_node,
        'cores_per_instance': cores_per_instance
    }

def run_ov_bcmk(bcmk_dir, model_path, exec_cores_per_socket=0, time=90):
 
    benchmark_app_bin = bcmk_dir + "/benchmark_app"
    
    cpu_info = fetch_cpu_info()
    logging.info(f"Get CPU INFO: {cpu_info}")

    sockets_num = cpu_info['sockets_num']
    cores_per_socket = cpu_info['cores_per_socket']
    exec_cores_per_socket = cores_per_socket if exec_cores_per_socket <= 0 else exec_cores_per_socket
    
    numa_cmds = []
    for i, socket in enumerate(range(sockets_num)):
        start_core_idx = i * cores_per_socket
        end_core_idx = start_core_idx + exec_cores_per_socket -1
        numa_cmd = ["numactl", "-m", str(i), "--physcpubind", f"{start_core_idx}-{end_core_idx}", 
                    benchmark_app_bin, "-m", model_path, "-t", str(time)]
        # Note: CLIP vision model uses pixel_values input with shape [1,3,224,224]
        # No need to specify -data_shape parameter as model is already converted correctly
        logging.info(f"Benchmark App command with numactl: {numa_cmd}")
        numa_cmds.append(numa_cmd)

    raw_out = []
    try:
        exec_proc_list = []
        for _cmd in numa_cmds:
            proc = subprocess.Popen(_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
            exec_proc_list.append(proc)

        for proc in exec_proc_list:
            proc.wait(timeout= time + 30)

        for proc in exec_proc_list:
            stdout, stderr = proc.communicate()
            if stderr:
                logging.error(f"Error occures : {stderr}")
            raw_out.append(stdout)
    except subprocess.CalledProcessError as ex:
        print(f"Execute benchmark app with model {model_name} on SPR platform Failed. ")
        logging.error(ex.returncode)
        logging.error(ex.output)

    latency_value = 0
    throughput_value = 0
    for _out in raw_out:
        for line in _out.split('\n'):
            if line.strip().startswith("[ INFO ]    Average:"):
                latency_value += float(extract_value(line))
            elif line.strip().startswith("[ INFO ] Throughput:"):
                throughput_value += float(extract_value(line))
    return {'latency': round(latency_value, 2), 'throughput': round(throughput_value, 2)}
