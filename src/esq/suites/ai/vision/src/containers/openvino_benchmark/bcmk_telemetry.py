# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess   #nosec B404
import shlex
import threading
import time
import os
import sys
import csv
from pathlib import Path
import copy
import re
import time
import traceback
from datetime import datetime
import logging

import numpy as np
import pandas as pd

from gpu_util import get_gpu_devices

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

GPU_TOP_RESULT_DICT = {
    'Freq MHz': ['req', 'act'],
    'IRQ RC6': ['/s', '%'],
    'Power W': ['gpu', 'pkg'],
    'IMC MiB/s': ['rd', 'wr'],
    'RCS': ['%', 'se', 'wa'],
    'RCS/0': ['%', 'se', 'wa'],
    'BCS': ['%', 'se', 'wa'],
    'VCS': ['%', 'se', 'wa'],
    'VCS/0': ['%', 'se', 'wa'],
    'VCS/1': ['%', 'se', 'wa'],
    'VECS': ['%', 'se', 'wa'],
    'CCS': ['%', 'se', 'wa'],
    'UNKN/0': ['%', 'se', 'wa']
}


def is_intel_xeon():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        if "Xeon" in cpuinfo:
            return True
        elif "XEON" in cpuinfo:
            return True
        else:
            return False
    except Exception as e:
        return False


def get_gpu_renders():
    renders = []
    #output = subprocess.check_output(["lsgpu"], stderr=subprocess.STDOUT, text=True)
    output = subprocess.check_output(["ls", "/dev/dri/"], stderr=subprocess.STDOUT, text=True)
    for line in output.split('\n'):
        if "renderD" in line and "drm:" in line:
            renders.append(line.strip().split()[-1])

    return renders

def get_physical_cores():
    try:
        # Execute the lscpu command and capture its output
        result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, text=True)
        output = result.stdout

        # Use regular expressions to find the number of cores per socket and the number of sockets
        cores_per_socket = int(re.search(r'Core\(s\) per socket:\s*(\d+)', output).group(1))
        sockets = int(re.search(r'Socket\(s\):\s*(\d+)', output).group(1))

        # Calculate the total number of physical cores
        total_physical_cores = cores_per_socket * sockets

        return total_physical_cores, sockets
    except Exception as e:
        logging.error(f"An error occurred when eval physical cores: {e}")
        return 1 

def cpu_usage_filter(line):
    return line is not None and len(line) > 0 and "procs" not in line and "swpd" not in line

def mem_usage_filter(line):
    return line is not None and len(line) > 0 and line.startswith("Mem:")

def mem_proc_usage_filter(line, key_wd="benchmark_app"):
    return line is not None and key_wd in line

def gpu_usage_filter(line):
    return line is not None and len(line.strip()) > 0

def dgpu_power_filter(line):
    return line is not None and len(line.strip()) > 0

# This function will run in a separate thread, collecting usage data
def collect_usage(command, output_file, event, filter_func):
    with open(output_file, "w") as f:
        # Explicitly detach from TTY by redirecting stdin to DEVNULL
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,  # Detach from TTY
            text=True
        )
        while not event.is_set():
            if "i7z" in command:
                time.sleep(1)
            else:
                line = process.stdout.readline()
                if filter_func(line):
                    f.write(line)
        if "i7z" in command:
            _filter_lines = []
            with open("i7z_output_raw.txt", 'r') as f:
                _filter_lines = f.readlines()

            cpu_phy_core_number, _ = get_physical_cores()
            filtered_lines = [l for l in _filter_lines if len(l.strip().split()) == (cpu_phy_core_number + 1)]

            with open("i7z_output.txt", 'w') as f:
                f.writelines(filtered_lines)

            process.kill()
            process.wait()
        else:
            process.terminate()

def fetch_bcmk_pids():
    bcmk_pids = []
    try_count = 0
    max_try = 6
    while len(bcmk_pids) == 0 and try_count < max_try:
        bcmk_pids = get_bcmk_pids()
        time.sleep(2)
        try_count += 1
    return bcmk_pids

def collect_mem_usage_after(output_file, event, filter_func):
    bcmk_pids = fetch_bcmk_pids()
    command = ["pidstat", "-r"]
    thread_mem_list = []
    for i, bcmk_pid in enumerate(bcmk_pids):
        command.append("-p")
        command.append(str(bcmk_pid))
    else:
        if bcmk_pids is None or len(bcmk_pids) == 0:
            logging.error("Cannot found benchmark_app process and collect its memory usage.")
    command.append("1")
    logging.info(f"Running Memory Usage gathering command {command}.")
    collect_usage(command, f"{output_file}.txt", event, filter_func)

def get_bcmk_pids():
    try:
        result = subprocess.run(['pgrep', "-f", 'benchmark_app'], capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        return [int(pid) for pid in pids if pid]
    except Exception as e:
        print(f"Error getting PIDs: {e}")
    return []

def print_trace():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    frames = traceback.extract_tb(exc_traceback)
    for frame in frames:
        logging.error(f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}, Code: {frame.line}")
    logging.error("Exception occurred,", exc_info=True)

def round_mean(mylist, precision=2):

    try:
        if mylist.size > 0:
            mean_value = round(mylist.mean(), precision)
        else:
            mean_value = 0 
    except:
        mean_value = 0
    return mean_value

def check_file(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def eval_cpu_freq(c_freq_file):
    if not check_file(c_freq_file):
        return -1.
    data = pd.read_csv(c_freq_file)
    column_data = data.iloc[:, 1]
    average_freq = column_data.mean()

    return round(average_freq/ 1000, 2)

def eval_cpu_usage(cpu_file):
    if not check_file(cpu_file):
        return -1.

    with open(cpu_file, "r") as f:
        lines = f.readlines()
        cpu_usage_list = []
        for line in lines:
            try:
                items = line.strip().split()
                one_cpu_usage = int(items[-5]) + int(items[-7]) + int(items[-8]) # -2/-4/-5 idx if no -t arg #-5 -7 -8
                cpu_usage_list.append(one_cpu_usage)
            except:
                logging.error("Exception for evaluating cpu usage.", exc_info=True)

        if len(cpu_usage_list) == 0:
            cpu_usage_list.append(0)
        _, sock_num = get_physical_cores()

        one_socket_cpu_usage = round(sum(cpu_usage_list) / len(cpu_usage_list), 2)
        if is_intel_xeon() and sock_num > 1:
            return 99.9 if one_socket_cpu_usage * 2 > 100. else one_socket_cpu_usage * 2 

        return one_socket_cpu_usage

def eval_mem_usage(mem_file):
    if not check_file(mem_file):
        return -1.

    with open(mem_file, "r") as f:
        lines = f.readlines()
        mem_proc_usage_list = []
        for line in lines:
            try:
                items = line.strip().split()
                one_mem_usage = float(items[-2])
                mem_proc_usage_list.append(one_mem_usage)
            except:
                logging.error("Exception for evaluating memory usage.", exc_info=True)

        if len(mem_proc_usage_list) == 0:
            mem_proc_usage_list.append(0)
        return round(sum(mem_proc_usage_list) / len(mem_proc_usage_list), 2)
    
def eval_mem_proc_usage(mem_file_name):

    mem_file = f"{mem_file_name}.txt"
    if not check_file(mem_file):
        return 0.0

    mem_data = {}
    with open(mem_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                items = line.strip().split()
                _time = items[0]
                one_mem_usage = float(items[-2])
                if _time in mem_data:
                    mem_data[_time].append(one_mem_usage)
                else:
                    mem_data[_time] = [one_mem_usage]
            except:
                logging.error("Exception for evaluating memory for benchmark_app usage.", exc_info=True)

    mem_usage_list = []
    for _t, _v in mem_data.items():
        single_mem_usage = sum(_v)
        mem_usage_list.append(single_mem_usage)
    if len(mem_usage_list) > 0:
        return round(sum(mem_usage_list)/len(mem_usage_list), 2)

    return 0.0

def eval_dgpu_power(dgpu_file):
    if not check_file(dgpu_file):
        return None

    lines = []
    with open(dgpu_file) as f:
        lines = f.readlines()
    if len(lines) < 5:
        return None

    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')

    dgpu_power_df = pd.read_csv(dgpu_file, sep=',')
    dgpu_power_df.columns = dgpu_power_df.columns.str.strip() #some space char in column

    dgpu_power_df['GPU Power (W)'] = pd.to_numeric(dgpu_power_df['GPU Power (W)'], errors='coerce')
    average_power = dgpu_power_df['GPU Power (W)'].mean()
    average_power = round(average_power, 2)
    return average_power

def eval_gpu_usage(gpu_file):
    if not check_file(gpu_file):
        return -1, -1, -1, (-1,), (-1,)
    lines = []
    with open(gpu_file) as f:
        lines = f.readlines()
    if len(lines) < 8:
        return -1, -1, -1, (-1,), (-1,)
    
    catelogy_line = lines[0]
    title_line = lines[1]
    data_lines = [l.strip() for l in lines if not l.strip().startswith('Freq') and not l.strip().startswith('req')]
    data_lines_2d = np.array([ [float(item) if item.replace('.', '', 1).isdigit() else np.nan for item in l.split()] for l in data_lines])
    data_lines_2d = data_lines_2d[4:-2]
    three_spaces = '   '
    two_spaces = '  '
    while three_spaces in catelogy_line:
        catelogy_line = catelogy_line.replace(three_spaces, two_spaces)

    result_category = catelogy_line.strip().split(two_spaces)
    subtitle_list = title_line.strip().split()

    rcs_0_count = catelogy_line.count("RCS/0")
    vcs_0_count = catelogy_line.count("VCS/0")
    vcs_1_count = catelogy_line.count("VCS/1")

    """ intel_gpu_top output in U24 OS:
    In Core: 
        1. dGPU power still need xpu-smi
        2. dGPU has CCS, no power info
            Freq MHz      IRQ RC6             RCS             BCS             VCS            VECS             CCS 
            req  act       /s   %       %  se  wa       %  se  wa       %  se  wa       %  se  wa       %  se  wa
        3. iGPU has no CCS
            Freq MHz      IRQ RC6     Power W             RCS             BCS             VCS            VECS 
            req  act       /s   %   gpu   pkg       %  se  wa       %  se  wa       %  se  wa       %  se  wa

    In Core Ultra:
        1. dGPU has CSS, no power info
            Freq MHz      IRQ RC6             RCS             BCS             VCS            VECS             CCS 
            req  act       /s   %       %  se  wa       %  se  wa       %  se  wa       %  se  wa       %  se  wa 
        2. iGPU has CCS, has pkg power, no gpu power on 165HL:
            Freq MHz      IRQ RC6             RCS             BCS             VCS            VECS             CCS 
            req  act       /s   %   pkg       %  se  wa       %  se  wa       %  se  wa       %  se  wa       %  se  wa
        3. iGPU has CCS, has Power W on 165H
        Freq MHz      IRQ RC6     Power W             RCS             BCS             VCS            VECS             CCS 
        req  act       /s   %   gpu   pkg       %  se  wa       %  se  wa       %  se  wa       %  se  wa       %  se  wa
        
    if has CCS, then using CCS replace RCS.
    """
  
    gpu_top_dict = copy.deepcopy(GPU_TOP_RESULT_DICT)

    if 'Power W' not in result_category and 'pkg' in subtitle_list: 
        gpu_top_dict['IRQ RC6'].append('pkg')
    
    gpu_result_idx_map = {}
    st_idx = 0 
    for _category in result_category:
        gpu_result_idx_map[_category] = {}
        sub_titles = gpu_top_dict.get(_category, [])
        for st in sub_titles:
            gpu_result_idx_map[_category][st] = st_idx
            st_idx += 1
 
    gpu_freq_idx = gpu_result_idx_map['Freq MHz']['act']
    gpu_freq_list = data_lines_2d[:, gpu_freq_idx].astype(float)
    avg_gpu_freq = round_mean(gpu_freq_list/1000)  #round(gpu_freq_list.mean()/1000., 2)

    gpu_vcs_0_idx = 0
    if vcs_0_count > 0:
        if vcs_1_count > 0:
            gpu_vcs_0_idx = (gpu_result_idx_map['VCS/0']['%'] + gpu_result_idx_map['VCS/1']['%'])/2
        else:
            gpu_vcs_0_idx = gpu_result_idx_map['VCS/0']['%']
    else:
        gpu_vcs_0_idx = gpu_result_idx_map['VCS']['%']
    gpu_vcs_0_idx=int(gpu_vcs_0_idx)
    gpu_vcs_0_list = data_lines_2d[:, gpu_vcs_0_idx].astype(float)
    avg_vcs0 = round_mean(gpu_vcs_0_list) #round(gpu_vcs_0_list.mean(), 2)

    avg_vcss = (avg_vcs0,)
    gpu_rcs0_idx = 0
    if rcs_0_count > 0:
        gpu_rcs0_idx = gpu_result_idx_map['RCS/0']['%']
    else:
        gpu_rcs0_idx = gpu_result_idx_map['RCS']['%']
    gpu_rcs0_idx=int(gpu_rcs0_idx)
    if "CCS" in result_category : #and "pkg" not in subtitle_list:
        gpu_rcs0_idx = gpu_result_idx_map['CCS']['%']

    gpu_rcs0_list = data_lines_2d[:, gpu_rcs0_idx].astype(float)
    avg_rcs0 = round_mean(gpu_rcs0_list) #round(gpu_rcs0_list.mean(), 2)


    try:
        pkg_power_idx = 0.
        if 'Power W' not in result_category and 'pkg' in subtitle_list:
            pkg_power_idx = gpu_result_idx_map['IRQ RC6']['pkg']
        else:
            pkg_power_idx = gpu_result_idx_map['Power W']['pkg']
        pkg_power_list = data_lines_2d[:, pkg_power_idx].astype(float)
        avg_pkg_power = round_mean(pkg_power_list) #round(pkg_power_list.mean(), 2)
    except:
        avg_pkg_power = 0.
    
    try:
        gpu_power_idx = gpu_result_idx_map['Power W']['gpu'] #only igpu so far
        gpu_power_list = data_lines_2d[:, gpu_power_idx].astype(float)
        avg_gpu_power = round_mean(gpu_power_list) #round(gpu_power_list.mean(), 2)
    except:
        avg_gpu_power = 0.

    return avg_gpu_freq, avg_gpu_power, avg_pkg_power, (avg_rcs0,), avg_vcss

def _start_cpu_usage_thread(event, out_file):
    thread_cpu = threading.Thread(target=collect_usage, args=(["vmstat", "-w", "-t", "1"], out_file, event, cpu_usage_filter))
    thread_cpu.start()
    return thread_cpu

def _start_cpu_freq_thread(event, out_file):

    def write_cpu_freq_to_file():
        open(out_file, "w").close()
        with open(out_file, "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            while not event.is_set():
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.readlines()
                mhz_values = [float(line.split(":")[1].strip()) for line in cpuinfo if "MHz" in line]
                avg_mhz = sum(mhz_values) / len(mhz_values) if mhz_values else 0
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csvwriter.writerow([current_date, avg_mhz])
                time.sleep(1)

    thread_c_freq = threading.Thread(target=write_cpu_freq_to_file)
    thread_c_freq.start()
    return thread_c_freq

def _start_mem_usage_thread(event, out_file):
    thread_mem = threading.Thread(target=collect_usage, args=(["free", "-w", "-s", "1"], out_file, event, mem_usage_filter))
    thread_mem.start()
    return thread_mem

def _start_mem_usage_4process_thread(event, out_file):
    thread_mem = threading.Thread(target=collect_mem_usage_after, args=(out_file, event, mem_proc_usage_filter))
    thread_mem.start()
    return thread_mem

def _start_gpu_usage_threads(event, gpu_devs):
    gpu_tds = []
    for _bcmk_dev, _dev_info in gpu_devs.items():
        rd_name = _dev_info['render_name']
        # Note: intel_gpu_top with -l flag should output to stdout without TTY interaction
        thread_gpu = threading.Thread(target=collect_usage, args=(["intel_gpu_top", "-d", f"drm:/dev/dri/{rd_name}", "-l"], f"gpu_usage_{_bcmk_dev}.txt", event, gpu_usage_filter))
        gpu_tds.append(thread_gpu)
    for g_td in gpu_tds:
        g_td.start()

    return gpu_tds

def _start_dgpu_power_threads(event, gpu_devs):
    dgpu_pw_tds = []
    for _bcmk_dev, _dev_info in gpu_devs.items():
        if _dev_info['device_type'] != 'dGPU':
            continue 
        rd_name = _dev_info['render_name']
        dev_id = _dev_info['device_id']
        thread_dgpu = threading.Thread(target=collect_usage, args=(["xpu-smi", "dump", "-d", str(dev_id), "-m", "1"], f"dgpu_power_dump_{_bcmk_dev}.txt", event, dgpu_power_filter))
        dgpu_pw_tds.append(thread_dgpu)
    
    for dg_td in dgpu_pw_tds:
        dg_td.start()

    return dgpu_pw_tds

def start_telemetry_thread(event, gpu_devs):
    cu_thread = _start_cpu_usage_thread(event, "cpu_usage.txt")
    cf_thread = _start_cpu_freq_thread(event, "cpu_avg_freq.txt")
    mem_thread = _start_mem_usage_thread(event, "memory_usage.txt")
    mem_bcmk_thread = _start_mem_usage_4process_thread(event, "memory_usage_bcmk")
    gpu_thread_list = []
    if len(gpu_devs) > 0:  #should be updated later
        gpu_thread_list = _start_gpu_usage_threads(event, gpu_devs)
        dgpu_power_thread = _start_dgpu_power_threads(event, gpu_devs) #using xpu-smi to get dgpu power
        if dgpu_power_thread is not None:
            gpu_thread_list += dgpu_power_thread

    return [cu_thread, cf_thread, mem_bcmk_thread] + gpu_thread_list

def telemetry_decorator(func):
    def wrapper(*args, **kwargs):

        event = threading.Event()
        gpu_devs = get_gpu_devices()
        thread_list = start_telemetry_thread(event, gpu_devs)

        result = func(*args, **kwargs)
        event.set()
        for td in thread_list:
            td.join()

        return update_telemetry(result, gpu_devs)

    return wrapper

def _eval_gpu_telemetry(out_result, gpu_devs):
    out_result['Package Power'] = -1. #init
    gpu_freq_list = []
    gpu_power_list = []
    gpu_rcs_usage = []
    gpu_vcs_usage = []

    # gpu_devs = get_gpu_devices()
    for _bcmk_dev, _dev_info in gpu_devs.items():
        rd_name = _dev_info['render_name']
        gpu_dev = _bcmk_dev
        gpu_type = _dev_info['device_type']

        avg_gpu_freq, avg_gpu_power, avg_pkg_power, rcs_usages, vcs_usages = eval_gpu_usage(f"gpu_usage_{_bcmk_dev}.txt")
        gpu_freq_list.append(f"{gpu_dev}:{avg_gpu_freq}")
        if 'iGPU' in gpu_type :
            gpu_power_list.append(f"{gpu_dev}:{avg_gpu_power}")
        else:
            dgpu_avg_power = eval_dgpu_power(f"dgpu_power_dump_{_bcmk_dev}.txt")
            gpu_power_list.append(f"{gpu_dev}:{dgpu_avg_power}")
        if avg_pkg_power > 0. :
            out_result['Package Power'] = avg_pkg_power
        gpu_rcs_usage.append([f"{gpu_dev}: {v}" for i, v in enumerate(rcs_usages)])
        gpu_vcs_usage.append([f"{gpu_dev}: {v}" for i, v in enumerate(vcs_usages)])
    
    out_result["GPU_Freq"] = "<br>".join(gpu_freq_list)
    out_result["GPU_Power"] = "<br>".join(gpu_power_list)
    out_result["GPU_RCS_Usage"] = "<br>".join(["<br>".join(item) for item in gpu_rcs_usage])
    out_result["GPU_VCS_Usage"] = "<br>".join(["<br>".join(item) for item in gpu_vcs_usage])

def update_telemetry(out_result={}, gpu_devs={}):
    try:
        out_result["CPU_Usage"] = eval_cpu_usage("cpu_usage.txt")
        out_result["CPU_Freq"] = eval_cpu_freq("cpu_avg_freq.txt")
        out_result["Memory_Usage"] = eval_mem_proc_usage("memory_usage_bcmk")
        if len(gpu_devs) > 0:
            _eval_gpu_telemetry(out_result, gpu_devs)
    except Exception as ex:
        logging.error("Error occurs whening update Telemetry info:")
        print_trace()
    finally:
        for k in ["CPU_Usage", "CPU_Freq", "Memory_Usage", "Package Power", "Package Power", "GPU_Freq", 
                  "GPU_Power", "GPU_RCS_Usage", "GPU_VCS_Usage"]:
            if out_result.get(k, None) is None:
                out_result[k] = 'na'
    return out_result

#############
## Test code
#############

@telemetry_decorator
def mywd(c, text="hello world"):

    rst = {'latency': "1", 'throughput': "99"}
    for _i in range(c):
        print(f"{_i + 1} times to say {text}")
        time.sleep(1)

    return rst

if __name__ == '__main__':
    import json
    print(json.dumps(mywd(9, "hello world"), indent=4))
