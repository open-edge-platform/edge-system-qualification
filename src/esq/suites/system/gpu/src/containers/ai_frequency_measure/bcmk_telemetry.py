# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import csv
import logging
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = Path(f"{CURR_DIR}/output").resolve()
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "freq_execution.log"),
        # logging.StreamHandler()
    ],
)

GPU_TOP_RESULT_DICT = {
    "Freq MHz": ["req", "act"],
    "IRQ RC6": ["/s", "%"],
    "Power W": ["gpu", "pkg"],
    "IMC MiB/s": ["rd", "wr"],
    "RCS": ["%", "se", "wa"],
    "BCS": ["%", "se", "wa"],
    "VCS": ["%", "se", "wa"],
    "VECS": ["%", "se", "wa"],
    "CCS": ["%", "se", "wa"],
    "UNKN/0": ["%", "se", "wa"],
}


def is_intel_xeon():
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        if "Xeon" in cpuinfo:
            return True
        elif "XEON" in cpuinfo:
            return True
        else:
            return False
    except Exception:
        return False


def get_gpu_renders():
    renders = []
    output = subprocess.check_output(["ls", "/dev/dri/"], stderr=subprocess.STDOUT, text=True)
    for line in output.split("\n"):
        if "renderD" in line or "drm:" in line:
            renders.append(line.strip().split()[-1])

    return renders


def get_physical_cores():
    try:
        wlscpu = shutil.which("lscpu")
        if not wlscpu:
            raise FileNotFoundError("lscpu not found")

        # Execute the lscpu command and capture its output
        result = subprocess.run([wlscpu], stdout=subprocess.PIPE, text=True, check=True)
        output = result.stdout

        # Use regular expressions to find the number of cores per socket and the number of sockets
        cores_per_socket = int(re.search(r"Core\(s\) per socket:\s*(\d+)", output).group(1))
        sockets = int(re.search(r"Socket\(s\):\s*(\d+)", output).group(1))

        # Calculate the total number of physical cores
        total_physical_cores = cores_per_socket * sockets

        return total_physical_cores, sockets
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return 1


def cpu_usage_filter(line):
    return line is not None and len(line) > 0 and "procs" not in line and "swpd" not in line


def gpu_usage_filter(line):
    return line is not None and len(line.strip()) > 0


def dgpu_power_filter(line):
    return line is not None and len(line.strip()) > 0


# def cpu_proc_usage_filter(line):
#     return line is not None and "benchmark_app" in line


# This function will run in a separate thread, collecting usage data
def collect_usage(command, output_file, event, filter_func):
    with open(output_file, "w", buffering=1) as f:  # Line buffering
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        cmd_str = " ".join(command)
        logging.info(f"Starting telemetry: {cmd_str} -> {output_file}")

        line_count = 0
        while not event.is_set():
            if "i7z" in command:
                time.sleep(1)
            else:
                line = process.stdout.readline()
                if line:  # Log if we got any line
                    if filter_func(line):
                        f.write(line)
                        f.flush()  # Ensure data is written immediately
                        line_count += 1
                        if line_count == 1:
                            logging.debug(f"First line written to {output_file}: {line.strip()[:100]}")

        logging.info(f"Stopped telemetry: {cmd_str}, wrote {line_count} lines")
        if "i7z" in command:
            _filter_lines = []
            with open("i7z_output_raw.txt", "r") as f:
                _filter_lines = f.readlines()

            cpu_phy_core_number, _ = get_physical_cores()
            filtered_lines = [l for l in _filter_lines if len(l.strip().split()) == (cpu_phy_core_number + 1)]

            with open("i7z_output.txt", "w") as f:
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


def collect_usage_after(command, output_file, event, filter_func):
    bcmk_pids = fetch_bcmk_pids()
    thread_mem_list = []
    for i, bcmk_pid in enumerate(bcmk_pids):
        command = [str(bcmk_pid) if item == "$bcmk_pid" else item for item in command]
        collect_usage(command, f"{output_file}_s{i + 1}.txt", event, filter_func)
    else:
        if bcmk_pids is None or len(bcmk_pids) == 0:
            logging.error("Cannot found benchmark_app process and collect its memory usage.")


def get_bcmk_pids():
    try:
        # pid = subprocess.check_output(["pgrep", "-f", "benchmark_app"]).decode('utf-8').strip()
        result = subprocess.run(["pgrep", "-f", "benchmark_app"], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
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


def check_file(file_path):
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def eval_cpu_freq(c_freq_file):
    if not check_file(c_freq_file):
        return -1.0

    cpu_freq_df = pd.read_csv(c_freq_file, header=None)
    cpu_freq_df["max_freq"] = cpu_freq_df.iloc[:, 1:].max(axis=1)
    cpu_freq_df["max_freq"] = cpu_freq_df["max_freq"].astype(float) / 1000

    # cpu_freq_df.iloc[:, 0] = pd.to_datetime(cpu_freq_df.iloc[:, 0], format='%Y-%m-%d %H:%M:%S')

    cpu_freq_df.rename(columns={cpu_freq_df.columns[0]: "date"}, inplace=True)
    cpu_freq_df["date"] = pd.to_datetime(cpu_freq_df["date"], format="%Y-%m-%d %H:%M:%S")
    return cpu_freq_df


def eval_cpu_usage(cpu_file):
    if not check_file(cpu_file):
        return None

    cpu_usage_df = pd.read_csv(cpu_file, sep=r"\s+", header=None)

    tuned_cu_df = cpu_usage_df.iloc[:, [-1, -2, -5, -7, -8]]
    new_column_names = ["date1", "date2", "us", "sy", "wa"]
    tuned_cu_df.columns = new_column_names

    tuned_cu_df["date"] = pd.to_datetime(tuned_cu_df["date2"].astype(str) + " " + tuned_cu_df["date1"].astype(str))
    # tuned_cu_df['date'] = pd.to_datetime(tuned_cu_df['date'].astype(str), format='%Y-%m-%d %H:%M:%S')
    tuned_cu_df.drop(["date1", "date2"], axis=1, inplace=True)

    tuned_cu_df["sum"] = tuned_cu_df.iloc[:, :3].sum(axis=1)

    _, sock_num = get_physical_cores()
    if sock_num > 1:
        tuned_cu_df["sum"] = tuned_cu_df["sum"] * 2
        tuned_cu_df["sum"] = tuned_cu_df["sum"].where(tuned_cu_df["sum"] <= 100, 99.9)
    return tuned_cu_df


def eval_dgpu_power(dgpu_file):
    if not check_file(dgpu_file):
        return None

    lines = []
    with open(dgpu_file) as f:
        lines = f.readlines()
    if len(lines) < 5:
        return None

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    dgpu_power_df = pd.read_csv(dgpu_file, sep=",")
    dgpu_power_df.columns = dgpu_power_df.columns.str.strip()  # some space char in column
    dgpu_power_df["Timestamp"] = dgpu_power_df["Timestamp"].str.slice(stop=8)
    dgpu_power_df["date"] = pd.to_datetime(current_date + " " + dgpu_power_df["Timestamp"])

    # Convert 'GPU Power (W)' column to numeric, replacing N/A with NaN
    if "GPU Power (W)" in dgpu_power_df.columns:
        dgpu_power_df["GPU Power (W)"] = pd.to_numeric(dgpu_power_df["GPU Power (W)"], errors='coerce')

        # Check if all values are NaN (all N/A values)
        if dgpu_power_df["GPU Power (W)"].isna().all():
            logging.warning(f"All dGPU power values are N/A in {dgpu_file} - power monitoring may not be supported on this platform")
            return None

    return dgpu_power_df


def normalize_engines(*engines: np.ndarray) -> np.ndarray:
    valid_engines = [e for e in engines if e.size > 0]

    if not valid_engines:
        raise ValueError("No non-empty input arrays provided.")

    shapes = [e.shape for e in valid_engines]
    if len(set(shapes)) != 1:
        raise ValueError(f"Shape mismatch b/w the input arrays: {shapes}")

    stacked = np.column_stack([e.ravel() for e in valid_engines])

    totals = stacked.sum(axis=1)

    if totals.max() > 0:
        overall_percent = (totals / totals.max()) * 100
    else:
        overall_percent = np.zeros_like(totals, dtype=float)

    overall_percent = np.rint(overall_percent).astype(int)

    return overall_percent


def eval_gpu_usage(gpu_file):
    if not check_file(gpu_file):
        return None
    lines = []
    with open(gpu_file) as f:
        lines = f.readlines()
    if len(lines) < 8:
        return None

    catelogy_line = lines[0]
    title_line = lines[1]
    data_lines = [l.strip() for l in lines if not l.strip().startswith("Freq") and not l.strip().startswith("req")]
    data_lines_2d = np.array(
        [[float(item) if item.replace(".", "", 1).isdigit() else np.nan for item in l.split()] for l in data_lines]
    )
    data_lines_2d = data_lines_2d[4:-2]

    three_spaces = "   "
    two_spaces = "  "
    while three_spaces in catelogy_line:
        catelogy_line = catelogy_line.replace(three_spaces, two_spaces)

    result_category = catelogy_line.strip().split(two_spaces)
    subtitle_list = title_line.strip().split()

    rcs_0_count = catelogy_line.count("RCS")

    gpu_top_dict = copy.deepcopy(GPU_TOP_RESULT_DICT)

    if "Power W" not in result_category and "pkg" in subtitle_list:
        gpu_top_dict["IRQ RC6"].append("pkg")

    gpu_result_idx_map = {}
    st_idx = 0
    for _category in result_category:
        gpu_result_idx_map[_category] = {}
        sub_titles = gpu_top_dict.get(_category, [])
        for st in sub_titles:
            gpu_result_idx_map[_category][st] = st_idx
            st_idx += 1

    gpu_freq_idx = gpu_result_idx_map["Freq MHz"]["act"]
    gpu_freq_list = data_lines_2d[:, gpu_freq_idx]

    gpu_util_idx = 0
    gpu_rcs_util = np.array([], dtype=float)
    gpu_ccs_util = np.array([], dtype=float)
    gpu_bcs_util = np.array([], dtype=float)

    try:
        if "RCS" in result_category:
            gpu_util_idx += 1
            gpu_rcs_idx = gpu_result_idx_map["RCS"]["%"]
            gpu_rcs_util = data_lines_2d[:, gpu_rcs_idx].astype(float)
        if "CCS" in result_category:  # and "pkg" not in subtitle_list:
            gpu_util_idx += 1
            gpu_ccs_idx = gpu_result_idx_map["CCS"]["%"]
            gpu_ccs_util = data_lines_2d[:, gpu_ccs_idx].astype(float)
        if "BCS" in result_category:
            gpu_util_idx += 1
            gpu_bcs_idx = gpu_result_idx_map["BCS"]["%"]
            gpu_bcs_util = data_lines_2d[:, gpu_bcs_idx].astype(float)

        logging.debug(
            f"GPU engine arrays - RCS: {len(gpu_rcs_util)}, CCS: {len(gpu_ccs_util)}, BCS: {len(gpu_bcs_util)}"
        )
        gpu_util = normalize_engines(gpu_rcs_util, gpu_ccs_util, gpu_bcs_util)
        logging.debug(f"GPU normalized utilization: {len(gpu_util)} samples, mean={gpu_util.mean():.2f}")
    except Exception as e:
        logging.error(f"Failed to calculate GPU utilization for {gpu_file}: {type(e).__name__}: {str(e)}")
        logging.error(f"  RCS util shape: {gpu_rcs_util.shape}, CCS: {gpu_ccs_util.shape}, BCS: {gpu_bcs_util.shape}")
        # Return None to indicate failure - don't try to create partial dataframe
        return None

    try:
        pkg_power_idx = 0
        if "Power W" not in result_category and "pkg" in subtitle_list:
            pkg_power_idx = gpu_result_idx_map["IRQ RC6"]["pkg"]
        else:
            pkg_power_idx = gpu_result_idx_map["Power W"]["pkg"]
        pkg_power_list = data_lines_2d[:, pkg_power_idx]
    except:
        pkg_power_idx = -1

    # Check if we have valid utilization data before creating dataframe
    if gpu_util_idx <= 0:
        logging.warning(f"No valid GPU utilization data for {gpu_file} - cannot create complete dataframe")
        return None

    if pkg_power_idx >= 0:
        # gpu_rst = pd.DataFrame(data_lines_2d[:, [gpu_freq_idx, gpu_util_idx, pkg_power_idx]], columns=['gpu_freq', 'gpu_util', 'pkg_power'])
        gpu_rst = pd.DataFrame(
            data={
                "gpu_freq": data_lines_2d[:, gpu_freq_idx].astype(float),
                "gpu_util": gpu_util,
                "pkg_power": data_lines_2d[:, pkg_power_idx].astype(float),
            }
        )
    else:
        # gpu_rst = pd.DataFrame(data_lines_2d[:, [gpu_freq_idx, gpu_util_idx]], columns=['gpu_freq', 'gpu_util'])
        gpu_rst = pd.DataFrame(data={"gpu_freq": data_lines_2d[:, gpu_freq_idx].astype(float), "gpu_util": gpu_util})

    gpu_rst["gpu_freq"] = gpu_rst["gpu_freq"] / 1000

    return gpu_rst


def _start_cpu_usage_thread(event, out_file):
    # Ensure path is string for subprocess
    out_file_str = str(out_file) if not isinstance(out_file, str) else out_file
    thread_cpu = threading.Thread(
        target=collect_usage, args=(["vmstat", "-w", "-t", "1"], out_file_str, event, cpu_usage_filter)
    )
    thread_cpu.start()
    return thread_cpu


def _start_cpu_freq_thread(event, out_file):
    def write_cpu_freq_to_file():
        # Ensure path is string for file operations
        out_file_str = str(out_file) if not isinstance(out_file, str) else out_file

        open(out_file_str, "w").close()
        with open(out_file_str, "a", newline="") as csvfile:
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


def _start_gpu_usage_threads(event, gpu_render_devs):
    gpu_tds = []
    for rd in gpu_render_devs:
        rd_name = rd.split("/")[-1]
        output_file = OUTPUT_DIR / f"gpu_usage_{rd_name}.txt"
        thread_gpu = threading.Thread(
            target=collect_usage,
            args=(["intel_gpu_top", "-d", f"drm:/dev/dri/{rd}", "-l"], str(output_file), event, gpu_usage_filter),
        )
        gpu_tds.append(thread_gpu)
    for g_td in gpu_tds:
        g_td.start()

    return gpu_tds


def _start_dgpu_power_threads(event, gpu_render_devs):
    for rd in gpu_render_devs:
        rd_name = rd.split("/")[-1]
        if "renderD129" in rd_name:
            # xpu-smi dump: -d 0 (device index 0), -m 1 (metric dump interval 1 sec)
            # Device index 0 for dGPU in xpu-smi (different from renderD numbering)
            output_file = OUTPUT_DIR / "dgpu_power_dump.txt"
            thread_dgpu = threading.Thread(
                target=collect_usage,
                args=(["xpu-smi", "dump", "-d", "0", "-m", "1"], str(output_file), event, dgpu_power_filter),
            )
            thread_dgpu.start()
            return thread_dgpu

    return None


def start_telemetry_thread(event):
    cu_thread = _start_cpu_usage_thread(event, OUTPUT_DIR / "cpu_usage.txt")
    # cu_bcmk_thread = _start_cpu_usage_4process_thread(event, "cpu_usage_bcmk")
    cf_thread = _start_cpu_freq_thread(event, OUTPUT_DIR / "cpu_avg_freq.txt")
    gpu_thread_list = []
    gpu_render_devices = get_gpu_renders()

    if len(gpu_render_devices) > 0:
        gpu_thread_list = _start_gpu_usage_threads(event, gpu_render_devices)

        dgpu_power_thread = _start_dgpu_power_threads(event, gpu_render_devices)
        if dgpu_power_thread is not None:
            gpu_thread_list.append(dgpu_power_thread)

    return [cu_thread, cf_thread] + gpu_thread_list


def telemetry_decorator(func):
    def wrapper(*args, **kwargs):
        event = threading.Event()
        # gpu_render_devices = get_gpu_renders()
        thread_list = start_telemetry_thread(event)
        try:
            result = func(*args, **kwargs)
        finally:
            event.set()

        for td in thread_list:
            td.join()

        return update_telemetry(result)

    return wrapper


def _select_best_dgpu(dgpu_candidates):
    """
    Select the best performing dGPU from multiple candidates.

    Selection criteria (in priority order):
    1. Highest average frequency (peak performance)
    2. Lowest frequency standard deviation (stability/consistency)
    3. Highest utilization (most engaged)

    Args:
        dgpu_candidates: List of dicts with keys: device, metrics, freq_avg, freq_stddev, utilization_avg

    Returns:
        dict: Best dGPU candidate with selection metadata
    """
    if not dgpu_candidates:
        return None

    if len(dgpu_candidates) == 1:
        best = dgpu_candidates[0]
        logging.info(f"Single dGPU detected: {best['device']}")
        return best

    # Sort by: 1) freq_avg DESC, 2) freq_stddev ASC, 3) utilization_avg DESC
    sorted_candidates = sorted(
        dgpu_candidates,
        key=lambda x: (
            -x['freq_avg'],           # Higher frequency is better (negate for DESC)
            x['freq_stddev'],         # Lower stddev is better (ASC)
            -x['utilization_avg']     # Higher utilization is better (negate for DESC)
        )
    )

    best = sorted_candidates[0]

    # Log selection details
    logging.info(f"Multiple dGPUs detected: {len(dgpu_candidates)} devices")
    for i, candidate in enumerate(sorted_candidates):
        rank_label = "SELECTED" if i == 0 else f"Rank {i+1}"
        logging.info(
            f"  [{rank_label}] {candidate['device']}: "
            f"freq_avg={candidate['freq_avg']:.2f} GHz, "
            f"freq_stddev={candidate['freq_stddev']:.4f}, "
            f"util_avg={candidate['utilization_avg']:.2f}%"
        )

    return best


def _eval_gpu_telemetry(out_result):
    gpu_usage = {}
    dgpu_candidates = []  # Collect all dGPU metrics for comparison

    rds = get_gpu_renders()

    if len(rds) == 0:
        return

    for rd in rds:
        rd_name = rd.split("/")[-1]
        if "128" in rd_name:
            gpu_dev = "igpu"
        else:
            gpu_dev = "dgpu"

        gpu_file = OUTPUT_DIR / f"gpu_usage_{rd_name}.txt"
        gpu_rst = eval_gpu_usage(str(gpu_file))

        if gpu_rst is not None:
            if gpu_dev == "igpu":
                # iGPU: use directly (typically only one)
                gpu_usage[gpu_dev] = gpu_rst
            else:
                # dGPU: collect for comparison
                freq_avg = gpu_rst["gpu_freq"].mean()
                freq_stddev = gpu_rst["gpu_freq"].std()
                utilization_avg = gpu_rst["gpu_util"].mean()

                # Validate metrics before adding candidate
                # Filter out invalid data: freq_avg <= 0.01 GHz (10 MHz) indicates no valid data
                if freq_avg > 0.01:
                    dgpu_candidates.append({
                        "device": rd_name,
                        "metrics": gpu_rst,
                        "freq_avg": freq_avg,
                        "freq_stddev": freq_stddev,
                        "utilization_avg": utilization_avg
                    })
                    logging.debug(
                        f"Collected dGPU candidate {rd_name}: "
                        f"freq_avg={freq_avg:.2f}, freq_stddev={freq_stddev:.4f}, util={utilization_avg:.2f}%"
                    )
                else:
                    logging.warning(
                        f"Skipping dGPU {rd_name} - invalid metrics (freq_avg={freq_avg:.4f} GHz, "
                        f"util={utilization_avg:.2f}%). GPU telemetry may have failed."
                    )
        else:
            logging.warning(f"Skipping {gpu_dev} data - eval_gpu_usage returned None for {gpu_file}")

    # Select best dGPU from candidates
    if dgpu_candidates:
        best_dgpu = _select_best_dgpu(dgpu_candidates)
        if best_dgpu:
            gpu_usage["dgpu"] = best_dgpu["metrics"]
            # Add metadata for visibility
            gpu_usage["dgpu_device_id"] = best_dgpu["device"]
            gpu_usage["dgpu_count"] = len(dgpu_candidates)
            logging.info(
                f"Selected dGPU: {best_dgpu['device']} (out of {len(dgpu_candidates)} available)"
            )
    else:
        # No valid dGPU candidates found (all had invalid metrics)
        # This can happen when intel_gpu_top returns all zeros
        logging.warning(
            "No valid dGPU candidates with valid metrics. "
            "All dGPU telemetry may have failed (intel_gpu_top returned zeros)."
        )

    out_result["GPU_Usage"] = gpu_usage

    dgpu_power_file = OUTPUT_DIR / "dgpu_power_dump.txt"
    dgpu_power_rst = eval_dgpu_power(str(dgpu_power_file))
    if dgpu_power_rst is not None:
        out_result["dGPU_Power"] = dgpu_power_rst


def update_telemetry(out_result={}):
    if out_result is None:
        out_result = {}

    try:
        cpu_usage_file = OUTPUT_DIR / "cpu_usage.txt"
        out_result["CPU_Usage"] = eval_cpu_usage(str(cpu_usage_file))
        cpu_freq_file = OUTPUT_DIR / "cpu_avg_freq.txt"
        out_result["CPU_Freq"] = eval_cpu_freq(str(cpu_freq_file))
        _eval_gpu_telemetry(out_result)
    except Exception:
        logging.error("Error occurs whening update Telemetry info:")
        print_trace()

    return out_result


@telemetry_decorator
def mywd(c, text="hello world"):
    rst = {"latency": "1", "throughput": "99"}
    for _i in range(c):
        print(f"{_i + 1} times to say {text}")
        time.sleep(1)

    return rst


if __name__ == "__main__":
    rst = mywd(9, "hello world")
    print(rst["CPU_Usage"].shape)
    print(rst["CPU_Freq"])
    print(rst["GPU_Usage"])
    print(rst.get("dGPU_Power", None))
