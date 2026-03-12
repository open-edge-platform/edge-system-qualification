# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import csv
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404 # For benchmark telemetry (lscpu, intel_gpu_top)
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pmu_gpu_collector import collect_gpu_telemetry_pmu, generate_gpu_telemetry_output
    PMU_TELEMETRY_AVAILABLE = True
except ImportError:
    PMU_TELEMETRY_AVAILABLE = False

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


PTL_IGPU_DEVICE_IDS = {
    "B080",
    "B081",
    "B082",
    "B083",
    "B084",
    "B085",
    "B086",
    "B087",
    "B08F",
    "B090",
    "B0A0",
    "B0B0",
}


def _device_type_from_pci_device_id(pci_device_id: str) -> str:
    if not pci_device_id:
        return "unknown"

    stripped = pci_device_id[2:] if pci_device_id.startswith("0x") else pci_device_id
    if not stripped:
        return "unknown"

    normalized = stripped.upper()

    # Panther Lake uses explicit PCI ID allow-list.
    if normalized in PTL_IGPU_DEVICE_IDS:
        return "igpu"

    first_char = normalized[0]
    if first_char in ["4", "9", "A"]:
        return "igpu"
    if first_char == "5":
        return "dgpu"
    if first_char == "7":
        return "igpu"

    return "unknown"


def _discover_render_device_types() -> dict[str, str]:
    render_device_types = {}

    try:
        command_output = subprocess.check_output(["intel_gpu_top", "-L"], text=True)
        card_pattern = re.compile(r"^(card\d+)\s+.*?pci:vendor=\w+,device=(\w+),card=\d+\n└─(renderD\d+)", re.MULTILINE)
        matches = card_pattern.findall(command_output)

        for _card_id, pci_device_id, render_name in matches:
            render_device_types[render_name] = _device_type_from_pci_device_id(pci_device_id)
    except Exception as e:
        logging.warning(f"Failed to discover render device mapping from intel_gpu_top -L: {e}")

    return render_device_types


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


def _extract_render_number(render_name: str) -> int:
    match = re.search(r"renderD(\d+)", str(render_name))
    if not match:
        return -1
    return int(match.group(1))


def _parse_requested_gpu_indices(device_spec: str) -> list[int]:
    if not device_spec:
        return []

    normalized = str(device_spec).strip()
    if not normalized:
        return []

    if normalized.startswith("MULTI:"):
        requested_tokens = [token.strip() for token in normalized[6:].split(",") if token.strip()]
    else:
        requested_tokens = [normalized]

    gpu_indices = []
    for token in requested_tokens:
        upper_token = token.upper()
        if upper_token == "GPU":
            gpu_indices.append(0)
            continue

        match = re.match(r"^GPU\.(\d+)$", upper_token)
        if match:
            gpu_indices.append(int(match.group(1)))

    unique_indices = []
    for idx in gpu_indices:
        if idx not in unique_indices:
            unique_indices.append(idx)

    return unique_indices


def _resolve_telemetry_gpu_renders(device_spec: str | None) -> list[str]:
    detected_renders = get_gpu_renders()
    if not detected_renders:
        return []

    sorted_renders = sorted(detected_renders, key=_extract_render_number)
    requested_indices = _parse_requested_gpu_indices(device_spec or "")
    if not requested_indices:
        logging.info(
            "No explicit GPU device index requested for telemetry; using all detected render nodes: "
            f"{sorted_renders}"
        )
        return sorted_renders

    selected_renders = [sorted_renders[idx] for idx in requested_indices if 0 <= idx < len(sorted_renders)]
    if selected_renders:
        logging.info(
            "Resolved telemetry target renders from workload device "
            f"'{device_spec}': {selected_renders} (detected={sorted_renders})"
        )
        return selected_renders

    logging.warning(
        "Unable to map requested GPU indices from workload device "
        f"'{device_spec}' to detected render nodes {sorted_renders}; using all detected render nodes"
    )
    return sorted_renders


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


def check_intel_gpu_top_available():
    """Check if intel_gpu_top is available and working."""
    try:
        result = subprocess.run(
            ["intel_gpu_top", "-h"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# def cpu_proc_usage_filter(line):
#     return line is not None and "benchmark_app" in line


# This function will run in a separate thread, collecting usage data
def collect_usage(command, output_file, event, filter_func):
    with open(output_file, "w", buffering=1) as f:  # Line buffering
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,  # Detach from TTY (aligned with OV benchmark telemetry)
            text=True,
        )

        cmd_str = " ".join(command)
        logging.info(f"Starting telemetry: {cmd_str} -> {output_file}")

        line_count = 0
        retried_alternate_device = False
        while not event.is_set():
            if "i7z" in command:
                time.sleep(1)
            else:
                if process.poll() is not None:
                    break
                line = process.stdout.readline()
                if line:  # Log if we got any line
                    if filter_func(line):
                        f.write(line)
                        f.flush()  # Ensure data is written immediately
                        line_count += 1
                        if line_count == 1:
                            logging.debug(f"First line written to {output_file}: {line.strip()[:100]}")
                elif process.poll() is not None:
                    stderr_text = ""
                    try:
                        stderr_text = (process.stderr.read() or "").strip()
                    except Exception:
                        stderr_text = ""

                    def _build_alternate_intel_gpu_top_command(cmd):
                        if len(cmd) < 4:
                            return None

                        if len(cmd) < 4 or cmd[0] != "intel_gpu_top":
                            return None

                        try:
                            device_arg_idx = cmd.index("-d") + 1
                        except ValueError:
                            return None

                        if device_arg_idx >= len(cmd):
                            return None

                        device_arg = str(cmd[device_arg_idx])
                        match = re.match(r"drm:/dev/dri/renderD(\d+)$", device_arg)
                        if not match:
                            return None

                        render_num = int(match.group(1))
                        card_num = render_num - 128
                        if card_num < 0:
                            return None

                        alt_cmd = list(cmd)
                        alt_cmd[device_arg_idx] = f"drm:/dev/dri/card{card_num}"
                        return alt_cmd

                    if not retried_alternate_device and process.returncode != 0:
                        alternate_command = _build_alternate_intel_gpu_top_command(command)
                        if alternate_command is not None:
                            logging.warning(
                                "Telemetry intel_gpu_top failed, retrying with alternate device node: "
                                f"{' '.join(command)} -> {' '.join(alternate_command)}; stderr: {stderr_text}"
                            )
                            process = subprocess.Popen(
                                alternate_command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                            )
                            command = alternate_command
                            cmd_str = " ".join(command)
                            retried_alternate_device = True
                            continue

                    if process.returncode != 0:
                        logging.warning(
                            f"Telemetry command exited early with code {process.returncode}: {cmd_str}; stderr: {stderr_text}"
                        )
                    else:
                        logging.info(
                            f"Telemetry command finished before stop signal: {cmd_str}; collected {line_count} lines"
                        )
                    break

        logging.info(f"Stopped telemetry: {cmd_str}, wrote {line_count} lines")
        if line_count == 0:
            try:
                stderr_output = process.stderr.read().strip() if process.stderr else ""
                if stderr_output:
                    logging.warning(f"Telemetry stderr for '{cmd_str}': {stderr_output}")

                    # Check for intel_gpu_top assertion failure (debugfs access issue)
                    if "intel_gpu_top" in cmd_str and "Assertion" in stderr_output:
                        logging.warning(f"intel_gpu_top failed with assertion error (likely debugfs permission issue)")
                        logging.info(f"Attempting fallback: PMU-based telemetry collection for {output_file}")

                        if PMU_TELEMETRY_AVAILABLE:
                            try:
                                # Extract device name from output filename
                                device_name = "renderD128" if "renderD128" in str(output_file) else "renderD129"

                                # Generate intel_gpu_top-compatible output using PMU/sysfs
                                # Use same duration as test (typically 10+ seconds for sampling)
                                pmu_output = generate_gpu_telemetry_output(device_name, duration_sec=10)

                                # Write compatible format to output file
                                with open(output_file, "w") as f:
                                    f.write(pmu_output)

                                logging.info(f"Fallback telemetry written to {output_file} using PMU/sysfs")
                                logging.info(f"Output has {len(pmu_output.splitlines())} lines, compatible with intel_gpu_top format")
                                return
                            except Exception as fallback_err:
                                logging.error(f"PMU fallback also failed: {fallback_err}")
            except Exception as stderr_err:
                logging.debug(f"Failed to read telemetry stderr for '{cmd_str}': {stderr_err}")

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
            if process.poll() is None:
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

    def _norm_key(col_name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(col_name).lower())

    norm_map = {_norm_key(column_name): column_name for column_name in dgpu_power_df.columns}

    timestamp_col = next((col for key, col in norm_map.items() if "timestamp" in key), None)
    if timestamp_col is None:
        return None

    power_col = next(
        (
            col
            for key, col in norm_map.items()
            if "power" in key and ("gpu" in key or "pkg" in key or "package" in key)
        ),
        None,
    )
    util_col = next((col for key, col in norm_map.items() if "util" in key and "gpu" in key), None)
    freq_col = next((col for key, col in norm_map.items() if "freq" in key and "gpu" in key), None)

    dgpu_power_df["Timestamp"] = dgpu_power_df[timestamp_col].astype(str).str.slice(stop=8)
    dgpu_power_df["date"] = pd.to_datetime(current_date + " " + dgpu_power_df["Timestamp"])

    if power_col is not None:
        dgpu_power_df["GPU Power (W)"] = pd.to_numeric(dgpu_power_df[power_col], errors="coerce")

    if util_col is not None:
        dgpu_power_df["GPU Utilization (%)"] = pd.to_numeric(dgpu_power_df[util_col], errors="coerce")

    if freq_col is not None:
        dgpu_power_df["GPU Frequency (MHz)"] = pd.to_numeric(dgpu_power_df[freq_col], errors="coerce")

    has_valid_power = "GPU Power (W)" in dgpu_power_df.columns and not dgpu_power_df["GPU Power (W)"].isna().all()
    has_valid_freq = (
        "GPU Frequency (MHz)" in dgpu_power_df.columns and not dgpu_power_df["GPU Frequency (MHz)"].isna().all()
    )

    if not has_valid_power and not has_valid_freq:
        logging.warning(
            f"No valid dGPU power/frequency values in {dgpu_file} - telemetry may not be supported on this platform"
        )
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
    if len(lines) < 3:
        return None

    catelogy_line = lines[0]
    title_line = lines[1]
    data_lines = [
        l.strip()
        for l in lines
        if l.strip() and not l.strip().startswith("Freq") and not l.strip().startswith("req")
    ]
    data_lines_2d = np.array(
        [[float(item) if item.replace(".", "", 1).isdigit() else np.nan for item in l.split()] for l in data_lines]
    )

    if data_lines_2d.ndim != 2 or data_lines_2d.shape[0] == 0:
        return None

    # For longer captures, trim warm-up and tail noise. For short runs, keep all samples.
    if data_lines_2d.shape[0] > 8:
        trimmed = data_lines_2d[4:-2]
        if trimmed.shape[0] > 0:
            data_lines_2d = trimmed

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

    act_freq_idx = gpu_result_idx_map["Freq MHz"]["act"]
    req_freq_idx = gpu_result_idx_map["Freq MHz"].get("req", act_freq_idx)
    act_freq_list = data_lines_2d[:, act_freq_idx].astype(float)
    req_freq_list = data_lines_2d[:, req_freq_idx].astype(float)
    gpu_freq_list = np.where(req_freq_list > act_freq_list, req_freq_list, act_freq_list)
    gpu_freq_list = np.where(gpu_freq_list > 0, gpu_freq_list, np.nan)

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
        # Keep frequency metrics even when engine utilization parsing fails.
        gpu_util = np.zeros(len(gpu_freq_list), dtype=float)

    try:
        pkg_power_idx = 0
        if "Power W" not in result_category and "pkg" in subtitle_list:
            pkg_power_idx = gpu_result_idx_map["IRQ RC6"]["pkg"]
        else:
            pkg_power_idx = gpu_result_idx_map["Power W"]["pkg"]
        pkg_power_list = data_lines_2d[:, pkg_power_idx]
    except:
        pkg_power_idx = -1

    # Ensure utilization vector aligns with frequency samples.
    if gpu_util_idx <= 0 or len(gpu_util) != len(gpu_freq_list):
        gpu_util = np.zeros(len(gpu_freq_list), dtype=float)

    if pkg_power_idx >= 0:
        # gpu_rst = pd.DataFrame(data_lines_2d[:, [gpu_freq_idx, gpu_util_idx, pkg_power_idx]], columns=['gpu_freq', 'gpu_util', 'pkg_power'])
        gpu_rst = pd.DataFrame(
            data={
                "gpu_freq": gpu_freq_list.astype(float),
                "gpu_util": gpu_util,
                "pkg_power": data_lines_2d[:, pkg_power_idx].astype(float),
            }
        )
    else:
        # gpu_rst = pd.DataFrame(data_lines_2d[:, [gpu_freq_idx, gpu_util_idx]], columns=['gpu_freq', 'gpu_util'])
        gpu_rst = pd.DataFrame(data={"gpu_freq": gpu_freq_list.astype(float), "gpu_util": gpu_util})

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

        gpu_top_cmd = ["intel_gpu_top", "-d", f"drm:/dev/dri/{rd}", "-l"]

        thread_gpu = threading.Thread(
            target=collect_usage,
            args=(gpu_top_cmd, str(output_file), event, gpu_usage_filter),
        )
        gpu_tds.append(thread_gpu)
    for g_td in gpu_tds:
        g_td.start()

    return gpu_tds


def _discover_xpu_device_ids(device_type_filter: str | None = None) -> list[str]:
    try:
        discovery_output = subprocess.check_output(["xpu-smi", "discovery", "-j"], text=True)
        discovery = json.loads(discovery_output)
        device_list = discovery.get("device_list") or []

        device_ids = []
        for device in device_list:
            device_type = _device_type_from_pci_device_id(str(device.get("pci_device_id", "")))
            if device_type_filter and device_type != device_type_filter:
                continue

            device_id = str(device.get("device_id", "")).strip()
            if device_id.isdigit():
                device_ids.append(device_id)

        return device_ids
    except Exception as e:
        logging.warning(f"Failed to discover xpu-smi device ids: {e}")
        return []


def _start_igpu_power_threads(event, gpu_render_devs):
    render_type_map = _discover_render_device_types()
    has_igpu_target = any(render_type_map.get(rd, "unknown") == "igpu" for rd in gpu_render_devs)

    if not has_igpu_target:
        has_igpu_target = any(str(rd).endswith("renderD128") for rd in gpu_render_devs)

    if not has_igpu_target:
        return None

    igpu_device_ids = _discover_xpu_device_ids("igpu")
    if not igpu_device_ids:
        return None

    output_file = OUTPUT_DIR / "igpu_power_dump.txt"
    thread_igpu = threading.Thread(
        target=collect_usage,
        args=(
            ["xpu-smi", "dump", "-d", igpu_device_ids[0], "-m", "0,1,2"],
            str(output_file),
            event,
            dgpu_power_filter,
        ),
    )
    thread_igpu.start()
    logging.info(f"Started xpu-smi fallback collection for iGPU device index {igpu_device_ids[0]}")
    return thread_igpu


def _start_dgpu_power_threads(event, gpu_render_devs):
    dgpu_device_ids = []

    dgpu_device_ids = _discover_xpu_device_ids("dgpu")

    if not dgpu_device_ids:
        # Fallback to legacy xpu-smi index 0 when dynamic mapping is unavailable.
        # Keep a permissive fallback because intel_gpu_top -L parsing can be inconsistent across drivers.
        non_igpu_renders = [rd for rd in gpu_render_devs if "renderD128" not in rd]
        render_type_map = _discover_render_device_types()
        has_dgpu_render = any(render_type_map.get(rd, "unknown") == "dgpu" for rd in gpu_render_devs)
        if has_dgpu_render or non_igpu_renders:
            dgpu_device_ids = ["0"]
            logging.warning(
                "Using fallback xpu-smi device index 0 for dGPU power collection; dynamic mapping was unavailable"
            )

    if dgpu_device_ids:
        output_file = OUTPUT_DIR / "dgpu_power_dump.txt"
        thread_dgpu = threading.Thread(
            target=collect_usage,
            args=(
                ["xpu-smi", "dump", "-d", dgpu_device_ids[0], "-m", "0,1,2"],
                str(output_file),
                event,
                dgpu_power_filter,
            ),
        )
        thread_dgpu.start()
        logging.info(f"Started xpu-smi power collection for dGPU device index {dgpu_device_ids[0]}")
        return thread_dgpu

    return None


def start_telemetry_thread(event):
    cu_thread = _start_cpu_usage_thread(event, OUTPUT_DIR / "cpu_usage.txt")
    # cu_bcmk_thread = _start_cpu_usage_4process_thread(event, "cpu_usage_bcmk")
    cf_thread = _start_cpu_freq_thread(event, OUTPUT_DIR / "cpu_avg_freq.txt")
    gpu_thread_list = []
    requested_device = os.environ.get("AI_FREQ_TELEMETRY_TARGET_DEVICE", "")
    gpu_render_devices = _resolve_telemetry_gpu_renders(requested_device)

    if len(gpu_render_devices) > 0:
        gpu_thread_list = _start_gpu_usage_threads(event, gpu_render_devices)

        igpu_power_thread = _start_igpu_power_threads(event, gpu_render_devices)
        if igpu_power_thread is not None:
            gpu_thread_list.append(igpu_power_thread)

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
            -x["freq_avg"],  # Higher frequency is better (negate for DESC)
            x["freq_stddev"],  # Lower stddev is better (ASC)
            -x["utilization_avg"],  # Higher utilization is better (negate for DESC)
        ),
    )

    best = sorted_candidates[0]

    # Log selection details
    logging.info(f"Multiple dGPUs detected: {len(dgpu_candidates)} devices")
    for i, candidate in enumerate(sorted_candidates):
        rank_label = "SELECTED" if i == 0 else f"Rank {i + 1}"
        logging.info(
            f"  [{rank_label}] {candidate['device']}: "
            f"freq_avg={candidate['freq_avg']:.2f} GHz, "
            f"freq_stddev={candidate['freq_stddev']:.4f}, "
            f"util_avg={candidate['utilization_avg']:.2f}%"
        )

    return best


def _eval_gpu_telemetry(out_result):
    def _expand_series_to_length(series, target_len, default_value=0.0):
        if target_len <= 0:
            return np.array([], dtype=float)

        if series is None:
            return np.full(target_len, default_value, dtype=float)

        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if numeric_series.empty:
            return np.full(target_len, default_value, dtype=float)

        values = numeric_series.to_numpy(dtype=float)
        if len(values) >= target_len:
            return values[-target_len:]

        return np.resize(values, target_len)

    def _build_igpu_fallback_from_xpu():
        igpu_power_file = OUTPUT_DIR / "igpu_power_dump.txt"
        igpu_power_rst = eval_dgpu_power(str(igpu_power_file))
        if igpu_power_rst is None or "GPU Frequency (MHz)" not in igpu_power_rst.columns:
            return None

        valid_rows = (~igpu_power_rst["GPU Frequency (MHz)"].isna()) & (igpu_power_rst["GPU Frequency (MHz)"] > 0)
        if not valid_rows.any():
            return None

        target_len = int(valid_rows.sum())
        fallback_payload = {
            "date": igpu_power_rst.loc[valid_rows, "date"],
            "gpu_freq": igpu_power_rst.loc[valid_rows, "GPU Frequency (MHz)"] / 1000.0,
            "gpu_util": _expand_series_to_length(
                igpu_power_rst.loc[valid_rows, "GPU Utilization (%)"]
                if "GPU Utilization (%)" in igpu_power_rst.columns
                else None,
                target_len,
                default_value=np.nan,
            ),
        }

        power_series = igpu_power_rst["GPU Power (W)"] if "GPU Power (W)" in igpu_power_rst.columns else None
        if power_series is not None:
            power_values = pd.to_numeric(power_series.loc[valid_rows], errors="coerce")
            if not power_values.isna().all():
                fallback_payload["pkg_power"] = power_values.to_numpy(dtype=float)

        return pd.DataFrame(fallback_payload)

    gpu_usage = {}
    dgpu_candidates = []  # Collect all dGPU metrics for comparison
    render_type_map = _discover_render_device_types()

    rds = get_gpu_renders()

    if len(rds) == 0:
        return

    for rd in rds:
        rd_name = rd.split("/")[-1]
        gpu_dev = render_type_map.get(rd_name, "unknown")
        if gpu_dev == "unknown":
            gpu_dev = "igpu" if "128" in rd_name else "dgpu"

        gpu_file = OUTPUT_DIR / f"gpu_usage_{rd_name}.txt"
        gpu_rst = eval_gpu_usage(str(gpu_file))

        if gpu_rst is not None:
            if gpu_dev == "igpu":
                # iGPU: use directly (typically only one)
                freq_series = gpu_rst["gpu_freq"] if "gpu_freq" in gpu_rst else None
                valid_freq = pd.to_numeric(freq_series, errors="coerce") if freq_series is not None else None
                has_valid_freq = valid_freq is not None and valid_freq.notna().any() and (valid_freq > 0.01).any()

                if not has_valid_freq:
                    igpu_fallback_df = _build_igpu_fallback_from_xpu()
                    if igpu_fallback_df is not None:
                        gpu_rst = igpu_fallback_df
                        logging.info(
                            f"Recovered iGPU frequency/utilization from xpu-smi for {rd_name}: "
                            f"freq_avg={gpu_rst['gpu_freq'].mean():.2f} GHz"
                        )
                    else:
                        logging.warning(
                            f"iGPU intel_gpu_top data invalid for {rd_name} and xpu-smi fallback is unavailable"
                        )

                gpu_usage[gpu_dev] = gpu_rst
            else:
                freq_source = "intel_gpu_top"
                util_source = "intel_gpu_top"
                power_source = "intel_gpu_top_pkg" if "pkg_power" in gpu_rst.columns else "unavailable"

                dgpu_power_file = OUTPUT_DIR / "dgpu_power_dump.txt"
                dgpu_power_rst = eval_dgpu_power(str(dgpu_power_file))

                fallback_df = None
                if dgpu_power_rst is not None and "GPU Frequency (MHz)" in dgpu_power_rst.columns:
                    valid_rows = (~dgpu_power_rst["GPU Frequency (MHz)"].isna()) & (
                        dgpu_power_rst["GPU Frequency (MHz)"] > 0
                    )
                    if valid_rows.any():
                        target_len = int(valid_rows.sum())
                        valid_freq_mhz = dgpu_power_rst.loc[valid_rows, "GPU Frequency (MHz)"]

                        fallback_payload = {
                            "date": dgpu_power_rst.loc[valid_rows, "date"],
                            "gpu_freq": valid_freq_mhz / 1000.0,
                            "gpu_util": _expand_series_to_length(
                                dgpu_power_rst.loc[valid_rows, "GPU Utilization (%)"]
                                if "GPU Utilization (%)" in dgpu_power_rst.columns
                                else (gpu_rst.get("gpu_util") if "gpu_util" in gpu_rst.columns else None),
                                target_len,
                                default_value=np.nan,
                            ),
                        }

                        if "pkg_power" in gpu_rst.columns:
                            fallback_payload["pkg_power"] = _expand_series_to_length(
                                gpu_rst.get("pkg_power"),
                                target_len,
                                default_value=np.nan,
                            )
                            power_source = "intel_gpu_top_pkg"
                        elif "GPU Power (W)" in dgpu_power_rst.columns:
                            power_values = pd.to_numeric(
                                dgpu_power_rst.loc[valid_rows, "GPU Power (W)"], errors="coerce"
                            )
                            if not power_values.isna().all():
                                fallback_payload["pkg_power"] = power_values.to_numpy(dtype=float)
                                power_source = "xpu-smi"

                        if "GPU Utilization (%)" in dgpu_power_rst.columns:
                            util_values = pd.to_numeric(
                                dgpu_power_rst.loc[valid_rows, "GPU Utilization (%)"], errors="coerce"
                            )
                            if util_values.notna().any():
                                util_source = "xpu-smi"
                            elif "gpu_util" in gpu_rst.columns:
                                util_source = "intel_gpu_top"
                            else:
                                util_source = "unavailable"

                        fallback_df = pd.DataFrame(fallback_payload)

                # dGPU: collect for comparison
                freq_avg = gpu_rst["gpu_freq"].mean()
                freq_stddev = gpu_rst["gpu_freq"].std()
                utilization_avg = gpu_rst["gpu_util"].mean()

                # Replace sparse/invalid intel_gpu_top data with full xpu-smi frequency timeline.
                if (
                    fallback_df is not None
                    and len(fallback_df) > 0
                    and ((pd.isna(freq_avg) or freq_avg <= 0.01) or len(gpu_rst) < 3)
                ):
                    gpu_rst = fallback_df
                    freq_source = "xpu-smi"
                    freq_avg = gpu_rst["gpu_freq"].mean()
                    freq_stddev = gpu_rst["gpu_freq"].std()
                    utilization_avg = gpu_rst["gpu_util"].mean()

                if "gpu_util" in gpu_rst.columns:
                    util_valid = pd.to_numeric(gpu_rst["gpu_util"], errors="coerce").notna().any()
                    if not util_valid:
                        util_source = "unavailable"
                else:
                    util_source = "unavailable"

                if "pkg_power" in gpu_rst.columns:
                    power_valid = pd.to_numeric(gpu_rst["pkg_power"], errors="coerce").notna().any()
                    if not power_valid:
                        power_source = "unavailable"
                else:
                    power_source = "unavailable"

                if pd.isna(freq_stddev):
                    freq_stddev = 0.0

                # Validate metrics before adding candidate
                # Filter out invalid data: freq_avg <= 0.01 GHz (10 MHz) indicates no valid data
                if freq_avg > 0.01:
                    dgpu_candidates.append(
                        {
                            "device": rd_name,
                            "metrics": gpu_rst,
                            "freq_avg": freq_avg,
                            "freq_stddev": freq_stddev,
                            "utilization_avg": utilization_avg,
                            "freq_source": freq_source,
                            "util_source": util_source,
                            "power_source": power_source,
                        }
                    )
                    logging.debug(
                        f"Collected dGPU candidate {rd_name}: "
                        f"freq_avg={freq_avg:.2f}, freq_stddev={freq_stddev:.4f}, util={utilization_avg:.2f}%"
                    )
                else:
                    logging.warning(
                        f"Skipping dGPU {rd_name} - invalid metrics (freq_avg={freq_avg:.4f} GHz, "
                        f"util={utilization_avg:.2f}%). GPU telemetry may have failed."
                    )
        elif gpu_dev == "igpu":
            igpu_fallback_df = _build_igpu_fallback_from_xpu()
            if igpu_fallback_df is not None:
                gpu_usage["igpu"] = igpu_fallback_df
                logging.info(
                    f"Recovered iGPU telemetry from xpu-smi fallback for {rd_name}: "
                    f"freq_avg={igpu_fallback_df['gpu_freq'].mean():.2f} GHz"
                )
            else:
                logging.warning(f"Skipping iGPU data - no intel_gpu_top data and no xpu-smi fallback for {rd_name}")
        elif gpu_dev == "dgpu":
            # intel_gpu_top may be sparse in short runs; recover from xpu-smi frequency stream.
            dgpu_power_file = OUTPUT_DIR / "dgpu_power_dump.txt"
            dgpu_power_rst = eval_dgpu_power(str(dgpu_power_file))
            if dgpu_power_rst is not None and "GPU Frequency (MHz)" in dgpu_power_rst.columns:
                valid_rows = (~dgpu_power_rst["GPU Frequency (MHz)"].isna()) & (
                    dgpu_power_rst["GPU Frequency (MHz)"] > 0
                )
                if valid_rows.any():
                    util_source = "unavailable"
                    power_source = "unavailable"
                    fallback_payload = {
                        "date": dgpu_power_rst.loc[valid_rows, "date"],
                        "gpu_freq": dgpu_power_rst.loc[valid_rows, "GPU Frequency (MHz)"] / 1000.0,
                        "gpu_util": _expand_series_to_length(
                            dgpu_power_rst.loc[valid_rows, "GPU Utilization (%)"]
                            if "GPU Utilization (%)" in dgpu_power_rst.columns
                            else None,
                            int(valid_rows.sum()),
                            default_value=np.nan,
                        ),
                    }
                    if "GPU Power (W)" in dgpu_power_rst.columns:
                        power_values = pd.to_numeric(dgpu_power_rst.loc[valid_rows, "GPU Power (W)"], errors="coerce")
                        if not power_values.isna().all():
                            fallback_payload["pkg_power"] = power_values.to_numpy(dtype=float)
                            power_source = "xpu-smi"

                    if "GPU Utilization (%)" in dgpu_power_rst.columns:
                        util_values = pd.to_numeric(
                            dgpu_power_rst.loc[valid_rows, "GPU Utilization (%)"], errors="coerce"
                        )
                        if util_values.notna().any():
                            util_source = "xpu-smi"

                    fallback_df = pd.DataFrame(fallback_payload)
                    freq_avg = fallback_df["gpu_freq"].mean()
                    freq_stddev = fallback_df["gpu_freq"].std()
                    if pd.isna(freq_stddev):
                        freq_stddev = 0.0
                    dgpu_candidates.append(
                        {
                            "device": rd_name,
                            "metrics": fallback_df,
                            "freq_avg": freq_avg,
                            "freq_stddev": freq_stddev,
                            "utilization_avg": 0.0,
                            "freq_source": "xpu-smi",
                            "util_source": util_source,
                            "power_source": power_source,
                        }
                    )
                    logging.info(f"Recovered dGPU frequency from xpu-smi for {rd_name}: freq_avg={freq_avg:.2f} GHz")
        else:
            logging.warning(f"Skipping {gpu_dev} data - eval_gpu_usage returned None for {gpu_file}")

    # Select best dGPU from candidates
    if not dgpu_candidates:
        # Final recovery path: construct synthetic dGPU candidate from xpu-smi frequency timeline.
        dgpu_power_file = OUTPUT_DIR / "dgpu_power_dump.txt"
        dgpu_power_rst = eval_dgpu_power(str(dgpu_power_file))
        if dgpu_power_rst is not None and "GPU Frequency (MHz)" in dgpu_power_rst.columns:
            valid_rows = (~dgpu_power_rst["GPU Frequency (MHz)"].isna()) & (dgpu_power_rst["GPU Frequency (MHz)"] > 0)
            if valid_rows.any():
                synthetic_len = int(valid_rows.sum())
                util_source = "unavailable"
                power_source = "unavailable"

                synthetic_payload = {
                    "date": dgpu_power_rst.loc[valid_rows, "date"],
                    "gpu_freq": dgpu_power_rst.loc[valid_rows, "GPU Frequency (MHz)"] / 1000.0,
                    "gpu_util": _expand_series_to_length(
                        dgpu_power_rst.loc[valid_rows, "GPU Utilization (%)"]
                        if "GPU Utilization (%)" in dgpu_power_rst.columns
                        else None,
                        synthetic_len,
                        default_value=np.nan,
                    ),
                }

                if "GPU Power (W)" in dgpu_power_rst.columns:
                    power_values = pd.to_numeric(dgpu_power_rst.loc[valid_rows, "GPU Power (W)"], errors="coerce")
                    if not power_values.isna().all():
                        synthetic_payload["pkg_power"] = power_values.to_numpy(dtype=float)
                        power_source = "xpu-smi"

                if "GPU Utilization (%)" in dgpu_power_rst.columns:
                    util_values = pd.to_numeric(dgpu_power_rst.loc[valid_rows, "GPU Utilization (%)"], errors="coerce")
                    if util_values.notna().any():
                        util_source = "xpu-smi"

                synthetic_df = pd.DataFrame(synthetic_payload)
                freq_avg = synthetic_df["gpu_freq"].mean()
                freq_stddev = synthetic_df["gpu_freq"].std()
                if pd.isna(freq_stddev):
                    freq_stddev = 0.0

                if freq_avg > 0.01:
                    dgpu_candidates.append(
                        {
                            "device": "xpu-smi-fallback",
                            "metrics": synthetic_df,
                            "freq_avg": freq_avg,
                            "freq_stddev": freq_stddev,
                            "utilization_avg": synthetic_df["gpu_util"].mean(),
                            "freq_source": "xpu-smi",
                            "util_source": util_source,
                            "power_source": power_source,
                        }
                    )
                    logging.info(f"Recovered dGPU candidate from xpu-smi fallback: freq_avg={freq_avg:.2f} GHz")

    if dgpu_candidates:
        best_dgpu = _select_best_dgpu(dgpu_candidates)
        if best_dgpu:
            gpu_usage["dgpu"] = best_dgpu["metrics"]
            # Add metadata for visibility
            gpu_usage["dgpu_device_id"] = best_dgpu["device"]
            gpu_usage["dgpu_count"] = len(dgpu_candidates)
            gpu_usage["dgpu_freq_source"] = best_dgpu.get("freq_source", "unknown")
            gpu_usage["dgpu_util_source"] = best_dgpu.get("util_source", "unknown")
            gpu_usage["dgpu_power_source"] = best_dgpu.get("power_source", "unknown")
            logging.info(f"Selected dGPU: {best_dgpu['device']} (out of {len(dgpu_candidates)} available)")
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
