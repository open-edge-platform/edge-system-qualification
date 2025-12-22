# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import os
import re
import shlex
import shutil
import subprocess as sp  # nosec B404
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import openvino as ov
import pandas as pd
from bcmk_telemetry import is_intel_xeon, telemetry_decorator

mpl.use("Agg")
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

if len(sys.argv) < 3 or not sys.argv[2]:
    logger.error("Error: Missing numeric argument")
    sys.exit(1)

device_arg = sys.argv[1]
time_arg = sys.argv[2]

__Device = device_arg  # string directly, no need to str()
if not isinstance(__Device, str):
    logger.error(f"Device argument is invalid: '{device_arg}'")
    sys.exit(1)

try:
    __TimeOfStableRun = float(time_arg)  # hours
except ValueError:
    logger.error(f"Input duration is invalid: '{time_arg}'")
    sys.exit(1)

if __TimeOfStableRun < 0.01:
    __TimeOfStableRun = 0.01

__TimeOfStableRun = int(__TimeOfStableRun * 3600)

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
mpl.rcParams["font.size"] = 11
# Time of GpuWatch should be modified in shell scripts.

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = Path(f"{CURR_DIR}/output").resolve()
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()
EXECUTION_DIR = "/home/dlstreamer/openvino_cpp_samples_build/intel64/Release"

LOG_PREF = re.sub(r"\W+", "", __Device)
OUTPUT_LOG = f"{LOG_PREF}-IntelVideoAIBoxDetails.log"
RUNTIME_LOG = f"{LOG_PREF}-stable_runtime.txt"
OUTPUT_IMG = f"{LOG_PREF}_InferencePerformance_FreqPlot.png"


def LogConfig():
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(OUTPUT_DIR / f"{OUTPUT_LOG}")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d][%(filename)s][%(funcName)s][%(lineno)d]%(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(streamHandler)


def Initial():
    abs_path = get_file_abs_path("runtime")
    # Remove file safely
    if os.path.exists(abs_path):
        try:
            os.remove(abs_path)
            logger.info(f"Removed old log file: {abs_path}")
        except OSError as e:
            logger.error(f"Failed to remove {abs_path}: {e}")
    else:
        os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
        logger.info("Creating result dir ...")

    logger.info("+------------------------------+")
    logger.info("|   Intel AI Box Test Tool     |")
    logger.info("+------------------------------+")
    logger.info("|Test Processor: {} |".format(GetProcessorName()))
    logger.info("+-------------------------------------------+")
    logger.info("|Test StartTime: {} |".format(datetime.datetime.now()))
    logger.info("+-------------------------------------------+")


def get_file_abs_path(file_type):
    """Ensure file_path is inside OUTPUT_DIR and name is safe."""
    base_dir = os.path.abspath(f"{OUTPUT_DIR}")

    if file_type == "outlog":
        file_path = os.path.join(base_dir, OUTPUT_LOG)
    elif file_type == "runtime":
        file_path = os.path.join(base_dir, RUNTIME_LOG)
    elif file_type == "outimg":
        file_path = os.path.join(base_dir, OUTPUT_IMG)

    abs_path = Path(file_path).resolve()

    # Ensure the target file stays inside OUTPUT_DIR
    if not str(abs_path).startswith(str(base_dir) + os.sep):
        logger.error(f"Invalid path: {abs_path} not within {base_dir}")
        sys.exit(1)

    # Allow only safe filename characters
    filename = abs_path.name
    if not re.match(r"^[\w.\-]+$", filename):
        logger.error(f"Unsafe filename detected: {filename}")
        sys.exit(1)

    return abs_path


def check_npu_device():
    # so far, have to use cpp benchmark_app to check if NPU is available device.

    benchmark_path = shutil.which("benchmark_app")
    if not benchmark_path:
        raise FileNotFoundError("benchmark_app not found")

    result = sp.run(
        [benchmark_path, "-h"], cwd=EXECUTION_DIR, shell=False, text=True, stdout=sp.PIPE, stderr=sp.PIPE, check=True
    )
    lines = result.stdout.splitlines()
    npu_lines = [line for line in lines if "Available target devices" in line and "NPU" in line]

    return len(npu_lines) > 0


def get_gpu_device():
    core = ov.Core()
    devices = core.get_available_devices()
    gpu_devices = [dev for dev in devices if dev.startswith("GPU")]
    dgpu_devices = [dev for dev in gpu_devices if "iGPU" not in core.get_property(dev, "FULL_DEVICE_NAME")]

    if not gpu_devices:
        test_devices = ["CPU"]
    elif not dgpu_devices:
        test_devices = gpu_devices
    else:
        test_devices = dgpu_devices

    if len(test_devices) > 1:
        multi_devices = ",".join(test_devices)
        return f"MULTI:{multi_devices}"
    else:
        return "GPU.0" if test_devices[0] == "GPU" else test_devices[0]


def GetProcessorName():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line.lower():
                    cpu_info = line.strip().split(":", 1)[1].strip()

                    # Default processor name
                    processor_name = cpu_info

                    # Match variants like "Intel(R) Core(TM) i7-12700H"
                    if "Core(TM)" in cpu_info:
                        match = re.search(r"Core\(TM\)\s+(.*)", cpu_info)
                        if match:
                            processor_name = match.group(1).strip()

                    elif "Celeron(R)" in cpu_info or "Xeon(R)" in cpu_info:
                        match = re.search(r"\(R\)\s+(.*)", cpu_info)
                        if match:
                            processor_name = match.group(1).strip()

                    return processor_name

        return "Unknown Processor"

    except Exception as e:
        return f"Error reading CPU info: {e}"


def TestStarter():
    telemetry_data = RunStable_Bcmk()
    cpu_util_list = telemetry_data["CPU_Usage"]
    cpu_freq_list = telemetry_data["CPU_Freq"]

    gpu_usage = telemetry_data.get("GPU_Usage", {})
    if gpu_usage:
        igpu_usage_list = gpu_usage.get("igpu", None)
        dgpu_usage_list = gpu_usage.get("dgpu", None)
        # Extract dGPU selection metadata
        dgpu_device_id = gpu_usage.get("dgpu_device_id", None)
        dgpu_count = gpu_usage.get("dgpu_count", 0)
        dgpu_power_list = telemetry_data.get("dGPU_Power", None)
    else:
        igpu_usage_list = None
        dgpu_usage_list = None
        dgpu_device_id = None
        dgpu_count = 0
        dgpu_power_list = None

    cpu_merged_df = pd.merge(
        cpu_util_list[["date", "sum"]], cpu_freq_list[["date", "max_freq"]], on="date", how="inner"
    )

    dfs = [cpu_merged_df, igpu_usage_list, dgpu_usage_list, dgpu_power_list]
    min_row_num = min(len(df) for df in dfs if df is not None and not df.empty)

    # align the data
    # min_row_num = min(len(cpu_merged_df), len(igpu_usage_list))

    cpu_merged_df_adjusted = cpu_merged_df.iloc[-min_row_num:]
    if igpu_usage_list is not None:
        igpu_usage_list_adjusted = igpu_usage_list.iloc[-min_row_num:]
        igpu_usage_list_adjusted["date"] = cpu_merged_df_adjusted["date"].values
    else:
        igpu_usage_list_adjusted = None

    if dgpu_usage_list is not None:
        dgpu_usage_list_adjusted = dgpu_usage_list.iloc[-min_row_num:]
        dgpu_usage_list_adjusted["date"] = cpu_merged_df_adjusted["date"].values
    else:
        dgpu_usage_list_adjusted = None

    if dgpu_power_list is not None:
        dgpu_power_list_adjusted = dgpu_power_list.iloc[-min_row_num:]
        # dgpu_usage_list_adjusted['date'] = cpu_merged_df_adjusted['date'].values
    else:
        dgpu_power_list_adjusted = None

    ParseStableRuntime(get_file_abs_path("outlog"))
    # draw the data in one graph - pass dGPU metadata
    draw_graph(cpu_merged_df_adjusted, igpu_usage_list_adjusted, dgpu_usage_list_adjusted, dgpu_power_list_adjusted, dgpu_device_id, dgpu_count)


def extract_value(line):
    m = re.search(r"\d+\.\d+", line)
    try:
        return m.group()
    except:
        return ""


def fetch_cpu_info():
    # Function to execute a command and return its output
    def execute_command(command):
        result = sp.run(command, shell=False, text=True, capture_output=True, check=True)
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
    sockets_num = find_value(r"Socket\(s\):\s*(\d+)")
    cores_per_socket = find_value(r"Core\(s\) per socket:\s*(\d+)")
    numa_nodes_num = find_value(r"NUMA node\(s\):\s*(\d+)")

    # Calculating values
    physical_cores_num = sockets_num * cores_per_socket if sockets_num and cores_per_socket else None
    cores_per_node = physical_cores_num // numa_nodes_num if physical_cores_num and numa_nodes_num else None
    cores_per_instance = cores_per_node

    # Returning the calculated values
    return {
        "sockets_num": sockets_num,
        "cores_per_socket": cores_per_socket,
        "physical_cores_num": physical_cores_num,
        "numa_nodes_num": numa_nodes_num,
        "cores_per_node": cores_per_node,
        "cores_per_instance": cores_per_instance,
    }


def run_spr_ov_bcmk(bcmk_dir, model_path, exec_cores_per_socket=0, time=90):
    benchmark_app_bin = bcmk_dir + "/benchmark_app"

    cpu_info = fetch_cpu_info()
    logging.info(f"Get CPU INFO: {cpu_info}")

    sockets_num = cpu_info["sockets_num"]
    cores_per_socket = cpu_info["cores_per_socket"]
    exec_cores_per_socket = cores_per_socket if exec_cores_per_socket <= 0 else exec_cores_per_socket

    numa_cmds = []
    for i, socket in enumerate(range(sockets_num)):
        start_core_idx = i * cores_per_socket
        end_core_idx = start_core_idx + exec_cores_per_socket - 1
        numa_cmd = [
            "numactl",
            "-m",
            str(i),
            "--physcpubind",
            f"{start_core_idx}-{end_core_idx}",
            benchmark_app_bin,
            "-m",
            model_path,
            "-t",
            str(time),
        ]
        logging.info(f"Benchmark App command with numactl: {numa_cmd}")
        numa_cmds.append(numa_cmd)

    raw_out = []
    try:
        for _cmd in numa_cmds:
            # Ensure the command is a list, not a string
            if isinstance(_cmd, str):
                _cmd = shlex.split(_cmd)

            # Safety: only allow numactl commands
            if not _cmd or not _cmd[0].endswith("numactl"):
                raise ValueError(f"Untrusted command: {_cmd}")

            result = sp.run(
                _cmd,
                text=True,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                timeout=time + 30,
                check=False,  # We handle errors manually
            )

            if result.stderr:
                logging.error(f"Error occurred: {result.stderr}")

            raw_out.append(result.stdout)
    except sp.CalledProcessError as ex:
        logger.error(f"Execute benchmark app with model {model_name} on SPR platform Failed. ")
        logger.error(ex.returncode)
        logger.error(ex.output)

    return raw_out


@telemetry_decorator
def RunStable_Bcmk():
    test_model_name = choose_model_4test()
    share_base = "/home/dlstreamer/share"
    models_base = f"{share_base}/models"  # Models mounted from esq_data/data/vertical/metro/models
    images_base = f"{share_base}/images"  # Images mounted from esq_data/data/vertical/metro/images

    # Model path structure: share/models/{model}/FP16/{model}.xml
    # Mounted from: esq_data/data/vertical/metro/models
    possible_paths = [
        f"{models_base}/{test_model_name}/FP16/{test_model_name}.xml",
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            logger.info(f"Found model at: {model_path}")
            break

    if not model_path:
        logger.error(f"Model file not found for: {test_model_name}")
        logger.error("Searched paths:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        # List what's actually in the share directory
        if os.path.exists(share_base):
            logger.error(f"Contents of {share_base}:")
            for root, dirs, files in os.walk(share_base):
                level = root.replace(share_base, "").count(os.sep)
                indent = " " * 2 * level
                logger.error(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files[:10]:  # Limit files shown
                    logger.error(f"{subindent}{file}")
        raise FileNotFoundError(f"Model XML not found for: {test_model_name}")

    # Test image is in the images directory
    test_image_path = f"{images_base}/car.png"

    if not os.path.exists(test_image_path):
        logger.error(f"Test image not found: {test_image_path}")
        logger.error(f"Expected location: {images_base}/car.png")
        # List images directory contents
        if os.path.exists(images_base):
            logger.error(f"Contents of {images_base}: {os.listdir(images_base)}")
        raise FileNotFoundError(f"Test image not found: {test_image_path}")

    logger.info(f"Using model: {test_model_name} at {model_path}")
    logger.info(f"Using test image: {test_image_path}")

    if not is_intel_xeon() or __Device != "CPU":
        # Build command as list for proper argument handling
        cmd_list = [
            "./benchmark_app",
            "-d",
            __Device,
            "-m",
            model_path,
            "-i",
            test_image_path,
            "-t",
            str(__TimeOfStableRun),
        ]

        logger.info(f"Running benchmark_app command: {' '.join(cmd_list)}")

        try:
            result = sp.run(cmd_list, cwd=EXECUTION_DIR, capture_output=True, text=True, check=True)
            output_filename = get_file_abs_path("outlog")

            try:
                with open(output_filename, "w") as output_file:
                    output_file.write(result.stdout)
            except OSError as e:
                logger.error(f"Error writing to file {output_filename}: {e}")

        except sp.CalledProcessError as e:
            logger.error(f"benchmark_app failed with exit code {e.returncode}")
            logger.error(f"Command: {' '.join(cmd_list)}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")

            # Try to provide helpful diagnostics
            logger.error("=== Diagnostics ===")
            logger.error(f"Model XML exists: {os.path.exists(model_path)}")
            logger.error(f"Image file exists: {os.path.exists(test_image_path)}")
            logger.error(f"benchmark_app exists: {os.path.exists(os.path.join(EXECUTION_DIR, 'benchmark_app'))}")

            # Check model file sizes
            if os.path.exists(model_path):
                try:
                    xml_size = os.path.getsize(model_path)
                    logger.error(f"Model XML size: {xml_size} bytes")
                    bin_path = model_path.replace(".xml", ".bin")
                    if os.path.exists(bin_path):
                        bin_size = os.path.getsize(bin_path)
                        logger.error(f"Model BIN size: {bin_size} bytes")
                    else:
                        logger.error(f"Model BIN file MISSING: {bin_path}")
                except Exception as size_err:
                    logger.error(f"Error checking model files: {size_err}")

            raise
    else:
        out_list = run_spr_ov_bcmk(EXECUTION_DIR, model_path, time=__TimeOfStableRun)
        output_filename = get_file_abs_path("outlog")
        try:
            with open(output_filename, "w") as output_file:
                for out_line in out_list:
                    output_file.write(out_line)
        except OSError as e:
            logger.error(f"Error writing to file {output_filename}: {e}")


def ParseStableRuntime(file_path):
    Network = 0  # Use Yolo_v4 Network
    nn_model = choose_model_4test()
    pattern = r"Throughput: (.*?) FPS"

    output_filename = get_file_abs_path("runtime")
    try:
        with open(output_filename, "w") as result:
            result.write("Network: {} Device: {}\n".format(nn_model, __Device))
    except OSError as e:
        logger.error(f"Error writing to file {output_filename}: {e}")

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            match = re.findall(pattern, line)
            if match:
                fps_log = float(match[0].strip())
                runtime_file = get_file_abs_path("runtime")
                try:
                    with open(get_file_abs_path("runtime"), "a") as result:
                        result.write(
                            "Duration time: {:.3f}h Average FPS: {:.2f}\n".format(__TimeOfStableRun / 3600, fps_log)
                        )
                    logger.info("Duration time: {:.3f}h Average FPS: {:.2f}".format(__TimeOfStableRun / 3600, fps_log))
                except OSError as e:
                    logger.error(f"Error writing to file {runtime_file}: {e}")


def DataParser(output_folder):
    ParseStableRuntime(get_file_abs_path("outlog"))

def draw_graph(cpu_df, igpu_df, dgpu_df, dgpu_power_df, dgpu_device_id=None, dgpu_count=0):
    # Initialize Metrics with default values including stability metrics
    summary_df = pd.DataFrame(
        [
            {
                "Function": "AI-Freq-LR",
                # iGPU performance metrics
                "frequency_max_igpu": 0.0,
                "utilization_igpu": 0.0,
                "max_power_igpu": 0.0,
                # iGPU stability metrics (lower is better)
                "frequency_stddev_igpu": 0.0,
                "frequency_min_igpu": 0.0,
                "frequency_range_igpu": 0.0,
                # dGPU performance metrics
                "frequency_max_dgpu": 0.0,
                "utilization_dgpu": 0.0,
                "max_power_dgpu": 0.0,
                # dGPU stability metrics (lower is better)
                "frequency_stddev_dgpu": 0.0,
                "frequency_min_dgpu": 0.0,
                "frequency_range_dgpu": 0.0,
                # CPU metrics
                "utilization_cpu": 0.0,
                "frequency_max_cpu": 0.0,
                # dGPU selection metadata
                "dgpu_device_id": dgpu_device_id if dgpu_device_id else "N/A",
                "dgpu_count": dgpu_count,
            }
        ]
    )

    cpu_df["date"] = pd.to_datetime(cpu_df["date"])
    cpu_df.sort_values("date", inplace=True)
    if igpu_df is not None:
        igpu_df["date"] = pd.to_datetime(igpu_df["date"])
        igpu_df.sort_values("date", inplace=True)
    if dgpu_df is not None:
        dgpu_df["date"] = pd.to_datetime(dgpu_df["date"])
        dgpu_df.sort_values("date", inplace=True)

    if igpu_df is not None and "pkg_power" in igpu_df.columns or dgpu_df is not None and "pkg_power" in dgpu_df.columns:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Device Utilization (%)", color="tab:blue")
    ax1.plot(cpu_df["date"], cpu_df["sum"], label="Norm CPU Utilization", color="#000080", linestyle="-")

    summary_df.at[0, "utilization_cpu"] = round(cpu_df["sum"].mean(), 2)

    if igpu_df is not None:
        ax1.plot(igpu_df["date"], igpu_df["gpu_util"], label="iGPU Utilization", color="#4169E1", linestyle="--")
        summary_df.at[0, "utilization_igpu"] = round(igpu_df["gpu_util"].mean(), 2)
    if dgpu_df is not None:
        ax1.plot(dgpu_df["date"], dgpu_df["gpu_util"], label="dGPU Utilization", color="#87CEEB", linestyle="-.")
        summary_df.at[0, "utilization_dgpu"] = round(dgpu_df["gpu_util"].mean(), 2)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(-5, 110)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax2 = ax2
    ax2.set_ylabel("Device Frequency (GHz)", color="tab:orange")
    ax2.plot(cpu_df["date"], cpu_df["max_freq"], label="CPU Max Frequency", color="#FF8C00", marker="x", linestyle="-")
    summary_df.at[0, "frequency_max_cpu"] = round(cpu_df["max_freq"].mean(), 2)

    if igpu_df is not None:
        ax2.plot(
            igpu_df["date"], igpu_df["gpu_freq"], label="iGPU Frequency", color="#FFA500", marker="x", linestyle="--"
        )
        # Performance metrics
        summary_df.at[0, "frequency_max_igpu"] = round(igpu_df["gpu_freq"].mean(), 2)
        # Stability metrics (lower is better for consistency)
        summary_df.at[0, "frequency_stddev_igpu"] = round(igpu_df["gpu_freq"].std(), 4)
        summary_df.at[0, "frequency_min_igpu"] = round(igpu_df["gpu_freq"].min(), 2)
        freq_range_igpu = igpu_df["gpu_freq"].max() - igpu_df["gpu_freq"].min()
        summary_df.at[0, "frequency_range_igpu"] = round(freq_range_igpu, 4)

    if dgpu_df is not None:
        ax2.plot(
            dgpu_df["date"], dgpu_df["gpu_freq"], label="dGPU Frequency", color="#FFD700", marker="x", linestyle="-."
        )
        # Performance metrics
        summary_df.at[0, "frequency_max_dgpu"] = round(dgpu_df["gpu_freq"].mean(), 2)
        # Stability metrics (lower is better for consistency)
        summary_df.at[0, "frequency_stddev_dgpu"] = round(dgpu_df["gpu_freq"].std(), 4)
        summary_df.at[0, "frequency_min_dgpu"] = round(dgpu_df["gpu_freq"].min(), 2)
        freq_range_dgpu = dgpu_df["gpu_freq"].max() - dgpu_df["gpu_freq"].min()
        summary_df.at[0, "frequency_range_dgpu"] = round(freq_range_dgpu, 4)

    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(-1, 2.5)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    show_power_plot = False
    if igpu_df is not None and "pkg_power" in igpu_df.columns or dgpu_df is not None and "pkg_power" in dgpu_df.columns:
        show_power_plot = True
        ax3_right_spine = ax3

    if show_power_plot:
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Power (W)", color="tab:purple")
        if igpu_df is not None:
            ax3_right_spine.plot(
                igpu_df["date"],
                igpu_df["pkg_power"],
                label="iGPU Power",
                color="#1E90FF",
                marker="o",
                linestyle="--",
            )
            summary_df.at[0, "max_power_igpu"] = round(igpu_df["pkg_power"].mean(), 2)

        # Plot dGPU power - prefer xpu-smi data, fallback to pkg_power from intel_gpu_top
        dgpu_power_plotted = False
        if dgpu_power_df is not None:
            # Calculate mean, skipping NaN values (converted from N/A)
            power_mean = dgpu_power_df["GPU Power (W)"].mean(skipna=True)
            # Only use xpu-smi data if we have valid values
            if not pd.isna(power_mean) and power_mean > 0:
                ax3_right_spine.plot(
                    dgpu_power_df["date"],
                    dgpu_power_df["GPU Power (W)"],
                    label="dGPU Power (xpu-smi)",
                    color="#DC143C",
                    marker="o",
                    linestyle="-.",
                )
                summary_df.at[0, "max_power_dgpu"] = round(power_mean, 2)
                dgpu_power_plotted = True

        # Fallback to pkg_power from intel_gpu_top if xpu-smi data not available
        if not dgpu_power_plotted and dgpu_df is not None and "pkg_power" in dgpu_df.columns:
            pkg_power_mean = dgpu_df["pkg_power"].mean(skipna=True)
            if not pd.isna(pkg_power_mean):
                ax3_right_spine.plot(
                    dgpu_df["date"],
                    dgpu_df["pkg_power"],
                    label="dGPU Power (pkg)",
                    color="#DC143C",
                    marker="o",
                    linestyle="-.",
                )
                summary_df.at[0, "max_power_dgpu"] = round(pkg_power_mean, 2)
                dgpu_power_plotted = True

        # Default to 0.0 if no power data available
        if not dgpu_power_plotted and dgpu_df is not None:
            summary_df.at[0, "max_power_dgpu"] = 0.0

        ax3_right_spine.tick_params(axis="y", labelcolor="tab:purple")
        ax3_right_spine.set_ylim(-5, 120)
        ax3_right_spine.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    runtime_file = get_file_abs_path("runtime")
    try:
        with open(runtime_file, "r") as f:
            title = f.read()
            # Add suffix if multiple dGPUs detected and best device selected
            if dgpu_count > 1 and dgpu_device_id:
                title = f"{title} - Best Device: {dgpu_device_id}"
            fig.suptitle(title)
    except OSError as e:
        logger.error(f"Error reading runtime file {runtime_file}: {e}")

    fig.autofmt_xdate()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    logger.info(f"Creating the PNG Image and saving it {OUTPUT_DIR}/")
    plt.savefig(get_file_abs_path("outimg"))

    summary_df.to_csv(f"{OUTPUT_DIR}/averages_summary.csv", index=False)

    logger.info("Averages saved to averages_summary.csv")

    # Log dGPU selection information
    if dgpu_count > 0:
        logger.info(f"dGPU Selection Summary: {dgpu_count} device(s) detected")
        if dgpu_device_id:
            logger.info(f"  Selected device: {dgpu_device_id}")
            if dgpu_count > 1:
                logger.info("  Selection based on: highest avg frequency + lowest stddev (stability)")
        logger.info(f"  dGPU metrics in summary: frequency_max={summary_df.at[0, 'frequency_max_dgpu']:.2f} GHz, "
                   f"stddev={summary_df.at[0, 'frequency_stddev_dgpu']:.4f}, "
                   f"utilization={summary_df.at[0, 'utilization_dgpu']:.2f}%")

def choose_model_4test():
    """
    Determine which model to use for testing.

    Priority:
    1. MODEL_NAME environment variable (set by test framework)
       - Test framework handles CPU detection and selects appropriate model
       - Atom/Celeron → yolov5n (set by test)
       - Core/Xeon → yolov5s (set by test)
    2. Default to yolov5s if MODEL_NAME not set

    Returns:
        str: Model name (e.g., 'yolov5n', 'yolov5s', 'yolov5m')
    """
    # Check for MODEL_NAME environment variable (set by test framework)
    # Test framework already handles CPU-based model selection
    env_model = os.environ.get("MODEL_NAME")
    if env_model:
        logger.info(f"Using model from test framework: {env_model}")
        return env_model

    # Default fallback (should rarely be used - test framework sets MODEL_NAME)
    logger.info("MODEL_NAME not set, defaulting to yolov5s")
    return "yolov5s"

def main():
    output_folder = f"{OUTPUT_DIR}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    LogConfig()
    logger.info("Initializing the AI Freq Runner!!")
    Initial()
    TestStarter()
    # DataParser(output_folder)


if __name__ == "__main__":
    # __Device = get_gpu_device()
    logger.info(f"AI Frequency Runner Input Device: {__Device}")
    logger.info(f"AI FRequency Runner Input Test Duration: {__TimeOfStableRun}")
    main()
