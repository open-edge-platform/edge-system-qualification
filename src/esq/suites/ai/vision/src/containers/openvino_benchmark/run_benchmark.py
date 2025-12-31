# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import logging
import os
import re
import shlex
import subprocess as sp  # nosec B404
import sys
from itertools import product
from pathlib import Path

import numpy as np
import openvino as ov
import pandas as pd

# from telemetry import telemetry_decorator, is_intel_xeon
from bcmk_telemetry import is_intel_xeon, telemetry_decorator
from ref_bcmk import get_device_ids_and_drm_info, get_ref_platform_and_value
from spr_bcmk import run_ov_bcmk

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = Path(f"{CURR_DIR}/output").resolve()
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "ov_execution.log"),
        # logging.StreamHandler()
    ],
)

"""
Default model precision: INT8
Default device: GPU device(s) on platform
Default batch size: Auto-batch,
"""
# 16 models to be used
benchmarkNetModles = [
    "resnet-50-tf",
    "efficientnet-b0",
    "ssdlite_mobilenet_v2",
    "mobilenet-v2-pytorch",
    "yolo-v5s",
    "yolo-v8s",
    "clip-vit-base-patch16",
]

model_precisions_values = ("FP16", "INT8")
batch_sizes_values = ("auto", "1", "4", "8", "16")

SAVE_DIR = "output/ov_results"


def check_models_4test():
    global benchmarkNetModles
    # Execute the `lscpu` command and capture its output
    lscpu_output = sp.check_output(["lscpu"]).decode("utf-8")

    # Check if the CPU model is Atom or Celeron
    if re.search(r"Model name:[\s]*.*Atom.*", lscpu_output) or re.search(r"Model name:[\s]*.*Celeron.*", lscpu_output):
        # Check if AVX2 is not supported
        if not re.search(r"Flags:[\s]*.*avx2.*", lscpu_output):
            # TODO: check if 'clip-vit-base-patch16' can run on these platforms
            benchmarkNetModles = ["yolo-v5s", "yolo-v8s", "clip-vit-base-patch16"]


def check_npu_device():
    # so far, have to use cpp benchmark_app to check if NPU is available device.
    execution_dir = "/home/dlstreamer/openvino_cpp_samples_build/intel64/Release"

    command = ["./benchmark_app", "-h"]
    result = sp.run(command, cwd=execution_dir, shell=False, text=True, stdout=sp.PIPE, stderr=sp.PIPE)
    lines = result.stdout.splitlines()
    npu_lines = [line for line in lines if "Available target devices" in line and "NPU" in line]

    return len(npu_lines) > 0


def get_available_devices():
    core = ov.Core()
    full_devices = ["CPU"]
    devices = core.get_available_devices()
    gpu_devices = [dev for dev in devices if dev.startswith("GPU")]
    dgpu_devices = [dev for dev in gpu_devices if "iGPU" not in core.get_property(dev, "FULL_DEVICE_NAME")]

    if len(gpu_devices) == 1 and "GPU" in gpu_devices:
        gpu_devices = ["GPU.0"]
    full_devices += gpu_devices
    if "NPU" in devices or check_npu_device():
        full_devices.append("NPU")

    if len(full_devices) > 1:
        multi_devices = ",".join(full_devices)
        full_devices.append(f"MULTI:{multi_devices}")

    return full_devices


def get_gpu_devices(full_devices):
    # return [dev for dev in full_devices if dev != 'CPU' and not dev.startswith('MULTI')]
    return [dev for dev in full_devices if dev.startswith("GPU")]


def get_model_path(model_name, precision):
    assert precision in ("INT8", "FP16"), "model precision is not valid, only be INT8 or FP16."
    # Use mounted models directory from host: /home/dlstreamer/share/models
    # This maps to esq_data/data/vertical/metro/models on the host
    models_dir = os.path.abspath("share/models")
    if model_name not in ("yolo-v5m-416", "yolo-v5s-416"):
        model_path = f"{models_dir}/{model_name}/{precision}/*.xml"
    else:
        if precision == "INT8":
            precision = "FP16-INT8"
        model_path = f"{models_dir}/{model_name}/{precision}/*.xml"
    try:
        return glob.glob(model_path)[0]
    except:
        logging.error(f"Can not find path: {model_path}")
        return None


def update_result_csv(results, model_name):
    ov_results = results.get(model_name, {})
    if not ov_results:
        return
    csv_file_prefix = "ov_result"
    csv_file_path = f"{SAVE_DIR}/{csv_file_prefix}_{model_name}.csv"
    try:
        pre_result = pd.read_csv(csv_file_path)
        # pre_result['Batch'] = pre_result['Batch'].astype(str)
        pre_result = pre_result.set_index(["Model", "Precision", "Device"])
    except FileNotFoundError:
        pre_result = pd.DataFrame()

    current_result = pd.DataFrame(ov_results)
    current_result.replace("", np.nan, inplace=True)
    current_result = current_result.set_index(["Model", "Precision", "Device"])

    # updated / merge previous results
    updated_result = current_result.combine_first(pre_result) if not pre_result.empty else current_result

    updated_result.reset_index(inplace=True)
    updated_result.to_csv(csv_file_path, index=False)


def extract_value(line):
    m = re.search(r"\d+\.\d+", line)
    try:
        return m.group()
    except:
        return "N/A"


def extra_gpu_values(gpu_res, gpu_dev_str):
    # gpu_res contains all gpu telemetry info, like "GPU.0:freq:1.1<br>GPU.1:freq:2.1<br>GPU.2:freq3.1"
    # gpu_dev_str: the gpu device(s) used for test, can be GPU.0, or Multi:GPU.0,GPU.1,CPU
    # expect result: according gpu_dev_str, get all GPU devices used, like GPU.0, GPU.1. then extra each GPU telemetry info from gpu_res

    # Split the gpu_dev_str to get individual devices
    if gpu_dev_str.startswith("Multi:"):
        devices = gpu_dev_str[6:].split(",")
    else:
        devices = [gpu_dev_str]

    gpu_devices = [dev for dev in devices if dev.startswith("GPU.")]

    telemetry_info = gpu_res.split("<br>")

    result = []
    for device in gpu_devices:
        for info in telemetry_info:
            if info.startswith(device):
                split_list = info.split(":")
                last_element = split_list[-1]
                result.append(last_element)
                break

    # Join the result list into a single string with <br> separator
    return "<br>".join(result)


def verify_multi_device(my_multi_dev_str, gt_multi_list):
    if gt_multi_list is None or len(gt_multi_list) == 0:
        return my_multi_dev_str

    gt_multi_str = gt_multi_list[0]
    my_multi_dev_str = my_multi_dev_str.replace(" ", "")
    my_multi_dev_str = my_multi_dev_str.upper()

    my_devs = set(my_multi_dev_str.split(":")[1].split(","))
    gt_devs = set(gt_multi_str.split(":")[1].split(","))
    if my_devs == gt_devs:
        return gt_multi_str

    return my_multi_dev_str


def execute(model_names=None, devices=None, precisions="INT8", batch_sizes="auto,", time=90):
    sep_char = ","
    if model_names is None:
        model_list = benchmarkNetModles
    else:
        model_list = model_names.split(sep_char)
        for _model in model_list:
            assert _model in benchmarkNetModles, f"Model name {_model} is not valid, can be {benchmarkNetModles} "

    all_devices = get_available_devices()
    if devices is None:
        device_list = get_gpu_devices(all_devices)
        if "NPU" in all_devices:
            device_list.append("NPU")
        if not device_list:
            device_list.append("CPU")
    elif devices == "all":
        device_list = all_devices
    else:
        devices = re.sub(r"GPU(?!\.)", "GPU.0", devices)
        # devices = devices.replace("GPU", "GPU.0")

        if "MULTI:" in devices:
            gt_multi_device = [dev for dev in all_devices if dev.startswith("MULTI:")]
            devices = verify_multi_device(devices, gt_multi_device)
            device_list = [devices]
        else:
            device_list = devices.split(sep_char)
        for _dev in device_list:
            assert _dev in all_devices, f"device {_dev} is not valid, can be {all_devices} "

    if precisions is None:
        precision_list = ["INT8"]
    else:
        precision_list = precisions.split(sep_char)
        for _precision in precision_list:
            assert _precision in model_precisions_values, (
                f"Model precision {_precision} is not valid, can be {model_precisions_values} "
            )

    batch_size_list = [
        "auto",
    ]
    # else:
    #     batch_size_list = batch_sizes.split(sep_char)
    #     for _bs in batch_size_list:
    #         assert _bs in batch_sizes_values, f'Model precision {_bs} is not valid, can be {batch_sizes_values} '

    exec_result = {}

    gpu_info = get_device_ids_and_drm_info()
    # group by model
    csv_file_prefix = "ov_result"
    for b_model in model_list:
        exec_result[b_model] = []
        logging.info(f"\n[Starting to run OpenVINO benchmark app with model: {b_model}...]")
        for m_precision, run_device, bs in product(precision_list, device_list, batch_size_list):
            if "NPU" in run_device and "yolo-v8" in b_model and m_precision == "INT8":
                logging.debug(f"Omit INT8 yolo-v8* model {b_model} with NPU device.")
                continue
            elif "NPU" in run_device and "yolo-v4-tf" == b_model:
                logging.debug(f"Omit {b_model} model with NPU device.")
                continue

            if "NPU" in run_device and "clip-vit-base-patch16" == b_model:
                logging.debug(f"Omit {b_model} model with NPU device.")
                continue

            if is_intel_xeon() and run_device == "CPU":
                single_result = execute_singl_spr(run_device, b_model, m_precision, bs, time)
            else:
                single_result = execute_single(run_device, b_model, m_precision, bs, time)
            if bs == "auto":
                (
                    ref_platform,
                    ref_value,
                    ref_freq,
                    ref_power,
                ) = get_ref_platform_and_value(gpu_info, b_model, m_precision, bs, run_device)
            else:
                # ref_platform, ref_value = "", ""
                continue  # only auto batch size to test.

            device_freq = (
                single_result["CPU_Freq"]
                if run_device == "CPU"
                else extra_gpu_values(single_result["GPU_Freq"], run_device)
            )
            exec_result[b_model].append(
                {
                    "Model": b_model,
                    "Precision": m_precision,
                    "Device": run_device,
                    # 'Batch': bs,
                    "Throughput": single_result["throughput"],
                    "Latency": single_result["latency"],
                    "Dev Freq": device_freq,
                    "Pkg Power": single_result["Package Power"],
                    "Ref Platform": ref_platform,
                    "Ref Throughput": ref_value,
                    "Ref Dev Freq": ref_freq,
                    "Ref Pkg Power": ref_power,
                    "Duration(s)": time,
                    "Result": "No Error" if single_result["throughput"] else "FAIL",
                }
            )

        update_result_csv(exec_result, b_model)


@telemetry_decorator
def execute_singl_spr(device, model_name, precision, batch_size="auto", time=90):
    execution_dir = "/home/dlstreamer/openvino_cpp_samples_build/intel64/Release"
    model_path = get_model_path(model_name, precision)

    return run_ov_bcmk(execution_dir, model_path, time=time)


@telemetry_decorator
def execute_single(device, model_name, precision, batch_size="auto", time=90):
    execution_dir = "/home/dlstreamer/openvino_cpp_samples_build/intel64/Release"
    model_path = get_model_path(model_name, precision)
    rst = {"latency": "", "throughput": ""}

    batch_opt = ""
    # if batch_size == "auto":
    #     if device.startswith('GPU'):
    #         device = f"BATCH:{device}"
    # else:
    #     batch_opt = f"-b {batch_size}"

    # if "yolo-v8" in model_name:
    #     _cmd = f"./benchmark_app -m {model_path} -d {device} {batch_opt} -data_shape '[1,3,640,640]' -t {time}"
    # else:
    _cmd = f"./benchmark_app -m {model_path} -d {device} {batch_opt}  -t {time}"
    # Note: CLIP vision model uses pixel_values input, not input_ids
    # The model is already converted with correct input shape [1,3,224,224]
    # No need to specify -data_shape parameter
    logging.info(_cmd)
    # return rst
    try:
        output = sp.check_output(shlex.split(_cmd), cwd=execution_dir, stderr=sp.STDOUT)
        for line in output.decode("utf-8").split("\n"):
            if line.strip().startswith("[ INFO ]    Average:"):
                rst["latency"] = extract_value(line)
            elif line.strip().startswith("[ INFO ] Throughput:"):
                rst["throughput"] = extract_value(line)
    except sp.CalledProcessError as ex:
        print(f"Execute benchmark app with model {model_name} Failed. ")
        logging.error(ex.returncode)
        logging.error(ex.output)

    return rst


if __name__ == "__main__":
    check_models_4test()

    parser = argparse.ArgumentParser(description="EAB OpenVINO Benchmark Tests.")
    parser.add_argument("-d", "--devices", default=None, help=f"Avaliable Devices: {get_available_devices()}")
    parser.add_argument("-m", "--models", default=None, help=f"OV models to use. Models: {benchmarkNetModles}")
    # parser.add_argument('-b', '--batch_sizes', default='auto,8', help='Batch Size: [auto,1,4,8,16], multiple values can be specified and separated by commas .')
    parser.add_argument("-p", "--precisions", default="INT8", help="Model Precision to use: [FP16,INT8].")
    parser.add_argument("-t", "--time", default=90, type=int, help="Time to run the benchmark.")
    # parser.add_argument('-o', '--result_dir', default=SAVE_DIR, help="the folder to save csv result files.")

    args = parser.parse_args()
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {SAVE_DIR}: {e}")

    # Sanitize time parameter to prevent command injection
    # Ensure it's a valid positive integer within reasonable range (1-3600 seconds)
    try:
        sanitized_time = int(args.time)
        if sanitized_time < 1 or sanitized_time > 3600:
            print(f"Error: time parameter must be between 1 and 3600 seconds. Got: {sanitized_time}")
            sys.exit(1)
    except (ValueError, TypeError):
        print(f"Error: time parameter must be a valid integer. Got: {args.time}")
        sys.exit(1)

    batch_args = "auto,"
    execute(args.models, args.devices, args.precisions, batch_args, sanitized_time)
