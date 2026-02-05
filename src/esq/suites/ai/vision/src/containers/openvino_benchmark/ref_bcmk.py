# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import logging
import os
import re
import shlex
import signal
import subprocess  # nosec B404 # For GPU detection and OpenVINO benchmark execution
import time
from pathlib import Path

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

iGPU_Dev_IDs = [
    "7DD5",
    "7D40",
    "7D45",
    "7D55",
    "7D60",
    "A7A9",
    "A7A8",
    "A7A1",
    "A7A0",
    "A721",
    "A720",
    "A78B",
    "A78A",
    "A789",
    "A788",
    "A783",
    "A782",
    "A781",
    "A780",
    "4907",
    "4905",
    "4680",
    "4682",
    "4688",
    "468A",
    "468B",
    "4690",
    "4692",
    "4693",
    "46D0",
    "46D1",
    "46D2",
    "4626",
    "4628",
    "462A",
    "46A0",
    "46A1",
    "46A2",
    "46A3",
    "46A6",
    "46A8",
    "46AA",
    "46B0",
    "46B1",
    "46B2",
    "46B3",
    "46C0",
    "46C1",
    "46C2",
    "46C3",
    "4C8A",
    "4C8B",
    "4C90",
    "4C9A",
    "4C8C",
    "4C80",
    "4E71",
    "4E61",
    "4E57",
    "4E55",
    "4E51",
    "4571",
    "4557",
    "4555",
    "4551",
    "4541",
    "9A59",
    "9A60",
    "9A68",
    "9A70",
    "9A40",
    "9A49",
    "9A78",
    "9AC0",
    "9AC9",
    "9AD9",
    "9AF8",
]

# Feb 03 2026 : Updated from https://github.com/intel/compute-runtime - added Panther Lake, Nova Lake, Battlemage

iGPU_Dev_IDs = [
    "64A0",  # ['IntelÂ® Arcâ\x84¢ Graphics', 'Xe2', 'Lunar Lake', '6.11', '64/56']
    "6420",  # ['IntelÂ® Graphics', 'Xe2', 'Lunar Lake', '6.11', '64/56']
    "64B0",  # ['IntelÂ® Graphics', 'Xe2', 'Lunar Lake', '6.11', '32']
    "7DD5",  # Pre-production ARL-H GPU
    "7D51",  # ['IntelÂ® Graphics', 'Xe-LPG', 'Arrow Lake-H', '6.9', '128/112']
    "7D67",  # ['IntelÂ® Graphics', 'Xe-LPG', 'Arrow Lake-S', '6.7', '64/48/32']
    "7D41",  # ['IntelÂ® Graphics', 'Xe-LPG', 'Arrow Lake-U', '6.9', '64']
    "7DD5",  # ['IntelÂ® Graphics', 'Xe-LPG', 'Meteor Lake', '6.7', '128/112']
    "7D45",  # ['IntelÂ® Graphics', 'Xe-LPG', 'Meteor Lake', '6.7', '64/48']
    "7D40",  # ['IntelÂ® Graphics', 'Xe-LPG', 'Meteor Lake', '6.7', '64/48']
    "7D55",  # ['IntelÂ® Arcâ\x84¢ Graphics', 'Xe-LPG', 'Meteor Lake', '6.7', '128/112']
    "A780",  # ['IntelÂ® UHD Graphics 770', 'Xe', 'Raptor Lake-S', '5.17', '32']
    "A781",  # ['IntelÂ® UHD Graphics', 'Xe', 'Raptor Lake-S', '5.17', '32']
    "A788",  # ['IntelÂ® UHD Graphics', 'Xe', 'Raptor Lake-S', '5.17', '32']
    "A789",  # ['IntelÂ® UHD Graphics', 'Xe', 'Raptor Lake-S', '5.17', '32']
    "A78A",  # ['IntelÂ® UHD Graphics', 'Xe', 'Raptor Lake-S', '5.19', '24']
    "A782",  # ['IntelÂ® UHD Graphics 730', 'Xe', 'Raptor Lake-S', '5.17', '24']
    "A78B",  # ['IntelÂ® UHD Graphics', 'Xe', 'Raptor Lake-S', '5.19', '16']
    "A783",  # ['IntelÂ® UHD Graphics 710', 'Xe', 'Raptor Lake-S', '5.17', '16']
    "A7A0",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Raptor Lake-P', '5.19', '96/80']
    "A7A1",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Raptor Lake-P', '5.19', '96/80']
    "A7A8",  # ['IntelÂ® UHD Graphics', 'Xe', 'Raptor Lake-P', '5.19', '64/48']
    "A7AA",  # ['IntelÂ® Graphics', 'Xe', 'Raptor Lake-P', '6.7', '96/80']
    "A7AB",  # ['IntelÂ® Graphics', 'Xe', 'Raptor Lake-P', '6.7', '64/48']
    "A7AC",  # ['IntelÂ® Graphics', 'Xe', 'Raptor Lake-U', '6.7', '96/80']
    "A7AD",  # ['IntelÂ® Graphics', 'Xe', 'Raptor Lake-U', '6.7', '64/48']
    "A7A9",  # ['IntelÂ® UHD Graphics', 'Xe', 'Raptor Lake-P', '5.19', '64/48']
    "A721",  # ['IntelÂ® UHD Graphics', 'Xe', 'Raptor Lake-P', '5.19', '96/80']
    "4905",  # ['IntelÂ® IrisÂ® Xe MAX Graphics', 'Xe', 'DG1', '5.16*', '96']
    "4907",  # ['Intel Server GPU SG-18M', 'Xe', 'DG1', '5.16*', '96']
    "4908",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'DG1', '5.16*', '80']
    "4909",  # ['IntelÂ® IrisÂ® Xe MAX 100 Graphics', 'Xe', 'DG1', '5.16*', '80']
    "4680",  # ['IntelÂ® UHD Graphics 770', 'Xe', 'Alder Lake-S', '5.16', '32']
    "4690",  # ['IntelÂ® UHD Graphics 770', 'Xe', 'Alder Lake-S', '5.16', '32']
    "4688",  # ['IntelÂ® UHD Graphics 770', 'Xe', 'Alder Lake-S', '5.16', '32']
    "468A",  # ['IntelÂ® UHD Graphics 770', 'Xe', 'Alder Lake-S', '5.16', '24']
    "468B",  # ['IntelÂ® UHD Graphics 770', 'Xe', 'Alder Lake-S', '6.1', '16']
    "4682",  # ['IntelÂ® UHD Graphics 730', 'Xe', 'Alder Lake-S', '5.16', '24']
    "4692",  # ['IntelÂ® UHD Graphics 730', 'Xe', 'Alder Lake-S', '5.16', '24']
    "4693",  # ['IntelÂ® UHD Graphics 710', 'Xe', 'Alder Lake-S', '5.16', '16']
    "46D3",  # ['IntelÂ® Graphics', 'Xe', 'Twin Lake', '6.9', '32']
    "46D4",  # ['IntelÂ® Graphics', 'Xe', 'Twin Lake', '6.9', '24']
    "46D0",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-N', '5.18', '32']
    "46D1",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-N', '5.18', '24']
    "46D2",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-N', '5.18', '16']
    "4626",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '96/80']
    "4628",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '96/80']
    "462A",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '96/80']
    "46A2",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '64']
    "46B3",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '64']
    "46C2",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '64']
    "46A3",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '64/48']
    "46B2",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '64/48']
    "46C3",  # ['IntelÂ® UHD Graphics', 'Xe', 'Alder Lake-P', '5.17', '64/48']
    "46A0",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '96']
    "46B0",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '96']
    "46C0",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '96']
    "46A6",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '96/80']
    "46AA",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '96/80']
    "46A8",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '96/80']
    "46A1",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '80']
    "46B1",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '80']
    "46C1",  # ['IntelÂ® IrisÂ® Xe Graphics', 'Xe', 'Alder Lake-P', '5.17', '80']
    "4C8A",  # ['IntelÂ® UHD Graphics 750', 'Xe', 'Rocket Lake', '5.13', '32']
    "4C8B",  # ['IntelÂ® UHD Graphics 730', 'Xe', 'Rocket Lake', '5.13', '24']
    "4C90",  # ['IntelÂ® UHD Graphics P750', 'Xe', 'Rocket Lake', '5.13', '24']
    "4C9A",  # ['IntelÂ® UHD Graphics P750', 'Xe', 'Rocket Lake', '5.13', '24']
    "4E71",  # ['IntelÂ® UHD Graphics', 'Xe', 'Jasper Lake', '5.15', '32']
    "4E61",  # ['IntelÂ® UHD Graphics', 'Xe', 'Jasper Lake', '5.15', '24']
    "4E57",  # ['IntelÂ® UHD Graphics', 'Xe', 'Jasper Lake', '5.15', '20']
    "4E55",  # ['IntelÂ® UHD Graphics', 'Xe', 'Jasper Lake', '5.15', '16']
    "4E51",  # ['IntelÂ® UHD Graphics', 'Xe', 'Jasper Lake', '5.15', '16']
    "4557",  # ['IntelÂ® UHD Graphics', 'Xe', 'Elkhart Lake', '5.15', '20']
    "4555",  # ['IntelÂ® UHD Graphics', 'Xe', 'Elkhart Lake', '5.15', '16']
    "4571",  # ['IntelÂ® UHD Graphics', 'Xe', 'Elkhart Lake', '5.15', '32']
    "4551",  # ['IntelÂ® UHD Graphics', 'Xe', 'Elkhart Lake', '5.15', '16']
    "4541",  # ['IntelÂ® UHD Graphics', 'Xe', 'Elkhart Lake', '5.15', '8']
    "9A59",  # ['IntelÂ® UHD Graphics', 'Xe', 'Tiger Lake', '5.7', '96']
    "9A78",  # ['IntelÂ® UHD Graphics', 'Xe', 'Tiger Lake', '5.7', '48']
    "9A60",  # ['IntelÂ® UHD Graphics', 'Xe', 'Tiger Lake', '5.7', '32']
    "9A70",  # ['IntelÂ® UHD Graphics', 'Xe', 'Tiger Lake', '5.7', '32']
    "9A68",  # ['IntelÂ® UHD Graphics', 'Xe', 'Tiger Lake', '5.7', '16']
    "9A40",  # ['Intel Iris Xe Graphics', 'Xe', 'Tiger Lake', '5.7', '96/80']
    "9A49",  # ['Intel Iris Xe Graphics', 'Xe', 'Tiger Lake', '5.7', '96/80']
    # Panther Lake (Xe3) - Updated Feb 2026
    "B080",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B081",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B082",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B083",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B084",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B085",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B086",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B087",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B08F",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-H', '6.14', 'TBD']
    "B090",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake-U', '6.14', 'TBD']
    "B0A0",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake', '6.14', 'TBD']
    "B0B0",  # ['Intel Arc Graphics', 'Xe3', 'Panther Lake', '6.14', 'TBD']
    "FD80",  # ['Intel Graphics', 'Xe3', 'Wildcat Lake', '6.14', 'TBD']
    "FD81",  # ['Intel Graphics', 'Xe3', 'Wildcat Lake', '6.14', 'TBD']
    # Nova Lake (Xe3) - Updated Feb 2026
    "D740",  # ['Intel Graphics', 'Xe3', 'Nova Lake-S', '6.14', 'TBD']
    "D741",  # ['Intel Graphics', 'Xe3', 'Nova Lake-U', '6.14', 'TBD']
    "D742",  # ['Intel Graphics', 'Xe3', 'Nova Lake-U', '6.14', 'TBD']
    "D743",  # ['Intel Graphics', 'Xe3', 'Nova Lake-S', '6.14', 'TBD']
    "D744",  # ['Intel Graphics', 'Xe3', 'Nova Lake-S', '6.14', 'TBD']
    "D745",  # ['Intel Graphics', 'Xe3', 'Nova Lake-U', '6.14', 'TBD']
]

dGPU_Dev_IDs = [
    # Battlemage (Xe2) - Updated Feb 2026
    "E202",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage', '6.14', 'TBD']
    "E209",  # ['Intel Arc B580 Graphics', 'Xe2', 'Battlemage', '6.14', '320']
    "E20B",  # ['Intel Arc B580 Graphics', 'Xe2', 'Battlemage', '6.11*', '320']
    "E20C",  # ['Intel Arc B570 Graphics', 'Xe2', 'Battlemage', '6.11*', '288']
    "E20D",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage', '6.14', 'TBD']
    "E210",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage', '6.14', 'TBD']
    "E211",  # ['Intel Arc Pro B60 Graphics', 'Xe2', 'Battlemage', '6.14', '320']
    "E212",  # ['Intel Arc Pro B50 Graphics', 'Xe2', 'Battlemage', '6.14', '256']
    "E215",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage', '6.14', 'TBD']
    "E216",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage', '6.14', 'TBD']
    "E220",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage G31', '6.14', 'TBD']
    "E221",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage G31', '6.14', 'TBD']
    "E222",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage G31', '6.14', 'TBD']
    "E223",  # ['Intel Arc Graphics', 'Xe2', 'Battlemage G31', '6.14', 'TBD']
    # Alchemist (Xe-HPG) Mobile
    "5690",  # ['Intel Arc A770M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '512']
    "5691",  # ['Intel Arc A730M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '384']
    "5692",  # ['Intel Arc A550M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '256']
    "5693",  # ['Intel Arc A370M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '128']
    "5694",  # ['Intel Arc A350M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '96']
    "5695",  # ['Intel Arc Graphics', 'Xe-HPG', 'Alchemist', '6.2', 'TBD']
    "5696",  # ['Intel Arc A570M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '256']
    "5697",  # ['Intel Arc A530M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '192']
    # Alchemist (Xe-HPG) Desktop
    "56A0",  # ['Intel Arc A770 Graphics', 'Xe-HPG', 'Alchemist', '6.2', '512']
    "56A1",  # ['Intel Arc A750 Graphics', 'Xe-HPG', 'Alchemist', '6.2', '448']
    "56A2",  # ['Intel Arc A580 Graphics', 'Xe-HPG', 'Alchemist', '6.2', '384']
    "56A3",  # ['Intel Arc Graphics', 'Xe-HPG', 'Alchemist', '6.2', 'TBD']
    "56A4",  # ['Intel Arc Graphics', 'Xe-HPG', 'Alchemist', '6.2', 'TBD']
    "56A5",  # ['Intel Arc A380 Graphics', 'Xe-HPG', 'Alchemist', '6.2', '128']
    "56A6",  # ['Intel Arc A310 LP Graphics', 'Xe-HPG', 'Alchemist', '6.2', '96']
    "56AF",  # ['Intel Arc A760A Graphics', 'Xe-HPG', 'Alchemist', '6.2', 'TBD']
    # Alchemist (Xe-HPG) Pro
    "56B0",  # ['Intel Arc Pro A30M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '128']
    "56B1",  # ['Intel Arc Pro A40/A50 Graphics', 'Xe-HPG', 'Alchemist', '6.2', '128']
    "56B2",  # ['Intel Arc Pro A60M Graphics', 'Xe-HPG', 'Alchemist', '6.2', '256']
    "56B3",  # ['Intel Arc Pro A60 Graphics', 'Xe-HPG', 'Alchemist', '6.2', '256']
    # Alchemist (Xe-HPG) Embedded
    "56BA",  # ['Intel Arc A380E Graphics', 'Xe-HPG', 'Alchemist', '6.7', '128']
    "56BB",  # ['Intel Arc A310E Graphics', 'Xe-HPG', 'Alchemist', '6.7', '96']
    "56BC",  # ['Intel Arc A370E Graphics', 'Xe-HPG', 'Alchemist', '6.7', '128']
    "56BD",  # ['Intel Arc A350E Graphics', 'Xe-HPG', 'Alchemist', '6.7', '96']
    "56BE",  # ['Intel Arc A750E Graphics', 'Xe-HPG', 'Alchemist', '6.7', 'TBD']
    "56BF",  # ['Intel Arc A580E Graphics', 'Xe-HPG', 'Alchemist', '6.7', 'TBD']
    # Data Center GPU Flex Series
    "56C0",  # ['Intel Data Center GPU Flex 170', 'Xe-HPG', 'Alchemist', '6.2', '512']
    "56C1",  # ['Intel Data Center GPU Flex 140', 'Xe-HPG', 'Alchemist', '6.2', '128']
    "56C2",  # ['Intel Data Center GPU Flex 170V', 'Xe-HPG', 'Alchemist', '6.2', 'TBD']
]

Ref_Platform_Bcmk_Settings = {
    "1VDBOX": {
        "description": "Platforms with 1VD Box iGPU",
        "samples": "i5-12400 (16G Mem)",
    },
    "2VDBOX": {
        "description": "Platforms with 2VD Box iGPU",
        "samples": "i7-1360P (16G Mem)",
    },
    "DGPU": {
        "description": "dGPU",
        "samples": "Arc A380",
    },
    "NPU": {
        "description": "MTL NPU device",
        "samples": "MTL 165H (32G Mem)",
    },
    "XEON": {
        "description": "Xeon based platform",
        "samples": "Xeon(R) Gold 6430 (512G Mem)",
    },
    "MULTI_WITH_DGPU": {
        "description": "MUTLI device with dGPU",
        "samples": "i5-12400 (16G Mem) + Arc A380",
    },
    "MULTI_WITH_NPU": {
        "description": "MUTLI device with NPU",
        "samples": "MTL 165H (32G Mem)",
    },
}


def align_multi_device(input_str):
    order = ["CPU", "GPU.0", "GPU.1", "NPU"]
    substrings = {key: False for key in order}

    for key in substrings.keys():
        if key in input_str:
            substrings[key] = True

    output_str = ",".join([key for key in order if substrings[key]])
    return "MULTI:" + output_str


def get_ref_value(ref_platform, model, precision, batch_size, device):
    if device == "GPU":
        device = "GPU.0"
    if device.startswith("MULTI:"):
        device = align_multi_device(device)
    csv_file = "bcmk_ref.csv"
    ref_value = 0
    ref_freq = 0
    ref_power = 0
    with open(csv_file, "r") as c_file:
        csv_data = csv.DictReader(c_file)
        for row in csv_data:
            if (
                row["Model"] == model
                and row["Reference Platform"] == ref_platform
                and row["Precision"] == precision
                and row["Batch Size"] == batch_size
                and device[:3].startswith(row["Device"][:3])
            ):
                ref_value = row["Reference Value"]
                ref_freq = row["Reference Freq"]
                ref_power = row["Reference Power"]
                if not ref_value:
                    ref_value = 0
                if not ref_freq:
                    ref_freq = 0
                if not ref_power:
                    ref_power = 0
                break
        else:
            logging.warning(
                f"Can not find the Ref Value with args: {ref_platform} {model} {precision} {batch_size} {device} "
            )

    return ref_value, ref_freq, ref_power


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
    except Exception as e:
        logging.error(f"Error reading /proc/cpuinfo: {e}")
        return False


def get_device_ids_and_drm_info():
    result = {}
    # if is_intel_xeon():
    #    return result

    lsgpu_result = subprocess.run(["lsgpu", "-n"], stdout=subprocess.PIPE, text=True)

    output = lsgpu_result.stdout
    if not output:
        return result

    pattern = re.compile(r"\b([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\b.*?(drm:/dev/dri/[^\s]+)")
    matches_info = pattern.findall(output)
    for device_id, drm_info in matches_info:
        result[drm_info] = device_id

    # if not result:
    match = re.search(r"(\bIntel.*?)\s+(\bdrm:/dev/dri/card0\b)", output)
    if match:
        intel_name, drm_path = match.groups()
        result[intel_name] = drm_path

    if not result:
        logging.warning(f"Can not parse gpu info from output of lsgpu: {output}")
    return result


def count_gpu_unit_count(drm_path, unit_name="VCS", duration=3):
    command = f"lsgpu -p  -d {drm_path} "
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(duration)
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    stdout, stderr = process.communicate()
    output = stdout.decode("utf-8")
    lines = output.split("\n")
    vcs_count = 0
    for line in lines:
        if "DEVPATH" in line:
            path = line.split(" :")[-1].strip()
            vcs1 = "/sys" + path + "/engine/vcs1"
            vcs0 = "/sys" + path + "/engine/vcs0"
            vcs_count = 0
            if Path(vcs1).exists():
                vcs_count = 2
            elif Path(vcs0).exists():
                vcs_count = 1

    return vcs_count


def get_device_id_by_gpu_index(index):
    lsgpu_result = subprocess.run(["lsgpu", "-n"], stdout=subprocess.PIPE, text=True)
    output = lsgpu_result.stdout

    render = f"renderD{128 + index}"

    pattern = re.compile(r"card(\d+)\s+\w+:(\w+)\s+drm:/dev/dri/card(\d+)\n└─(renderD\d+)\s+drm:/dev/dri/renderD\d+")
    matches = pattern.findall(output)

    render_to_info = {render: (f"drm:/dev/dri/card{card}", device_id) for card, device_id, _, render in matches}

    if render in render_to_info:
        return render_to_info[render]
    else:
        return None


def get_ref_platform_and_value(my_gpu_info, model, precision, batch_size, device):
    contain_dgpu = False
    is_multi_device = device.startswith("MULTI:")
    is_gpu_device = device.startswith("GPU")
    gpu_index = device.split(".")[-1]
    ref_platform_info = None
    if is_intel_xeon() and device == "CPU":
        ref_platform_info = Ref_Platform_Bcmk_Settings["XEON"]
    elif os.path.exists("/dev/accel"):
        ref_platform_info = Ref_Platform_Bcmk_Settings["NPU"]
    elif is_multi_device:
        if "NPU" in device:
            ref_platform_info = Ref_Platform_Bcmk_Settings["MULTI_WITH_NPU"]
        elif "GPU.1" in device:
            ref_platform_info = Ref_Platform_Bcmk_Settings["MULTI_WITH_DGPU"]
    elif is_gpu_device:
        try:
            gpu_index = int(gpu_index)
        except Exception:
            print(f"Invalid gpu index: {gpu_index}")
            return "none", 0, 0, 0

        result = get_device_id_by_gpu_index(gpu_index)
        if result:
            if result[1].upper() in dGPU_Dev_IDs:
                ref_platform_info = Ref_Platform_Bcmk_Settings["DGPU"]
            else:
                vdbox_count = count_gpu_unit_count(result[0])
                if vdbox_count == 1:
                    ref_platform_info = Ref_Platform_Bcmk_Settings["1VDBOX"]
                else:
                    ref_platform_info = Ref_Platform_Bcmk_Settings["2VDBOX"]

    if ref_platform_info is None:
        return "none", 0, 0, 0

    ref_plat = ref_platform_info["samples"]

    ref_value, ref_freq, ref_power = get_ref_value(ref_plat, model, precision, batch_size, device)

    return ref_plat, ref_value, ref_freq, ref_power


if __name__ == "__main__":
    my_gpu_info = get_device_ids_and_drm_info()
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "FP16", "auto", "CPU"))
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "INT8", "auto", "GPU.0"))
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "INT8", "auto", "GPU.1"))
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "INT8", "auto", "GPU.2"))
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "INT8", "auto", "MULTI:CPU,GPU.0,GPU.1"))
