# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import csv
import subprocess   #nosec B404
import re
import time
import signal
import shlex
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

iGPU_Dev_IDs = ["7DD5", "7D40", "7D45", "7D55", "7D60", "A7A9", "A7A8", "A7A1", "A7A0", "A721", "A720", "A78B", "A78A", "A789", "A788", "A783", "A782", "A781", "A780", "4907", "4905", "4680", "4682", "4688", "468A", "468B", "4690", "4692", "4693", "46D0", "46D1", "46D2", "4626", "4628", "462A", "46A0", "46A1", "46A2", "46A3", "46A6", "46A8", "46AA", "46B0", "46B1", "46B2", "46B3", "46C0", "46C1", "46C2", "46C3", "4C8A", "4C8B", "4C90", "4C9A", "4C8C", "4C80", "4E71", "4E61", "4E57", "4E55", "4E51", "4571", "4557", "4555", "4551", "4541", "9A59", "9A60", "9A68", "9A70", "9A40", "9A49", "9A78", "9AC0", "9AC9", "9AD9", "9AF8"]
dGPU_Dev_IDs = ["56B3", "56B2", "56A4", "56A3", "5697", "5696", "5695", "56B1", "56B0", "56A6", "56A5", "56A1", "56A0", "5694", "5693", "5692", "5691", "5690"]

Ref_Platform_Bcmk_Settings = {
    '1VDBOX': {
        'description': "Platforms with 1VD Box iGPU",
        'samples': "i5-12400 (16G Mem)",
        },
    '2VDBOX':{
        'description': "Platforms with 2VD Box iGPU",
        'samples': "i7-1360P (16G Mem)",
        },
    'DGPU':{
        'description': "dGPU",
        'samples': "Arc A380",
        },
    'NPU':{
        'description': "MTL NPU device",
        'samples': "MTL 165H (32G Mem)",
        },
    'XEON':{
        'description': "Xeon based platform",
        'samples': "Xeon(R) Gold 6430 (512G Mem)",
    },
    'MULTI_WITH_DGPU':{
        'description': "MUTLI device with dGPU",
        'samples': "i5-12400 (16G Mem) + Arc A380",
        },
    'MULTI_WITH_NPU':{
        'description': "MUTLI device with NPU",
        'samples': "MTL 165H (32G Mem)",
    }

}

def align_multi_device(input_str):
    order = ['CPU', 'GPU.0', 'GPU.1', 'NPU']
    substrings = {key: False for key in order}

    for key in substrings.keys():
        if key in input_str:
            substrings[key] = True

    output_str = ','.join([key for key in order if substrings[key]])
    return 'MULTI:' + output_str

def get_ref_value(ref_platform, model, precision, batch_size, device):
    if device == 'GPU':
        device = 'GPU.0'
    if device.startswith("MULTI:"):
        device = align_multi_device(device)
    csv_file = "bcmk_ref.csv"
    ref_value = 0
    ref_freq = 0
    with open(csv_file, 'r') as c_file:
        csv_data = csv.DictReader(c_file)
        for row in csv_data:
            if (row['Model'] == model and
                row['Reference Platform'] == ref_platform and
                row['Precision'] == precision and
                row['Batch Size'] == batch_size and
                device[:3].startswith(row['Device'][:3])) :
                
                ref_value = row["Reference Value"]
                ref_freq = row["Reference Freq"]
                if not ref_value:
                    ref_value = 0
                if not ref_freq:
                    ref_freq = 0
                break
        else:
            logging.warning(f"Can not find the Ref Value with args: {ref_platform} {model} {precision} {batch_size} {device} ")

    return ref_value,ref_freq

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
        logging.error(f"Error reading /proc/cpuinfo: {e}")
        return False

def get_device_ids_and_drm_info():
    result = {}
    #if is_intel_xeon():
    #    return result

    lsgpu_result = subprocess.run(['lsgpu', '-n'], stdout=subprocess.PIPE, text=True)

    output = lsgpu_result.stdout
    if not output:
        return result

    pattern = re.compile(r'\b([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\b.*?(drm:/dev/dri/[^\s]+)')
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

def count_gpu_unit_count(drm_path, unit_name='VCS', duration=3):
    command = f"lsgpu -p  -d {drm_path} "
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(duration)
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    stdout, stderr = process.communicate()
    output = stdout.decode('utf-8')
    lines = output.split("\n")
    vcs_count = 0
    for line in lines:
        if "DEVPATH" in line:
            path = line.split(" :")[-1].strip()
            vcs1 = "/sys" + path + "/engine/vcs1"
            vcs0 = "/sys" +  path + "/engine/vcs0"
            vcs_count = 0
            if Path(vcs1).exists():
                vcs_count = 2
            elif Path(vcs0).exists():
                vcs_count = 1

    return vcs_count

def get_device_id_by_gpu_index(index):

    lsgpu_result = subprocess.run(['lsgpu', '-n'], stdout=subprocess.PIPE, text=True)
    output = lsgpu_result.stdout

    render = f"renderD{128 + index}"

    pattern = re.compile(r'card(\d+)\s+\w+:(\w+)\s+drm:/dev/dri/card(\d+)\n└─(renderD\d+)\s+drm:/dev/dri/renderD\d+')
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
        ref_platform_info = Ref_Platform_Bcmk_Settings['XEON']
    elif os.path.exists('/dev/accel'):
        ref_platform_info = Ref_Platform_Bcmk_Settings['NPU']
    elif is_multi_device:
        if "NPU" in device:
            ref_platform_info = Ref_Platform_Bcmk_Settings['MULTI_WITH_NPU']
        elif "GPU.1" in device:
            ref_platform_info = Ref_Platform_Bcmk_Settings['MULTI_WITH_DGPU']
    elif is_gpu_device:
        try:
            gpu_index = int(gpu_index)
        except:
             print("wrong input")
             return "none", 0, 0

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
        return "none", 0, 0

    ref_plat = ref_platform_info['samples']

    ref_value,ref_freq = get_ref_value(ref_plat, model, precision, batch_size, device)

    return ref_plat, ref_value, ref_freq

if __name__ == '__main__':
    my_gpu_info = get_device_ids_and_drm_info()
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "FP16", "auto", "CPU"))
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "INT8", "auto", "GPU.0"))
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "INT8", "auto", "GPU.1"))
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "INT8", "auto", "GPU.2"))
    print(get_ref_platform_and_value(my_gpu_info, "resnet-50-tf", "INT8", "auto", "MULTI:CPU,GPU.0,GPU.1"))
