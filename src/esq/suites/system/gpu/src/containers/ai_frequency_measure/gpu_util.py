# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess   # nosec B404
import re
import json

def _get_device_type(device_id):
    first_char = device_id[0].upper()
    if first_char in ['4', '9', 'A', 'a']:
        return "iGPU"
    elif first_char == '5':
        return "dGPU"
    elif first_char == '7':
        return "iGPU_MTL"
    else:
        return "iGPU" ## maybe incorrect

def parse_gpu_types():

    try:
        # Run lspci safely without shell
        lspci_out = subprocess.check_output(["lspci", "-nn"], text=True)

        # Apply your grep equivalent in Python
        lspci_output = "\n".join(
            line for line in lspci_out.splitlines()
            if re.search(r"DISPLAY|VGA", line, re.IGNORECASE)
        )

    except subprocess.CalledProcessError as e:
        print(f"Error executing lspci command: {e}")
        return

    device_pattern = re.compile(r'\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]')

    matches = device_pattern.findall(lspci_output)

    devices = []
    for match in matches:
        vendor_id, device_id = match
        device_type = _get_device_type(device_id)
        devices.append({
            "device_id": device_id,
            "device_type": device_type
        })

    return devices

def parse_igt():
    command_output = subprocess.check_output(['intel_gpu_top', '-L'], text=True)
    # command_output = """
    # card1                    Intel Meteorlake (Gen12)          pci:vendor=8086,device=7D55,card=0
    # └─renderD128
    # 
    # card0                    Intel Dg2 (Gen12)                 pci:vendor=8086,device=56A5,card=0
    # └─renderD129
    # """
    card_pattern = re.compile(
        r"^(card\d+)\s+(.*?)\s+pci:vendor=(\w+),device=(\w+),card=(\d+)\n└─(renderD\d+)",
        re.MULTILINE
    )
    
    matches = card_pattern.findall(command_output)
    
    cards = []
    for match in matches:
        card_id, device_name, vendor, device, card, render_name = match
        cards.append({
            "card_id": card_id,
            "pci_info": f"pci:vendor={vendor},device={device},card={card}",
            "render_name": render_name
        })
        
    return cards

def _parse_xs_discovery():
    try:
        xpu_smi_output = subprocess.check_output(['xpu-smi', 'discovery', '-j'], text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing xpu-smi command: {e}")
        return None

    _data = {}
    try:
        _data = json.loads(xpu_smi_output)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    return _data

def get_gpu_devices():
    xsd_data = _parse_xs_discovery()
    igt_data = parse_igt()
    
    drm_to_card_id = {f"/dev/dri/{info['card_id']}": info for info in igt_data}
    
    device_dict = {}

    if xsd_data["device_list"] is None:
        return device_dict

    for device in xsd_data["device_list"]:
        drm_device = device["drm_device"]
        if drm_device in drm_to_card_id:
            additional_info = drm_to_card_id[drm_device]
            device["card_id"] = additional_info["card_id"]
            device["pci_info"] = additional_info["pci_info"]

            pci_device_id = device["pci_device_id"]
            stripped_device_id = pci_device_id[2:]
            device["render_name"] = additional_info["render_name"]
            device_type = _get_device_type(stripped_device_id)
            device["device_type"] = device_type
            
            bcmk_device = f"GPU.{device['device_id']}"
            device_dict[bcmk_device] = device
          
    return device_dict

def main():
    print("Get GPU Devices: ")
    gpu_devices = get_gpu_devices()
    
    print("\nGet GPU Types:")
    parse_gpu_types()

if __name__ == "__main__":
    main()
