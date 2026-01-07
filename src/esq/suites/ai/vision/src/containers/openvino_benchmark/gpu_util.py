# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import re
import subprocess  # nosec B404


def _get_device_type(device_id):
    first_char = device_id[0].upper()
    if first_char in ["4", "9", "A", "a"]:
        return "iGPU"
    elif first_char == "5":
        return "dGPU"
    elif first_char == "7":
        return "iGPU_MTL"
    else:
        return "iGPU"  ## maybe incorrect


def parse_gpu_types():
    try:
        # Use shell pipeline without shell=True by using two subprocess calls
        lspci_proc = subprocess.Popen(["lspci", "-nn"], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
        grep_proc = subprocess.Popen(
            ["grep", "-Ei", "DISPLAY|VGA"], stdin=lspci_proc.stdout, stdout=subprocess.PIPE, text=True
        )
        lspci_proc.stdout.close()  # Allow lspci_proc to receive SIGPIPE if grep_proc exits
        lspci_output, _ = grep_proc.communicate()
        if grep_proc.returncode not in (0, 1):  # grep returns 1 when no match found
            raise subprocess.CalledProcessError(grep_proc.returncode, "grep")
    except subprocess.CalledProcessError as e:
        print(f"Error executing lspci command: {e}")
        return
    except FileNotFoundError as e:
        print(f"Command not found: {e}")
        return

    device_pattern = re.compile(r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]")

    matches = device_pattern.findall(lspci_output)

    devices = []
    for match in matches:
        vendor_id, device_id = match
        device_type = _get_device_type(device_id)
        devices.append({"device_id": device_id, "device_type": device_type})

    # json_output = json.dumps(devices, indent=4)
    # print(json_output)
    return devices


def parse_igt():
    command_output = subprocess.check_output(["intel_gpu_top", "-L"], text=True)
    # command_output = """
    # card1                    Intel Meteorlake (Gen12)          pci:vendor=8086,device=7D55,card=0
    # └─renderD128
    #
    # card0                    Intel Dg2 (Gen12)                 pci:vendor=8086,device=56A5,card=0
    # └─renderD129
    # """
    card_pattern = re.compile(
        r"^(card\d+)\s+(.*?)\s+pci:vendor=(\w+),device=(\w+),card=(\d+)\n└─(renderD\d+)", re.MULTILINE
    )

    matches = card_pattern.findall(command_output)

    cards = []
    for match in matches:
        card_id, device_name, vendor, device, card, render_name = match
        cards.append(
            {
                "card_id": card_id,
                "pci_info": f"pci:vendor={vendor},device={device},card={card}",
                "render_name": render_name,
            }
        )

    # print(json.dumps(cards, indent=4))

    return cards


def _parse_xs_discovery():
    try:
        xpu_smi_output = subprocess.check_output(["xpu-smi", "discovery", "-j"], text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing xpu-smi command: {e}")
        return None

    _data = {}
    try:
        _data = json.loads(xpu_smi_output)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    # print(json.dumps(_data, indent=4))
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

    # print(device_dict)
    return device_dict


def main():
    print("Get GPU Devices: ")
    gpu_devices = get_gpu_devices()

    print("\nGet GPU Types:")
    parse_gpu_types()


if __name__ == "__main__":
    main()
