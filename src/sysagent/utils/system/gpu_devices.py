# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Intel GPU Device List Configuration

This module defines a comprehensive list of Intel GPU devices with their specifications.

Device Dictionary Keys:
- pci_id: PCI device identifier (string)
- name: Human-readable device name (string)
- architecture: GPU architecture (string)
- codename: Internal codename (string)
- kernel: Minimum supported Linux kernel version (string)
- eu: Execution unit count (string)
"""

import re
from typing import Any, Dict, List, Optional

GPU_DEVICE_LIST = [
    # ========================================
    # Xe2 Architecture (Latest Generation)
    # ========================================
    # Battlemage Series
    {
        "pci_id": "E20B",
        "name": "Intel® Arc™ B580 Graphics",
        "architecture": "Xe2",
        "codename": "Battlemage",
        "kernel": "6.11*",
        "eu": "320",
    },
    {
        "pci_id": "E20C",
        "name": "Intel® Arc™ B570 Graphics",
        "architecture": "Xe2",
        "codename": "Battlemage",
        "kernel": "6.11*",
        "eu": "288",
    },
    {
        "pci_id": "E212",
        "name": "Intel® Arc™ Graphics",
        "architecture": "Xe2",
        "codename": "Battlemage-G31",
        "kernel": "6.11*",
        "eu": "256",
    },
    # Lunar Lake Series
    {
        "pci_id": "6420",
        "name": "Intel® Graphics",
        "architecture": "Xe2",
        "codename": "Lunar Lake",
        "kernel": "6.10*",
        "eu": "64",
    },
    {
        "pci_id": "6422",
        "name": "Intel® Graphics",
        "architecture": "Xe2",
        "codename": "Lunar Lake",
        "kernel": "6.10*",
        "eu": "64",
    },
    # Arrow Lake Series
    {
        "pci_id": "7D51",
        "name": "Intel® Graphics",
        "architecture": "Xe-LPG",
        "codename": "Arrow Lake-H",
        "kernel": "6.9",
        "eu": "128/112",
    },
    {
        "pci_id": "7D67",
        "name": "Intel® Graphics",
        "architecture": "Xe-LPG",
        "codename": "Arrow Lake-S",
        "kernel": "6.7",
        "eu": "64/48/32",
    },
    {
        "pci_id": "7D41",
        "name": "Intel® Graphics",
        "architecture": "Xe-LPG",
        "codename": "Arrow Lake-U",
        "kernel": "6.9",
        "eu": "64",
    },
    # Meteor Lake Series
    {
        "pci_id": "7DD5",
        "name": "Intel® Graphics",
        "architecture": "Xe-LPG",
        "codename": "Meteor Lake-G",
        "kernel": "6.6",
        "eu": "128",
    },
    {
        "pci_id": "7D55",
        "name": "Intel® Graphics",
        "architecture": "Xe-LPG",
        "codename": "Meteor Lake-H",
        "kernel": "6.6",
        "eu": "128",
    },
    {
        "pci_id": "7D60",
        "name": "Intel® Graphics",
        "architecture": "Xe-LPG",
        "codename": "Meteor Lake-S",
        "kernel": "6.6",
        "eu": "64",
    },
    {
        "pci_id": "7D45",
        "name": "Intel® Graphics",
        "architecture": "Xe-LPG",
        "codename": "Meteor Lake-U",
        "kernel": "6.6",
        "eu": "64",
    },
    {
        "pci_id": "7D40",
        "name": "Intel® Graphics",
        "architecture": "Xe-LPG",
        "codename": "Meteor Lake-P",
        "kernel": "6.6",
        "eu": "96/64",
    },
    # ========================================
    # Xe-HPG Architecture (Arc Series)
    # ========================================
    # Alchemist Series (DG2)
    {
        "pci_id": "56A0",
        "name": "Intel® Arc™ A770 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "512",
    },
    {
        "pci_id": "56A1",
        "name": "Intel® Arc™ A750 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "448",
    },
    {
        "pci_id": "56A2",
        "name": "Intel® Arc™ A580 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "384",
    },
    {
        "pci_id": "56A3",
        "name": "Intel® Arc™ A380 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "128",
    },
    {
        "pci_id": "56A4",
        "name": "Intel® Arc™ A310 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "96",
    },
    {
        "pci_id": "56A5",
        "name": "Intel® Arc™ Pro A60 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "192",
    },
    {
        "pci_id": "56A6",
        "name": "Intel® Arc™ Pro A60M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "192",
    },
    {
        "pci_id": "5690",
        "name": "Intel® Arc™ A370M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "128",
    },
    {
        "pci_id": "5691",
        "name": "Intel® Arc™ A350M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "96",
    },
    {
        "pci_id": "5692",
        "name": "Intel® Arc™ A550M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "192",
    },
    {
        "pci_id": "5693",
        "name": "Intel® Arc™ A370M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "128",
    },
    {
        "pci_id": "5694",
        "name": "Intel® Arc™ A350M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "96",
    },
    {
        "pci_id": "5695",
        "name": "Intel® Arc™ A730M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "384",
    },
    {
        "pci_id": "5696",
        "name": "Intel® Arc™ A550M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "192",
    },
    {
        "pci_id": "5697",
        "name": "Intel® Arc™ A770M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "512",
    },
    {
        "pci_id": "56B0",
        "name": "Intel® Arc™ Pro A40 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "128",
    },
    {
        "pci_id": "56B1",
        "name": "Intel® Arc™ Pro A50 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "192",
    },
    {
        "pci_id": "56B2",
        "name": "Intel® Arc™ Pro A30M Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "128",
    },
    {
        "pci_id": "56B3",
        "name": "Intel® Arc™ Pro A40 Graphics",
        "architecture": "Xe-HPG",
        "codename": "Alchemist",
        "kernel": "6.2",
        "eu": "128",
    },
    # ========================================
    # Xe-LP Architecture (12th Gen and newer)
    # ========================================
    # Raptor Lake-S Refresh Series
    {
        "pci_id": "A7A8",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-S Refresh",
        "kernel": "6.7",
        "eu": "32",
    },
    {
        "pci_id": "A7A9",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-S Refresh",
        "kernel": "6.7",
        "eu": "16",
    },
    {
        "pci_id": "A7AA",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-S Refresh",
        "kernel": "6.7",
        "eu": "16",
    },
    {
        "pci_id": "A7AB",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-S Refresh",
        "kernel": "6.7",
        "eu": "16",
    },
    # Raptor Lake-S Series
    {
        "pci_id": "A788",
        "name": "Intel® UHD Graphics 770",
        "architecture": "Xe",
        "codename": "Raptor Lake-S",
        "kernel": "6.7",
        "eu": "32",
    },
    {
        "pci_id": "A789",
        "name": "Intel® UHD Graphics 750",
        "architecture": "Xe",
        "codename": "Raptor Lake-S",
        "kernel": "6.7",
        "eu": "32",
    },
    {
        "pci_id": "A78A",
        "name": "Intel® UHD Graphics 730",
        "architecture": "Xe",
        "codename": "Raptor Lake-S",
        "kernel": "6.7",
        "eu": "24",
    },
    {
        "pci_id": "A78B",
        "name": "Intel® UHD Graphics 710",
        "architecture": "Xe",
        "codename": "Raptor Lake-S",
        "kernel": "6.7",
        "eu": "16",
    },
    # Raptor Lake-P Series
    {
        "pci_id": "A720",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-P",
        "kernel": "6.7",
        "eu": "96",
    },
    {
        "pci_id": "A721",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-P",
        "kernel": "6.7",
        "eu": "80",
    },
    {
        "pci_id": "A7A0",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-P",
        "kernel": "6.7",
        "eu": "96",
    },
    {
        "pci_id": "A7A1",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-P",
        "kernel": "6.7",
        "eu": "80",
    },
    {
        "pci_id": "A7A8",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-P",
        "kernel": "6.7",
        "eu": "48",
    },
    # Raptor Lake-U Series
    {
        "pci_id": "A7AC",
        "name": "Intel® Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-U",
        "kernel": "6.7",
        "eu": "96/80",
    },
    {
        "pci_id": "A7AD",
        "name": "Intel® Graphics",
        "architecture": "Xe",
        "codename": "Raptor Lake-U",
        "kernel": "6.7",
        "eu": "64/48",
    },
    # DG1 Series (Discrete)
    {
        "pci_id": "4905",
        "name": "Intel® Iris® Xe MAX Graphics",
        "architecture": "Xe",
        "codename": "DG1",
        "kernel": "5.16*",
        "eu": "96",
    },
    {
        "pci_id": "4907",
        "name": "Intel® Server GPU",
        "architecture": "Xe",
        "codename": "DG1",
        "kernel": "5.16*",
        "eu": "80",
    },
    {
        "pci_id": "4908",
        "name": "Intel® Iris® Xe MAX Graphics",
        "architecture": "Xe",
        "codename": "DG1",
        "kernel": "5.16*",
        "eu": "96",
    },
    {
        "pci_id": "4909",
        "name": "Intel® Iris® Xe MAX Graphics",
        "architecture": "Xe",
        "codename": "DG1",
        "kernel": "5.16*",
        "eu": "96",
    },
    # Alder Lake-S Series
    {
        "pci_id": "4680",
        "name": "Intel® UHD Graphics 770",
        "architecture": "Xe",
        "codename": "Alder Lake-S",
        "kernel": "5.17",
        "eu": "32",
    },
    {
        "pci_id": "4682",
        "name": "Intel® UHD Graphics 730",
        "architecture": "Xe",
        "codename": "Alder Lake-S",
        "kernel": "5.17",
        "eu": "24",
    },
    {
        "pci_id": "4690",
        "name": "Intel® UHD Graphics 770",
        "architecture": "Xe",
        "codename": "Alder Lake-S",
        "kernel": "5.17",
        "eu": "32",
    },
    {
        "pci_id": "4692",
        "name": "Intel® UHD Graphics 730",
        "architecture": "Xe",
        "codename": "Alder Lake-S",
        "kernel": "5.17",
        "eu": "24",
    },
    {
        "pci_id": "4693",
        "name": "Intel® UHD Graphics 710",
        "architecture": "Xe",
        "codename": "Alder Lake-S",
        "kernel": "5.17",
        "eu": "16",
    },
    # Alder Lake-P Series
    {
        "pci_id": "46A8",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "64/48",
    },
    {
        "pci_id": "46B2",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "64/48",
    },
    {
        "pci_id": "46C3",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "64/48",
    },
    {
        "pci_id": "46A0",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96",
    },
    {
        "pci_id": "46B0",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96",
    },
    {
        "pci_id": "46C0",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96",
    },
    {
        "pci_id": "46A6",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96/80",
    },
    {
        "pci_id": "46AA",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96/80",
    },
    {
        "pci_id": "46B1",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96/80",
    },
    {
        "pci_id": "46C1",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96/80",
    },
    {
        "pci_id": "46A1",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "80",
    },
    {
        "pci_id": "46A3",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "80",
    },
    {
        "pci_id": "4626",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96",
    },
    {
        "pci_id": "4628",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "80",
    },
    {
        "pci_id": "462A",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "80",
    },
    {
        "pci_id": "4636",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "96",
    },
    {
        "pci_id": "4638",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "80",
    },
    {
        "pci_id": "463A",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-P",
        "kernel": "5.17",
        "eu": "80",
    },
    # Alder Lake-M Series
    {
        "pci_id": "46C2",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-M",
        "kernel": "5.17",
        "eu": "48",
    },
    # Alder Lake-N Series
    {
        "pci_id": "46D0",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-N",
        "kernel": "6.0",
        "eu": "32",
    },
    {
        "pci_id": "46D1",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-N",
        "kernel": "6.0",
        "eu": "24",
    },
    {
        "pci_id": "46D2",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Alder Lake-N",
        "kernel": "6.0",
        "eu": "16",
    },
    # Tiger Lake Series
    {
        "pci_id": "9A49",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-LP",
        "kernel": "5.9",
        "eu": "96",
    },
    {
        "pci_id": "9A40",
        "name": "Intel® Iris® Xe Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-LP",
        "kernel": "5.9",
        "eu": "96",
    },
    {
        "pci_id": "9A60",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-LP",
        "kernel": "5.9",
        "eu": "32",
    },
    {
        "pci_id": "9A68",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-LP",
        "kernel": "5.9",
        "eu": "48",
    },
    {
        "pci_id": "9A70",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-LP",
        "kernel": "5.9",
        "eu": "12",
    },
    {
        "pci_id": "9A78",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-LP",
        "kernel": "5.9",
        "eu": "16",
    },
    # Tiger Lake-H Series
    {
        "pci_id": "9A01",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-H",
        "kernel": "5.13",
        "eu": "32",
    },
    {
        "pci_id": "9A02",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-H",
        "kernel": "5.13",
        "eu": "16",
    },
    {
        "pci_id": "9A09",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-H",
        "kernel": "5.13",
        "eu": "16",
    },
    {
        "pci_id": "9A0A",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-H",
        "kernel": "5.13",
        "eu": "16",
    },
    {
        "pci_id": "9A0B",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-H",
        "kernel": "5.13",
        "eu": "12",
    },
    {
        "pci_id": "9A0C",
        "name": "Intel® UHD Graphics",
        "architecture": "Xe",
        "codename": "Tiger Lake-H",
        "kernel": "5.13",
        "eu": "8",
    },
    # ========================================
    # Gen12 Architecture (Tiger Lake / Rocket Lake)
    # ========================================
    # Rocket Lake Series
    {
        "pci_id": "4C80",
        "name": "Intel® UHD Graphics 750",
        "architecture": "Gen12",
        "codename": "Rocket Lake",
        "kernel": "5.11",
        "eu": "32",
    },
    {
        "pci_id": "4C8A",
        "name": "Intel® UHD Graphics 730",
        "architecture": "Gen12",
        "codename": "Rocket Lake",
        "kernel": "5.11",
        "eu": "24",
    },
    {
        "pci_id": "4C8B",
        "name": "Intel® UHD Graphics 710",
        "architecture": "Gen12",
        "codename": "Rocket Lake",
        "kernel": "5.11",
        "eu": "16",
    },
    {
        "pci_id": "4C8C",
        "name": "Intel® UHD Graphics P750",
        "architecture": "Gen12",
        "codename": "Rocket Lake",
        "kernel": "5.11",
        "eu": "32",
    },
    {
        "pci_id": "4C90",
        "name": "Intel® UHD Graphics 750",
        "architecture": "Gen12",
        "codename": "Rocket Lake",
        "kernel": "5.11",
        "eu": "32",
    },
    {
        "pci_id": "4C9A",
        "name": "Intel® UHD Graphics 730",
        "architecture": "Gen12",
        "codename": "Rocket Lake",
        "kernel": "5.11",
        "eu": "24",
    },
    # ========================================
    # Gen11 Architecture (Ice Lake)
    # ========================================
    # Ice Lake-LP Series
    {
        "pci_id": "8A50",
        "name": "Intel® Iris® Plus Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "64",
    },
    {
        "pci_id": "8A51",
        "name": "Intel® Iris® Plus Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "48",
    },
    {
        "pci_id": "8A52",
        "name": "Intel® Iris® Plus Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "48",
    },
    {
        "pci_id": "8A53",
        "name": "Intel® Iris® Plus Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "48",
    },
    {
        "pci_id": "8A56",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "32",
    },
    {
        "pci_id": "8A57",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "32",
    },
    {
        "pci_id": "8A58",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "32",
    },
    {
        "pci_id": "8A59",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "32",
    },
    {
        "pci_id": "8A5A",
        "name": "Intel® Iris® Plus Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "48",
    },
    {
        "pci_id": "8A5B",
        "name": "Intel® Iris® Plus Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "48",
    },
    {
        "pci_id": "8A5C",
        "name": "Intel® Iris® Plus Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "48",
    },
    {
        "pci_id": "8A5D",
        "name": "Intel® Iris® Plus Graphics",
        "architecture": "Gen11",
        "codename": "Ice Lake-LP",
        "kernel": "5.2",
        "eu": "64",
    },
    # Elkhart Lake Series
    {
        "pci_id": "4551",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Elkhart Lake",
        "kernel": "5.8",
        "eu": "32",
    },
    {
        "pci_id": "4541",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Elkhart Lake",
        "kernel": "5.8",
        "eu": "32",
    },
    {
        "pci_id": "4E51",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Jasper Lake",
        "kernel": "5.8",
        "eu": "32",
    },
    {
        "pci_id": "4E61",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Jasper Lake",
        "kernel": "5.8",
        "eu": "24",
    },
    {
        "pci_id": "4E71",
        "name": "Intel® UHD Graphics",
        "architecture": "Gen11",
        "codename": "Jasper Lake",
        "kernel": "5.8",
        "eu": "16",
    },
    # ========================================
    # Gen9.5 Architecture (Coffee Lake / Whiskey Lake)
    # ========================================
    # Coffee Lake Series
    {
        "pci_id": "3E90",
        "name": "Intel® UHD Graphics 610",
        "architecture": "Gen9.5",
        "codename": "Coffee Lake",
        "kernel": "4.20",
        "eu": "12",
    },
    {
        "pci_id": "3E91",
        "name": "Intel® UHD Graphics 630",
        "architecture": "Gen9.5",
        "codename": "Coffee Lake",
        "kernel": "4.20",
        "eu": "24",
    },
    {
        "pci_id": "3E92",
        "name": "Intel® UHD Graphics 630",
        "architecture": "Gen9.5",
        "codename": "Coffee Lake",
        "kernel": "4.20",
        "eu": "24",
    },
    {
        "pci_id": "3E93",
        "name": "Intel® UHD Graphics 610",
        "architecture": "Gen9.5",
        "codename": "Coffee Lake",
        "kernel": "4.20",
        "eu": "12",
    },
    {
        "pci_id": "3E94",
        "name": "Intel® UHD Graphics P630",
        "architecture": "Gen9.5",
        "codename": "Coffee Lake",
        "kernel": "4.20",
        "eu": "24",
    },
    {
        "pci_id": "3E96",
        "name": "Intel® UHD Graphics P630",
        "architecture": "Gen9.5",
        "codename": "Coffee Lake",
        "kernel": "4.20",
        "eu": "24",
    },
    {
        "pci_id": "3E98",
        "name": "Intel® UHD Graphics 630",
        "architecture": "Gen9.5",
        "codename": "Coffee Lake",
        "kernel": "4.20",
        "eu": "24",
    },
    {
        "pci_id": "3E9B",
        "name": "Intel® UHD Graphics 630",
        "architecture": "Gen9.5",
        "codename": "Coffee Lake",
        "kernel": "4.20",
        "eu": "24",
    },
    {
        "pci_id": "3EA0",
        "name": "Intel® UHD Graphics 620",
        "architecture": "Gen9.5",
        "codename": "Whiskey Lake",
        "kernel": "4.20",
        "eu": "24",
    },
    # ========================================
    # Gen9 Architecture (Skylake / Kaby Lake)
    # ========================================
    # Skylake Series
    {
        "pci_id": "1906",
        "name": "Intel® HD Graphics 510",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "12",
    },
    {
        "pci_id": "1912",
        "name": "Intel® HD Graphics 530",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "24",
    },
    {
        "pci_id": "1916",
        "name": "Intel® HD Graphics 520",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "24",
    },
    {
        "pci_id": "191B",
        "name": "Intel® HD Graphics 530",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "24",
    },
    {
        "pci_id": "191D",
        "name": "Intel® HD Graphics P530",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "24",
    },
    {
        "pci_id": "191E",
        "name": "Intel® HD Graphics 515",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "24",
    },
    {
        "pci_id": "1921",
        "name": "Intel® HD Graphics 520",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "24",
    },
    {
        "pci_id": "1923",
        "name": "Intel® HD Graphics 535",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "24",
    },
    {
        "pci_id": "1926",
        "name": "Intel® Iris® Graphics 540",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "48",
    },
    {
        "pci_id": "1927",
        "name": "Intel® Iris® Graphics 550",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "48",
    },
    {
        "pci_id": "192B",
        "name": "Intel® Iris® Graphics 555",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "48",
    },
    {
        "pci_id": "192D",
        "name": "Intel® Iris® Graphics P555",
        "architecture": "Gen9",
        "codename": "Skylake",
        "kernel": "4.6",
        "eu": "48",
    },
    # Kaby Lake Series
    {
        "pci_id": "5906",
        "name": "Intel® HD Graphics 610",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "12",
    },
    {
        "pci_id": "5912",
        "name": "Intel® HD Graphics 630",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "24",
    },
    {
        "pci_id": "5916",
        "name": "Intel® HD Graphics 620",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "24",
    },
    {
        "pci_id": "591A",
        "name": "Intel® HD Graphics P630",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "24",
    },
    {
        "pci_id": "591B",
        "name": "Intel® HD Graphics 630",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "24",
    },
    {
        "pci_id": "591D",
        "name": "Intel® HD Graphics P630",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "24",
    },
    {
        "pci_id": "591E",
        "name": "Intel® HD Graphics 615",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "24",
    },
    {
        "pci_id": "5921",
        "name": "Intel® HD Graphics 620",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "24",
    },
    {
        "pci_id": "5923",
        "name": "Intel® HD Graphics 635",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "24",
    },
    {
        "pci_id": "5926",
        "name": "Intel® Iris® Plus Graphics 640",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "48",
    },
    {
        "pci_id": "5927",
        "name": "Intel® Iris® Plus Graphics 650",
        "architecture": "Gen9.5",
        "codename": "Kaby Lake",
        "kernel": "4.9",
        "eu": "48",
    },
]


def _normalize_pci_id(pci_id: str) -> str:
    """Normalize PCI ID for consistent lookup."""
    if not pci_id:
        return ""
    return pci_id.upper().replace("0X", "")


def get_device_by_pci_id(pci_id: str) -> Optional[Dict[str, Any]]:
    """
    Get device information by PCI ID.

    Args:
        pci_id: PCI device ID (case insensitive, with or without 0x prefix)

    Returns:
        Device information dictionary or None if not found
    """
    if not pci_id:
        return None

    normalized_id = _normalize_pci_id(pci_id)

    for device in GPU_DEVICE_LIST:
        if device["pci_id"] == normalized_id:
            return device

    return None


def get_device_by_name(
    name: str, case_sensitive: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Get device information by name.

    Args:
        name: Device name to search for
        case_sensitive: Whether to perform case-sensitive matching

    Returns:
        Device information dictionary or None if not found
    """
    if not name:
        return None

    search_name = name if case_sensitive else name.lower()

    for device in GPU_DEVICE_LIST:
        device_name = device["name"] if case_sensitive else device["name"].lower()
        if search_name in device_name:
            return device

    return None


def get_devices_by_architecture(architecture: str) -> List[Dict[str, Any]]:
    """
    Get all devices for a specific architecture.

    Args:
        architecture: GPU architecture (e.g., "Xe", "Gen12", "Gen11")

    Returns:
        List of device information dictionaries
    """
    if not architecture:
        return []

    return [
        device
        for device in GPU_DEVICE_LIST
        if device["architecture"].lower() == architecture.lower()
    ]


def get_devices_by_codename(codename: str) -> List[Dict[str, Any]]:
    """
    Get all devices for a specific codename.

    Args:
        codename: Device codename (e.g., "Tiger Lake", "Alder Lake")

    Returns:
        List of device information dictionaries
    """
    if not codename:
        return []

    return [
        device
        for device in GPU_DEVICE_LIST
        if codename.lower() in device["codename"].lower()
    ]


def search_devices(query: str) -> List[Dict[str, Any]]:
    """
    Search devices by any field containing the query string.

    Args:
        query: Search query string

    Returns:
        List of matching device information dictionaries
    """
    if not query:
        return GPU_DEVICE_LIST

    query_lower = query.lower()
    results = []

    for device in GPU_DEVICE_LIST:
        # Search in all string fields
        searchable_text = " ".join(
            [
                device.get("pci_id", ""),
                device.get("name", ""),
                device.get("architecture", ""),
                device.get("codename", ""),
                device.get("kernel", ""),
                device.get("eu", ""),
            ]
        ).lower()

        if query_lower in searchable_text:
            results.append(device)

    return results


def get_supported_architectures() -> List[str]:
    """
    Get list of all supported GPU architectures.

    Returns:
        Sorted list of unique architecture names
    """
    architectures = set(device["architecture"] for device in GPU_DEVICE_LIST)
    return sorted(architectures)


def get_supported_codenames() -> List[str]:
    """
    Get list of all supported GPU codenames.

    Returns:
        Sorted list of unique codename values
    """
    codenames = set(device["codename"] for device in GPU_DEVICE_LIST)
    return sorted(codenames)


def validate_device_info(device: Dict[str, Any]) -> bool:
    """
    Validate that a device dictionary has all required fields.

    Args:
        device: Device information dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["pci_id", "name", "architecture", "codename", "kernel", "eu"]

    if not isinstance(device, dict):
        return False

    for field in required_fields:
        if field not in device or not isinstance(device[field], str):
            return False

    # Validate PCI ID format (4 hex characters)
    pci_id = device["pci_id"]
    if not re.match(r"^[0-9A-Fa-f]{4}$", pci_id):
        return False

    return True


def get_device_count() -> int:
    """
    Get total number of devices in the database.

    Returns:
        Number of devices
    """
    return len(GPU_DEVICE_LIST)


def get_device_stats() -> Dict[str, int]:
    """
    Get statistics about the device database.

    Returns:
        Dictionary with architecture and codename counts
    """
    architectures = {}
    codenames = {}

    for device in GPU_DEVICE_LIST:
        arch = device["architecture"]
        codename = device["codename"]

        architectures[arch] = architectures.get(arch, 0) + 1
        codenames[codename] = codenames.get(codename, 0) + 1

    return {
        "total_devices": len(GPU_DEVICE_LIST),
        "architectures": dict(sorted(architectures.items())),
        "codenames": dict(sorted(codenames.items())),
    }
