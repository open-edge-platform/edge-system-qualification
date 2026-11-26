# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Dependency-free CPU identification.

Reads ``/proc/cpuinfo`` on Linux and falls back to ``lscpu`` and sysfs cache
information when fields are unavailable.
``read_cpu_info()`` returns the same keys the former ``py-cpuinfo`` call
provided, omitting any value that cannot be determined.
"""

import logging
import os
import platform
import re
import sys
from typing import Any

from sysagent.utils.core.process import run_command

logger = logging.getLogger(__name__)

_PROC_CPUINFO = "/proc/cpuinfo"
_SYSFS_CPU0_CACHE = "/sys/devices/system/cpu/cpu0/cache"

_SIZE_MULTIPLIERS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}

_ARCH_MAP = {
    "x86_64": ("X86_64", 64),
    "amd64": ("X86_64", 64),
    "x64": ("X86_64", 64),
    "aarch64": ("ARM_8", 64),
    "arm64": ("ARM_8", 64),
}


def _parse_size_bytes(text: str) -> int | None:
    """Convert strings such as ``30720 KB``, ``30 MiB (1 instance)`` or ``30720K`` to bytes."""
    match = re.match(r"\s*(\d+(?:\.\d+)?)\s*([kmg])?i?b?", text.lower())
    if not match:
        return None
    return int(float(match.group(1)) * _SIZE_MULTIPLIERS[match.group(2) or ""])


def _to_int(text: str | None) -> int | None:
    try:
        return int(text) if text else None
    except ValueError:
        return None


def _parse_key_values(text: str) -> dict[str, str]:
    """Parse ``key : value`` lines (``/proc/cpuinfo`` and ``lscpu``); first occurrence wins."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, found, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if found and key and value:
            fields.setdefault(key, value)
    return fields


def _first(fields: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if fields.get(key):
            return fields[key]
    return None


def _extract(fields: dict[str, str]) -> dict[str, Any]:
    """Map raw ``/proc/cpuinfo`` or ``lscpu`` fields to the normalized info keys."""
    info: dict[str, Any] = {}

    for key in ("model name", "cpu", "hardware", "uarch"):
        value = fields.get(key)
        if value and not value.isdigit():
            info["brand_raw"] = value
            break

    vendor = _first(fields, "vendor_id", "vendor id", "vendor")
    if vendor:
        info["vendor_id_raw"] = vendor

    for info_key, field_key in (("family", "cpu family"), ("model", "model"), ("stepping", "stepping")):
        number = _to_int(fields.get(field_key))
        if number is not None:
            info[info_key] = number

    microcode = fields.get("microcode")
    if microcode:
        info["microcode"] = microcode

    flags = _first(fields, "flags", "features")
    if flags:
        info["flags"] = sorted(set(flags.split()))

    cache = _first(fields, "cache size", "l3 cache")
    cache_bytes = _parse_size_bytes(cache) if cache else None
    if cache_bytes:
        info["l3_cache_size"] = cache_bytes

    return info


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _run_text(command: list[str]) -> str | None:
    try:
        result = run_command(command, timeout=5)
        return result.stdout if result.returncode == 0 else None
    except Exception as exc:
        logger.debug(f"{command[0]} unavailable: {exc}")
        return None


def _l3_size_from_sysfs() -> int | None:
    """Read the L3 cache size of cpu0 from sysfs (covers architectures without ``cache size``)."""
    try:
        indexes = sorted(os.listdir(_SYSFS_CPU0_CACHE))
    except OSError:
        return None
    for index in indexes:
        base = f"{_SYSFS_CPU0_CACHE}/{index}"
        level = _read_text(f"{base}/level")
        size = _read_text(f"{base}/size")
        if level and level.strip() == "3" and size:
            return _parse_size_bytes(size)
    return None


def _detect_arch() -> tuple[str, int]:
    machine = platform.machine()
    lowered = machine.lower()
    if lowered in _ARCH_MAP:
        return _ARCH_MAP[lowered]
    return machine.upper() or "Unknown", 64 if sys.maxsize > 2**32 else 32


def read_cpu_info() -> dict[str, Any]:
    """Return CPU identification data.

    Keys (present only when determinable): ``brand_raw``, ``vendor_id_raw``, ``family``,
    ``model``, ``stepping``, ``microcode``, ``flags`` (sorted list), ``l3_cache_size``
    (bytes), plus always ``arch``, ``bits`` and ``count`` (logical CPUs).
    """
    info: dict[str, Any] = {}

    proc_text = _read_text(_PROC_CPUINFO)
    if proc_text:
        info.update(_extract(_parse_key_values(proc_text)))

    if "brand_raw" not in info or "flags" not in info:
        lscpu_text = _run_text(["lscpu"])
        if lscpu_text:
            for key, value in _extract(_parse_key_values(lscpu_text)).items():
                info.setdefault(key, value)

    if "l3_cache_size" not in info:
        l3_size = _l3_size_from_sysfs()
        if l3_size:
            info["l3_cache_size"] = l3_size

    info["arch"], info["bits"] = _detect_arch()
    info["count"] = os.cpu_count() or 1
    return info
