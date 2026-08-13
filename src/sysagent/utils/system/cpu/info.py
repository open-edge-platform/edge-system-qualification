# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CPU hardware information collection.

Part of the ``sysagent.utils.system.cpu`` sub-package.

Collects CPU brand, architecture, frequency, feature flags, topology, and
RDT capabilities into a unified dictionary suitable for system reporting.
"""

import logging
import os
import platform
from typing import Any

import cpuinfo
import psutil

from .cpufreq import collect_cpufreq_info
from .cpuidle import collect_cpuidle_info
from .generation import detect_cpu_generation_and_segment
from .rdt import collect_cache_partition_info
from .topology import collect_cpu_core_types

logger = logging.getLogger(__name__)


def _get_cpu_socket_count() -> int | None:
    """Extract the number of physical CPU sockets from /proc/cpuinfo.

    Returns the count of unique ``physical id`` entries, or ``None`` when the
    field is absent (common on single-socket systems).
    """
    try:
        cpuinfo_path = "/proc/cpuinfo"
        if not os.path.exists(cpuinfo_path):
            logger.debug("'/proc/cpuinfo' not found, cannot determine socket count")
            return None

        physical_ids: set = set()
        with open(cpuinfo_path) as f:
            for line in f:
                if line.startswith("physical id"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            physical_ids.add(parts[1].strip())
                        except (ValueError, IndexError) as exc:
                            logger.debug(f"Failed to parse physical id: {line.strip()} — {exc}")

        if not physical_ids:
            logger.debug("No 'physical id' entries found in /proc/cpuinfo, cannot determine socket count")
            return None

        socket_count = len(physical_ids)
        logger.debug(f"Detected {socket_count} CPU socket(s) from /proc/cpuinfo")
        return socket_count

    except Exception as exc:
        logger.warning(f"Failed to read CPU socket count from /proc/cpuinfo: {exc}")
        return None


def _detect_cpu_features(cpu_info_data: dict[str, Any]) -> dict[str, Any]:
    """Detect advanced CPU features from raw cpuinfo flags.

    Returns a dict with boolean fields:
    ``virtualization``, ``hyper_threading``, ``turbo_boost``,
    ``aes_ni``, ``avx``, ``avx2``, ``avx512``.
    """
    features = {
        "virtualization": False,
        "hyper_threading": False,
        "turbo_boost": False,
        "aes_ni": False,
        "avx": False,
        "avx2": False,
        "avx512": False,
    }

    flags = cpu_info_data.get("flags", [])
    if isinstance(flags, list):
        flags_lower = [f.lower() for f in flags]
        features["virtualization"] = any(f in flags_lower for f in ["vmx", "svm"])
        features["aes_ni"] = "aes" in flags_lower
        features["avx"] = "avx" in flags_lower
        features["avx2"] = "avx2" in flags_lower
        features["avx512"] = any("avx512" in f for f in flags_lower)

    logical_cores = cpu_info_data.get("count", psutil.cpu_count(logical=True))
    physical_cores = psutil.cpu_count(logical=False)
    features["hyper_threading"] = logical_cores > physical_cores

    return features


def collect_cpu_info(openvino_cpu=None) -> dict[str, Any]:
    """Collect CPU information including cores, architecture, and OpenVINO capabilities.

    Parameters
    ----------
    openvino_cpu:
        List of OpenVINO CPU device dicts (from ``collect_openvino_devices``).

    Returns
    -------
    dict
        Comprehensive CPU dict with keys: ``brand``, ``architecture``, ``bits``,
        ``count``, ``logical_count``, ``sockets``, ``frequency``, ``flags``,
        ``vendor_id``, ``family``, ``model``, ``stepping``, ``cache_size``,
        ``microcode``, ``openvino``, feature booleans, ``generation_info``,
        ``core_types`` (with ``groups`` list, each group carrying per-type ``rdt``),
        ``rdt`` (global hardware-level RDT capabilities).
    """
    try:
        cpu_info_data = cpuinfo.get_cpu_info()
        socket_count = _get_cpu_socket_count()

        cpu_info: dict[str, Any] = {
            "brand": cpu_info_data.get("brand_raw", "Unknown"),
            "architecture": cpu_info_data.get("arch", platform.machine()),
            "bits": cpu_info_data.get("bits", 64),
            "count": psutil.cpu_count(logical=False),
            "logical_count": psutil.cpu_count(logical=True),
            "sockets": socket_count,
            "frequency": {
                "current": round(psutil.cpu_freq().current, 2) if psutil.cpu_freq() else None,
                "min": round(psutil.cpu_freq().min, 2) if psutil.cpu_freq() else None,
                "max": round(psutil.cpu_freq().max, 2) if psutil.cpu_freq() else None,
            },
            "flags": cpu_info_data.get("flags", []),
            "vendor_id": cpu_info_data.get("vendor_id_raw", "Unknown"),
            "family": cpu_info_data.get("family", 0),
            "model": cpu_info_data.get("model", 0),
            "stepping": cpu_info_data.get("stepping", 0),
            "cache_size": cpu_info_data.get("l3_cache_size", "Unknown"),
            "microcode": cpu_info_data.get("microcode", "Unknown"),
        }

        # OpenVINO CPU device properties
        if openvino_cpu:
            for ov_cpu in openvino_cpu:
                if ov_cpu:
                    quick_access = ov_cpu.get("quick_access", {})
                    device_info = ov_cpu.get("device", {})
                    all_props = device_info.get("all_properties", {})

                    cpu_openvino = {
                        "device_name": quick_access.get("device_name", ov_cpu.get("device_name", "CPU")),
                        "full_device_name": quick_access.get(
                            "full_device_name", ov_cpu.get("full_device_name", "Unknown")
                        ),
                        "device_type": quick_access.get("device_type", ov_cpu.get("device_type", "CPU")),
                        "capabilities": quick_access.get("capabilities", ov_cpu.get("capabilities", [])),
                        "vendor": quick_access.get("vendor", ov_cpu.get("vendor", "Unknown")),
                    }
                    if "CPU_THREADS_NUM" in all_props:
                        cpu_openvino["threads"] = all_props["CPU_THREADS_NUM"]
                    if "PERFORMANCE_HINT_NUM_REQUESTS" in all_props:
                        cpu_openvino["performance_hint_requests"] = all_props["PERFORMANCE_HINT_NUM_REQUESTS"]
                    if "INFERENCE_NUM_THREADS" in all_props:
                        cpu_openvino["inference_threads"] = all_props["INFERENCE_NUM_THREADS"]

                    cpu_info["openvino"] = cpu_openvino
                    break

        # Extended feature flags (AVX, AES-NI, HT, etc.)
        cpu_info.update(_detect_cpu_features(cpu_info_data))

        # Intel generation and segment detection
        if "Intel" in cpu_info.get("vendor_id", ""):
            try:
                cpu_info["generation_info"] = detect_cpu_generation_and_segment(
                    family=cpu_info.get("family", 0),
                    model=cpu_info.get("model", 0),
                    stepping=cpu_info.get("stepping", 0),
                    brand=cpu_info.get("brand", ""),
                    core_count=cpu_info.get("count", 0),
                )
                logger.debug(f"Detected Intel CPU: {cpu_info['generation_info']}")
            except Exception as exc:
                logger.warning(f"Failed to detect CPU generation/segment: {exc}")
                cpu_info["generation_info"] = {
                    "codename": "Unknown",
                    "series": "unknown",
                    "generation": "Unknown",
                    "segment": "unknown",
                    "is_supported": False,
                }

        # Hybrid P-core / E-core / LP E-core topology
        cpu_info["core_types"] = collect_cpu_core_types()

        # Intel RDT — current cache partition configuration via rdmsr
        cpu_info["rdt"] = collect_cache_partition_info()

        # CPU idle states (C-states) — per-cpu per-state data from cpuidle sysfs
        cpu_info["cpuidle"] = collect_cpuidle_info()

        # CPU frequency scaling — per-cpu governor, scaling max/min freq, and
        # global EPB/EPP policy from the cpufreq sysfs tree
        cpu_info["cpufreq"] = collect_cpufreq_info()

        return cpu_info

    except Exception as exc:
        logger.warning(f"Failed to collect CPU info: {exc}")
        return {"error": str(exc)}
