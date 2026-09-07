# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch device resolution utilities for ESQ test suites."""

import logging

logger = logging.getLogger(__name__)

# Maps ESQ device category to the PyTorch device string.
# ``None`` means the category is not supported by this backend.
_PT_DEVICE_MAP: dict[str, str | None] = {
    "cpu": "cpu",
    "igpu": "xpu",
    "dgpu": "xpu",
    "npu": None,
}


def resolve_torch_devices(devices: list) -> list[tuple[str, str]]:
    """Resolve ESQ device categories to (torch_device, ov_device_id) pairs.

    Uses OpenVINO to cross-reference GPU device indices so that dGPU categories
    map to indexed ``xpu:N`` strings matching PyTorch's XPU enumeration order.
    The ``ov_device_id`` (e.g. ``"GPU.1"``) is returned alongside so callers can
    derive consistent metric names via ``get_metric_name_for_device``.

    On systems where discrete GPUs are present, PyTorch XPU enumerates them as
    ``xpu:0``, ``xpu:1``, ... in the same order as OpenVINO's discrete GPU
    indices. The iGPU is not enumerated as an XPU device on such systems.

    Args:
        devices: ESQ device category list, e.g. ``["dgpu"]``.

    Returns:
        List of ``(torch_device_str, ov_device_id)`` tuples, e.g.:
        - ``[("cpu", "CPU")]``
        - ``[("xpu:0", "GPU")]`` for a single iGPU system
        - ``[("xpu:0", "GPU.1"), ("xpu:1", "GPU.2")]`` for two dGPUs
    """
    import pytest

    dev_cats = [str(d).lower() for d in devices]

    if "cpu" in dev_cats:
        return [("cpu", "CPU")]

    if all(d not in ("igpu", "dgpu") for d in dev_cats):
        pytest.skip(
            f"PyTorch backend does not support device categories: {devices}. "
            f"Supported categories: cpu, igpu (xpu), dgpu (xpu)."
        )

    try:
        from sysagent.utils.system.ov_helper import get_available_devices_by_category

        all_gpu_dict = get_available_devices_by_category(device_categories=["igpu", "dgpu"])

        dgpu_ids = sorted(
            [d for d, info in all_gpu_dict.items() if "discrete" in info["device_type"].lower()]
        )
        igpu_ids = sorted(
            [d for d, info in all_gpu_dict.items() if "integrated" in info["device_type"].lower()]
        )

        if "dgpu" in dev_cats:
            if not dgpu_ids:
                pytest.skip("PyTorch backend: no discrete GPU (dGPU) available.")
            # Track occurrence count per name so same-name dGPUs get distinct indices.
            name_seen: dict[str, int] = {}
            results = []
            for ov_id in dgpu_ids:
                ov_name = all_gpu_dict[ov_id].get("full_name", ov_id)
                key = ov_name.lower()
                occurrence = name_seen.get(key, 0)
                name_seen[key] = occurrence + 1
                results.append((f"xpu:match||{ov_name}||{occurrence}", ov_id))
            return results

        if "igpu" in dev_cats:
            if not igpu_ids:
                pytest.skip("PyTorch backend: no integrated GPU (iGPU) available.")
            ov_name = all_gpu_dict[igpu_ids[0]].get("full_name", igpu_ids[0])
            return [(f"xpu:match||{ov_name}||0", igpu_ids[0])]

    except Exception as exc:
        logger.warning(
            "Could not determine XPU index via OpenVINO (%s); falling back to 'xpu'.", exc
        )

    return [("xpu", "GPU")]


def resolve_torch_device(devices: list) -> str:
    """Resolve an ESQ device-category list to a single PyTorch device string.

    Wraps :func:`resolve_torch_devices` and returns the first device string.
    Use :func:`resolve_torch_devices` directly to support multiple dGPUs.
    """
    pairs = resolve_torch_devices(devices)
    return pairs[0][0] if pairs else "xpu"
