# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Read supplemental GPU memory utilization through the xpu-smi CLI."""

from __future__ import annotations

import csv
import logging
import re
import shutil
import subprocess  # nosec B404 # Fixed xpu-smi arguments; no shell or user input.

logger = logging.getLogger(__name__)


class XpuSmiUtils:
    """Unprivileged wrapper for the single xpu-smi metric ESQ supplements."""

    @staticmethod
    def memory_util_map() -> dict[str, float]:
        if shutil.which("xpu-smi") is None:
            return {}

        try:
            proc = subprocess.run(  # nosec B603 # Fixed executable and arguments; shell=False.
                [
                    "xpu-smi",
                    "--query-gpu=pci.bus_id,utilization.memory",
                    "--format=csv,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Unable to read GPU memory utilization from xpu-smi: %s", exc)
            return {}

        if proc.returncode != 0:
            logger.warning("xpu-smi memory-utilization query failed: %s", proc.stderr.strip())
            return {}

        memory_util: dict[str, float] = {}
        for columns in csv.reader(proc.stdout.splitlines(), skipinitialspace=True):
            if len(columns) < 2:
                continue
            bdf = columns[0].strip().lower()
            if ":" not in bdf or "." not in bdf:
                continue
            value = XpuSmiUtils._parse_numeric(columns[-1])
            if bdf and value is not None:
                memory_util[bdf] = max(0.0, min(100.0, value))
        return memory_util

    @staticmethod
    def _parse_numeric(cell: str) -> float | None:
        if not cell or cell.upper() == "N/A":
            return None
        match = re.search(r"[-+]?\d*\.?\d+", cell)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None
