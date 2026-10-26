# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
OpenVINO device health/throughput check built on the official ``benchmark_app``.

Compiles a tiny model and reports throughput (FPS) for one device. Callers run
the command from ``build_benchmark_command`` in a subprocess with a timeout, so
a device that hangs in compile/inference (e.g. incompatible BIOS, firmware or
driver) is killed instead of hanging the CLI.

``benchmark_app`` is used directly (``-hint throughput``) because a hand-rolled
Python async loop measured 1.2x-5x lower FPS than it on the same model, with a
different device ranking, due to per-request Python overhead.

This is a health check, not a capability score. The model is a single ReLU
with almost no compute, so results are dominated by per-request overhead and
host<->device transfer. A discrete GPU pays that over PCIe and can read lower
than the CPU or NPU, while real AI models amortize it and reverse the ranking.
Use the AI profiles (real models) to compare devices.
"""

import re
import sys
from pathlib import Path

# Short since this runs once per detected device in 'esq run' prerun checks.
BENCHMARK_DURATION_S = 1

# benchmark_app ships inside the openvino package; launching it through the current
# interpreter avoids depending on the console script being on PATH.
_BENCHMARK_APP_LAUNCHER = "import sys; from openvino.tools.benchmark.main import main; sys.exit(main())"

_THROUGHPUT_PATTERN = re.compile(r"Throughput:\s+([\d.]+)\s+FPS")


def create_benchmark_model(directory: str) -> str:
    """Save the tiny dummy model as OpenVINO IR in ``directory`` and return the .xml path."""
    import numpy as np
    import openvino as ov

    input_node = ov.opset10.parameter([1, 3, 224, 224], dtype=np.float32, name="input")
    model = ov.Model([ov.opset10.relu(input_node)], [input_node], "prerun_smoke_model")
    model_path = str(Path(directory) / "smoke.xml")
    ov.save_model(model, model_path)
    return model_path


def build_benchmark_command(model_path: str, device: str, duration_s: int = BENCHMARK_DURATION_S) -> list[str]:
    """Command line that runs ``benchmark_app`` on ``device`` with the throughput hint."""
    return [
        sys.executable,
        "-c",
        _BENCHMARK_APP_LAUNCHER,
        "-m",
        model_path,
        "-d",
        device,
        "-hint",
        "throughput",
        "-t",
        str(duration_s),
    ]


def parse_throughput_fps(output: str) -> float | None:
    """Extract the FPS value from ``benchmark_app`` output, or None if absent."""
    match = _THROUGHPUT_PATTERN.search(output or "")
    return float(match.group(1)) if match else None
