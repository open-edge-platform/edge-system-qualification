# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLM Prompt execution: run the load generator and aggregate its records."""

import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from sysagent.utils.core.venv import VenvManager

logger = logging.getLogger(__name__)


def run_client(
    manager: VenvManager,
    venv_name: str,
    client_script: str,
    results_dir: str,
    media_dir: str,
    client_config: dict[str, Any],
    timeout: int,
) -> list[dict[str, Any]]:
    """
    Run the load generator in its isolated venv against the OVMS server and return its per-request records.

    The server port is published on 127.0.0.1, so the client reaches it directly.
    """
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "vlm_requests.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    config = dict(client_config)
    config["media_dir"] = media_dir
    config["output_file"] = output_path
    config_path = os.path.join(results_dir, "client_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    python_bin = manager.get_python_executable(venv_name)
    if not python_bin:
        raise RuntimeError(f"VLM client venv not found: {venv_name}")

    start = time.time()
    result = manager.run_command_in_venv(
        venv_name=venv_name,
        command=[python_bin, client_script, config_path],
        timeout=timeout,
        check=False,
        stream_output=True,
    )
    if result.timed_out:
        raise TimeoutError(f"VLM client timed out after {timeout}s")
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"VLM client failed with exit code {result.returncode} after {time.time() - start:.1f}s "
            f"({client_config.get('workers')} worker(s)): {output[-500:] or 'no output'}"
        )
    logger.info(f"VLM client finished in {time.time() - start:.1f}s")

    if client_config.get("warmup_only"):
        return []
    if not os.path.exists(output_path):
        raise RuntimeError(f"VLM client produced no output: {output_path}")
    return read_records(output_path)


def read_records(path: str) -> list[dict[str, Any]]:
    """Read the per-request records (one JSON object per line) written by the load generator."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-request records into the standard LLM serving quantities.

    - ``avg_latency``: mean request latency, the full duration of a request (``request_latency`` of
      GenAI-Perf and AIPerf, ``gen_ai.client.inference.duration`` of OpenTelemetry).
    - ``output_token_throughput``: total generated tokens divided by the measurement window, from the start
      of the first measured request to the end of the last one (``output_token_throughput`` of GenAI-Perf
      and AIPerf). It includes the time the load generator spends between requests.
    - ``avg_throughput_per_request``: mean of the generated tokens per second of each request. This is the
      ``throughput_mean_sec`` of the Order Accuracy applications, kept for comparing with them. There is no
      standard name for it.
    - ``avg_prompt_tokens`` and ``avg_completion_tokens``: input and output sequence length.
    """
    measured = [r for r in records if r.get("ok") and not r.get("warmup")]
    failed = [r for r in records if not r.get("ok")]
    if not measured:
        return {"calls": 0, "failed": len(failed), "errors": [r.get("error") for r in failed[:3]]}

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    latencies = [r["latency_sec"] for r in measured]
    completion = [r["completion_tokens"] for r in measured]
    prompt = [r["prompt_tokens"] for r in measured]
    if all("start_time" in r for r in measured):
        window = max(r["start_time"] + r["latency_sec"] for r in measured) - min(r["start_time"] for r in measured)
    else:
        window = sum(latencies)
    return {
        "calls": len(measured),
        "failed": len(failed),
        "errors": [r.get("error") for r in failed[:3]],
        "avg_latency": mean(latencies),
        "window_sec": window,
        "output_token_throughput": sum(completion) / window if window > 0 else 0.0,
        "avg_throughput_per_request": mean([c / lat for c, lat in zip(completion, latencies) if lat > 0] or [0.0]),
        "avg_completion_tokens": mean(completion),
        "avg_prompt_tokens": mean(prompt),
        "sample_output": measured[0].get("text", ""),
    }


def aggregate_ttft(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Time to first token of the streaming probe: the first request after the server start is the cold start,
    the others are the warm starts (their mean is reported).
    """
    cold = [r["ttft_sec"] for r in records if r.get("ok") and r.get("cold")]
    warm = [r["ttft_sec"] for r in records if r.get("ok") and not r.get("cold")]
    return {
        "cold": cold[0] if cold else None,
        "warm": sum(warm) / len(warm) if warm else None,
        "warm_requests": len(warm),
        "failed": len([r for r in records if not r.get("ok")]),
        "errors": [r.get("error") for r in records if not r.get("ok")][:3],
    }


def latency_metric_ms(latencies_sec: list[float], metric: str) -> float:
    """Latency in milliseconds for the configured metric (avg, p95 or max); 0 when there is no data."""
    if not latencies_sec:
        return 0.0
    ordered = sorted(latencies_sec)
    if metric == "max":
        value = ordered[-1]
    elif metric == "p95":
        value = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
    else:
        value = sum(ordered) / len(ordered)
    return value * 1000


def run_stream_density(
    run_iteration: Callable[[int], list[dict[str, Any]]],
    target_latency_ms: float,
    latency_metric: str,
    density_increment: int,
    min_requests: int,
    max_workers: int,
    search: str = "adaptive",
) -> dict[str, Any]:
    """
    Find the highest number of concurrent workers whose latency stays within the target.

    Step ``n`` runs ``1 + (n - 1) * density_increment`` workers through ``run_iteration(workers)``, which
    returns the request records. The source applications test 1, 2, 3, ... steps until the latency
    exceeds the target or their limit of 50 steps is reached. Both ``search`` modes give that result for a
    latency that grows with the number of workers:

    - ``adaptive`` (default): the latency-guided search of ``test_dlstreamer``
      (``fps_guided_adaptive_search``). It starts at one worker, jumps by the headroom to the target, then
      refines with a binary search, so far fewer steps are run.
    - ``linear``: one step at a time like the source applications.

    The search never runs more than ``max_workers`` workers. When the largest allowed step still passes,
    ``streams_max`` is that limit and not a measured maximum, and ``limit_reached`` is set.

    Returns:
        Dict with ``streams_max``, ``limit_reached``, the per-step summaries and the records of the best step.
        When no step passes, ``best_records`` is empty and ``reference_records`` holds the records of the
        smallest step that was run, so the standard metrics can still be reported for the failed test.
    """
    # Highest step whose worker count is within max_workers
    last_step = max(1, (max_workers - 1) // density_increment + 1)
    iterations: list[dict[str, Any]] = []
    results: dict[int, dict[str, Any]] = {}

    def workers_for(step: int) -> int:
        return 1 + (step - 1) * density_increment

    def evaluate(step: int) -> tuple[float, bool]:
        """Run a step. Returns the latency headroom (target / latency, 0 without data) and whether it passed."""
        workers = workers_for(step)
        records = run_iteration(workers)
        ok = [r for r in records if r.get("ok")]
        latencies = [r["latency_sec"] for r in ok]
        latency_ms = latency_metric_ms(latencies, latency_metric)
        passed = 0 < latency_ms <= target_latency_ms
        iterations.append(
            {
                "iteration": len(iterations) + 1,
                "workers": workers,
                "requests": len(records),
                "successful": len(ok),
                "avg_latency_ms": round(latency_metric_ms(latencies, "avg"), 1),
                "p95_latency_ms": round(latency_metric_ms(latencies, "p95"), 1),
                "max_latency_ms": round(latency_metric_ms(latencies, "max"), 1),
                "passed": passed,
            }
        )
        results[step] = {"workers": workers, "records": records, "passed": passed}
        if len(records) < min_requests:
            logger.warning(f"Step {step}: {len(records)} requests, fewer than min_requests={min_requests}")
        logger.info(
            f"Density iteration {len(iterations)}: {workers} worker(s), {latency_metric} latency "
            f"{latency_ms / 1000:.2f}s, target {target_latency_ms / 1000:.2f}s, {'passed' if passed else 'failed'}"
        )
        headroom = target_latency_ms / latency_ms if latency_ms > 0 else 0.0
        return headroom, bool(ok)

    best_step = 0
    if search == "linear":
        for step in range(1, last_step + 1):
            _, has_data = evaluate(step)
            if not has_data or not results[step]["passed"]:
                break
            best_step = step
    else:
        from esq.utils.media.adaptive_search import AdaptiveSearchConfig, fps_guided_adaptive_search

        def run_step(step: int) -> tuple[float, int]:
            # The search compares the first value with a target of 1.0, so the latency headroom
            # (target / latency) passes exactly when the latency is within the target
            headroom, has_data = evaluate(step)
            return headroom, 0 if has_data else 1

        best_step, search_info = fps_guided_adaptive_search(
            run_pipeline_func=run_step,
            max_streams=last_step,
            target_fps=1.0,
            initial_count=1,
            config=AdaptiveSearchConfig(min_start=1, enable_confirmation=False),
            logger=logger,
            value_label="x latency headroom",
        )
        logger.info(f"Adaptive search ran {search_info.iterations} steps for {best_step} passing step(s)")

    best = results.get(best_step)
    limit_reached = False
    if best is not None:
        limit_reached = best_step == last_step and best["passed"]
        if limit_reached:
            logger.warning(
                f"Worker limit reached: {best['workers']} workers are still within the target, so the real "
                f"maximum is higher. Raise max_workers to search further."
            )
    return {
        "streams_max": best["workers"] if best else 0,
        "limit_reached": limit_reached,
        "iterations": iterations,
        "best_records": best["records"] if best else [],
        "reference_records": best["records"] if best else results[min(results)]["records"] if results else [],
        "reference_workers": best["workers"] if best else workers_for(min(results)) if results else 0,
    }
