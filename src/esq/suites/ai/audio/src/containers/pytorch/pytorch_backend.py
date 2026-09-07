# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch backend for Whisper ASR benchmarking using openai-whisper."""

import logging
import os
import time

logger = logging.getLogger(__name__)


class _SkipTest(Exception):
    """Raised inside the container to signal that the test should be skipped."""

def _find_xpu_by_name(expected_hint: str) -> str:
    """Match an OV-reported device name (with occurrence index) to a concrete xpu:N."""
    import re
    import torch

    # Hint format: "<ov_name>||<occurrence>"  e.g. "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)||1"
    parts = expected_hint.split("||", 1)
    expected_name = parts[0]
    occurrence = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise _SkipTest(
            f"No XPU devices available inside the container. "
            f"Expected device '{expected_name}' was visible via OpenVINO on the host."
        )

    # OV appends type annotations like "(iGPU)" or "(dGPU)" that PyTorch omits.
    def _normalize(name: str) -> str:
        return re.sub(r"\s*\([id]GPU\)\s*$", "", name.strip(), flags=re.IGNORECASE).strip().lower()

    expected_norm = _normalize(expected_name)
    n = torch.xpu.device_count()
    match_count = 0
    for i in range(n):
        props = torch.xpu.get_device_properties(i)
        if _normalize(props.name) == expected_norm:
            if match_count == occurrence:
                logger.info("Matched OV device '%s' (occurrence %d) to xpu:%d", expected_name, occurrence, i)
                return f"xpu:{i}"
            match_count += 1

    available = [torch.xpu.get_device_properties(i).name for i in range(n)]
    raise _SkipTest(
        f"Device '{expected_name}' occurrence {occurrence} (seen by OpenVINO on host) was not found among "
        f"PyTorch XPU devices inside the container: {available}."
    )


def prepare_pytorch_model(model_size: str, download_root: str) -> None:
    """Pre-download Whisper model weights to *download_root* (no-op if already cached).

    Args:
        model_size: Whisper model size string, e.g. ``"base"``, ``"turbo"``.
        download_root: Directory where model weights will be stored.
    """
    import whisper as openai_whisper

    os.makedirs(download_root, exist_ok=True)
    logger.info("Pre-downloading whisper-%s PyTorch weights (if not cached)...", model_size)
    openai_whisper.load_model(model_size, device="cpu", download_root=download_root)
    logger.info("PyTorch model weights ready")


def run_pytorch_inference(
    model_size: str,
    torch_device: str,
    pytorch_model_dir: str,
    wav_file_path: str,
    warmup_runs: int,
    num_runs: int,
    reference_transcript: str,
) -> dict:
    """Run Whisper inference with the PyTorch (openai-whisper) backend.

    Args:
        model_size: Whisper model size string, e.g. ``"base"``, ``"turbo"``.
        torch_device: PyTorch device string: ``"cpu"`` or ``"xpu"``.
        pytorch_model_dir: Directory used as the download root for model weights.
        wav_file_path: Path to a 16 kHz mono WAV file.
        warmup_runs: Number of un-timed warm-up passes before measurement.
        num_runs: Number of timed inference passes to average.
        reference_transcript: Ground-truth text for WER/CER calculation.

    Returns:
        A dict with keys ``"metrics"``, ``"parameters"``, and ``"metadata"``.
    """
    import torch
    import whisper as openai_whisper

    if torch_device.startswith("xpu:match||"):
        torch_device = _find_xpu_by_name(torch_device.split("||", 1)[1])

    if torch_device.startswith("xpu") and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError(
            f"torch_device='{torch_device}' requested but torch.xpu is not available inside the container. "
        )

    try:
        from esq.suites.ai.audio.src.containers.utils.metrics_helper import char_error_rate, word_error_rate
    except ImportError:
        from metrics_helper import char_error_rate, word_error_rate  # container-local copy

    def _sync():
        """Synchronise device so perf_counter captures real kernel completion."""
        if torch_device.startswith("xpu"):
            torch.xpu.synchronize()

    logger.info("Loading whisper-%s on torch device '%s'...", model_size, torch_device)
    pt_model = openai_whisper.load_model(
        model_size, device=torch_device, download_root=pytorch_model_dir
    )

    # Load and preprocess audio once (outside the timed loop)
    audio_raw = openai_whisper.load_audio(wav_file_path)
    audio_duration_s = len(audio_raw) / 16000.0
    audio_padded = openai_whisper.pad_or_trim(audio_raw)

    decode_options = openai_whisper.DecodingOptions(language="en")

    def _infer():
        mel = openai_whisper.log_mel_spectrogram(
            audio_padded, n_mels=pt_model.dims.n_mels
        ).to(pt_model.device)
        return openai_whisper.decode(pt_model, mel, decode_options)

    # Warm-up passes
    if warmup_runs > 0:
        logger.info("Running %d warm-up pass(es)...", warmup_runs)
        for _ in range(warmup_runs):
            _infer()
            _sync()
        logger.info("Warm-up complete")

    # Timed inference passes
    logger.info("Running %d timed inference pass(es)...", num_runs)
    run_times = []
    transcription = ""
    for run_idx in range(num_runs):
        t0 = time.perf_counter()
        pt_result = _infer()
        _sync()
        run_times.append(time.perf_counter() - t0)
        transcription = pt_result.text.strip()
        logger.info(
            "  Run %d/%d: %.1f ms  '%s'",
            run_idx + 1,
            num_runs,
            run_times[-1] * 1000,
            transcription,
        )

    avg_s = sum(run_times) / len(run_times)
    latency_ms = round(avg_s * 1000, 2)
    rtf = round(avg_s / audio_duration_s, 4)
    wer = round(word_error_rate(reference_transcript, transcription), 2)
    cer = round(char_error_rate(reference_transcript, transcription), 2)

    logger.info("Transcription : '%s'", transcription)
    logger.info("Avg Latency: %s ms | RTF: %s | WER: %s%% | CER: %s%%", latency_ms, rtf, wer, cer)

    return {
        "metrics": {"latency_ms": latency_ms, "rtf": rtf, "wer": wer, "cer": cer},
        "parameters": {
            "Torch Device": torch_device,
            "Audio Duration (s)": round(audio_duration_s, 2),
            "Transcription": transcription,
        },
        "metadata": {
            "status": True,
            "transcription": transcription,
            "audio_duration_s": round(audio_duration_s, 2),
            "avg_inference_time_s": round(avg_s, 4),
            "run_times_ms": [round(t * 1000, 2) for t in run_times],
            "min_latency_ms": round(min(run_times) * 1000, 2),
            "max_latency_ms": round(max(run_times) * 1000, 2),
        },
    }


def _sanitize_path(path: str) -> str:
    """Sanitize a filesystem path."""
    return "".join(c for c in path)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="PyTorch Whisper ASR benchmark")
    parser.add_argument("--model-size", required=True, help="Whisper model size (e.g. tiny, base)")
    parser.add_argument("--torch-device", default="cpu", help="PyTorch device string (cpu or xpu)")
    parser.add_argument("--model-dir", required=True, help="Directory used as download root for model weights")
    parser.add_argument("--wav-file", default="", help="Path to 16 kHz WAV audio file")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warm-up passes")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of timed inference passes")
    parser.add_argument("--reference", default="", help="Ground-truth transcript for WER/CER")
    parser.add_argument("--output-file", default="", help="Path to write JSON result (required for inference)")
    parser.add_argument("--prepare-model", action="store_true", help="Pre-download model weights and exit")
    args = parser.parse_args()

    safe_model_dir = _sanitize_path(args.model_dir)
    safe_wav_file = _sanitize_path(args.wav_file)
    safe_output_file = _sanitize_path(args.output_file)

    if args.prepare_model:
        try:
            prepare_pytorch_model(model_size=args.model_size, download_root=safe_model_dir)
            logger.info("Model preparation completed successfully")
        except Exception as exc:
            logger.error("Model preparation failed: %s", exc, exc_info=True)
            sys.exit(1)
        sys.exit(0)

    try:
        result = run_pytorch_inference(
            model_size=args.model_size,
            torch_device=args.torch_device,
            pytorch_model_dir=safe_model_dir,
            wav_file_path=safe_wav_file,
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs,
            reference_transcript=args.reference,
        )
        os.makedirs(os.path.dirname(safe_output_file), exist_ok=True)
        with open(safe_output_file, "w") as f:
            json.dump(result, f)
        logger.info("Result written to %s", safe_output_file)
    except _SkipTest as skip_exc:
        skip_result = {
            "metrics": {"latency_ms": -1.0, "rtf": -1.0, "wer": -1.0, "cer": -1.0},
            "parameters": {"Torch Device": args.torch_device},
            "metadata": {"status": "skip", "skip_reason": str(skip_exc)},
        }
        os.makedirs(os.path.dirname(safe_output_file), exist_ok=True)
        with open(safe_output_file, "w") as f:
            json.dump(skip_result, f)
        logger.info("Test skipped: %s", skip_exc)
        sys.exit(0)
    except Exception as exc:
        error_result = {
            "metrics": {"latency_ms": -1.0, "rtf": -1.0, "wer": -1.0, "cer": -1.0},
            "parameters": {"Torch Device": args.torch_device},
            "metadata": {"status": False, "error": str(exc)},
        }
        os.makedirs(os.path.dirname(safe_output_file), exist_ok=True)
        with open(safe_output_file, "w") as f:
            json.dump(error_result, f)
        logger.error("Inference failed: %s", exc, exc_info=True)
        sys.exit(1)
