# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO backend for Whisper ASR benchmarking."""

import logging
import os
import subprocess # nosec B404 # For running optimum-cli to export models
import time

logger = logging.getLogger(__name__)


def is_valid_openvino_whisper_model(model_dir: str) -> bool:
    """Return True if model_dir already contains exported OpenVINO Whisper files."""
    markers = [
        "encoder_model.xml",
        "decoder_model.xml",
        "openvino_encoder_model.xml",
    ]
    return any(os.path.exists(os.path.join(model_dir, f)) for f in markers)


def ensure_openvino_whisper_model(
    model_hf_id: str,
    model_dir: str,
    export_timeout: float,
) -> None:
    """Export a Whisper model to OpenVINO IR format via optimum-cli if not already present.

    Args:
        model_hf_id: HuggingFace model identifier, e.g. ``"openai/whisper-base"``.
        model_dir: Target directory for the exported IR files.
        export_timeout: Maximum seconds to wait for the export command.

    Raises:
        TimeoutError: If the export command exceeds *export_timeout* seconds.
        RuntimeError: If the export command fails or produces no model files.
    """
    if os.path.isdir(model_dir) and is_valid_openvino_whisper_model(model_dir):
        logger.info("OpenVINO Whisper model already exists at %s, skipping export", model_dir)
        return

    os.makedirs(model_dir, exist_ok=True)
    logger.info("Exporting '%s' to OpenVINO format at %s ...", model_hf_id, model_dir)

    export_cmd = [
        "optimum-cli", "export", "openvino",
        "--trust-remote-code",
        "--model", model_hf_id,
        model_dir,
    ]
    try:
        proc = subprocess.run(
            export_cmd,
            timeout=export_timeout,
            capture_output=True,
            text=True,
        )
        timed_out = False
        returncode = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = -1
        stderr = ""

    if timed_out:
        raise TimeoutError(
            f"Whisper OpenVINO export timed out after {export_timeout}s for '{model_hf_id}'"
        )
    if returncode != 0:
        raise RuntimeError(
            f"Whisper OpenVINO export failed (exit {returncode}) for '{model_hf_id}':\n{stderr}"
        )
    if not is_valid_openvino_whisper_model(model_dir):
        raise RuntimeError(
            f"Export completed but expected OpenVINO model files not found in {model_dir}"
        )

    logger.info("Model '%s' successfully exported to %s", model_hf_id, model_dir)


def run_openvino_inference(
    model_dir: str,
    ov_device: str,
    wav_file_path: str,
    data_dir: str,
    warmup_runs: int,
    num_runs: int,
    reference_transcript: str,
) -> dict:
    """Run Whisper inference with the OpenVINO GenAI backend.

    Args:
        model_dir: Directory containing the exported OpenVINO IR model.
        ov_device: OpenVINO device string, e.g. ``"CPU"``, ``"GPU"``.
        wav_file_path: Path to a 16 kHz mono WAV file.
        data_dir: Suite data directory (used for GPU kernel cache).
        warmup_runs: Number of un-timed warm-up passes before measurement.
        num_runs: Number of timed inference passes to average.
        reference_transcript: Ground-truth text for WER/CER calculation.

    Returns:
        A dict with keys ``"metrics"``, ``"parameters"``, and ``"metadata"``.
    """
    import librosa
    import openvino_genai

    try:
        from esq.suites.ai.audio.src.containers.utils.metrics_helper import char_error_rate, word_error_rate
    except ImportError:
        from metrics_helper import char_error_rate, word_error_rate  # container-local copy

    # Load audio and derive duration
    raw_speech, _ = librosa.load(wav_file_path, sr=16000)
    audio_duration_s = len(raw_speech) / 16000.0
    raw_speech_list = raw_speech.tolist()

    # Pipeline configuration
    ov_config = {"word_timestamps": True}
    if ov_device != "CPU":
        cache_dir = os.path.join(data_dir, "cache", "asr_cache")
        os.makedirs(cache_dir, exist_ok=True)
        ov_config["CACHE_DIR"] = cache_dir

    pipe = openvino_genai.ASRPipeline(model_dir, ov_device, **ov_config)

    gen_config = pipe.get_generation_config()
    gen_config.language = "<|en|>"
    gen_config.task = "transcribe"
    gen_config.return_timestamps = True
    gen_config.word_timestamps = True

    # Warm-up passes
    if warmup_runs > 0:
        logger.info("Running %d warm-up pass(es)...", warmup_runs)
        for _ in range(warmup_runs):
            pipe.generate(raw_speech_list, gen_config)
        logger.info("Warm-up complete")

    # Timed inference passes
    logger.info("Running %d timed inference pass(es)...", num_runs)
    run_times = []
    transcription = ""
    for run_idx in range(num_runs):
        t0 = time.perf_counter()
        gen_result = pipe.generate(raw_speech_list, gen_config)
        run_times.append(time.perf_counter() - t0)
        if hasattr(gen_result, "texts") and gen_result.texts:
            transcription = str(gen_result.texts[0]).strip()
        else:
            transcription = str(gen_result).strip()
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
            "Device ID": ov_device,
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

    parser = argparse.ArgumentParser(description="OpenVINO Whisper ASR benchmark")
    parser.add_argument("--model-dir", required=True, help="Exported OpenVINO model directory")
    parser.add_argument("--device", default="CPU", help="OpenVINO device string (e.g. CPU, GPU)")
    parser.add_argument("--wav-file", default="", help="Path to 16 kHz WAV audio file")
    parser.add_argument("--data-dir", default="", help="Suite data directory (used for GPU kernel cache)")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warm-up passes")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of timed inference passes")
    parser.add_argument("--reference", default="", help="Ground-truth transcript for WER/CER")
    parser.add_argument("--output-file", default="", help="Path to write JSON result (required for inference)")
    parser.add_argument("--export-model", action="store_true", help="Export model to OpenVINO format and exit")
    parser.add_argument("--model-hf-id", default="", help="HuggingFace model ID (used with --export-model)")
    parser.add_argument("--export-timeout", type=float, default=1800, help="Timeout for model export in seconds")
    args = parser.parse_args()

    safe_model_dir = _sanitize_path(args.model_dir)
    safe_wav_file = _sanitize_path(args.wav_file)
    safe_data_dir = _sanitize_path(args.data_dir)
    safe_output_file = _sanitize_path(args.output_file)

    if args.export_model:
        try:
            ensure_openvino_whisper_model(
                model_hf_id=args.model_hf_id,
                model_dir=safe_model_dir,
                export_timeout=args.export_timeout,
            )
            logger.info("Model export completed successfully")
        except Exception as exc:
            logger.error("Model export failed: %s", exc, exc_info=True)
            sys.exit(1)
        sys.exit(0)

    try:
        result = run_openvino_inference(
            model_dir=safe_model_dir,
            ov_device=args.device,
            wav_file_path=safe_wav_file,
            data_dir=safe_data_dir,
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs,
            reference_transcript=args.reference,
        )
        os.makedirs(os.path.dirname(safe_output_file), exist_ok=True)
        with open(safe_output_file, "w") as f:
            json.dump(result, f)
        logger.info("Result written to %s", safe_output_file)
    except Exception as exc:
        error_result = {
            "metrics": {"latency_ms": -1.0, "rtf": -1.0, "wer": -1.0, "cer": -1.0},
            "parameters": {"Device ID": args.device},
            "metadata": {"status": False, "error": str(exc)},
        }
        os.makedirs(os.path.dirname(safe_output_file), exist_ok=True)
        with open(safe_output_file, "w") as f:
            json.dump(error_result, f)
        logger.error("Inference failed: %s", exc, exc_info=True)
        sys.exit(1)
