# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Minimal load generator for prompt-based VLM benchmarks.

Sends chat-completion requests (system prompt, user prompt and images) to an
OpenAI-compatible endpoint (OpenVINO Model Server ``/v3/chat/completions``) from a
configurable number of concurrent workers and records per-request latency and token
usage as JSON lines. The caller aggregates the records into test metrics.

Runs in its own venv (see requirements.txt), so it must not import ESQ packages.

Usage: vlm_client.py <config.json>
"""

import base64
import glob
import json
import os
import sys
import threading
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".h264", ".webm")

# Default preprocessing, overridden per profile through ``image_preprocessing``
DEFAULT_PREPROCESSING = {
    "min_size": 224,
    "contrast": 1.15,
    "color": 1.0,
    "sharpen_percent": 60,
    "progressive": False,
    "enhance_after_padding": False,
}


def preprocess_image(img, target, options):
    """Fit the image into a square white canvas, with light contrast, color and sharpness enhancement."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    scale = min(target / width, target / height)
    min_size = options["min_size"]
    new_w = min(max(round(width * scale), min_size), target)
    new_h = min(max(round(height * scale), min_size), target)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def enhance(image):
        image = ImageEnhance.Contrast(image).enhance(options["contrast"])
        if options["color"] != 1.0:
            image = ImageEnhance.Color(image).enhance(options["color"])
        return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=options["sharpen_percent"], threshold=3))

    # Applications differ in whether they enhance the image before or after padding it
    if not options["enhance_after_padding"]:
        resized = enhance(resized)
    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    canvas.paste(resized, ((target - new_w) // 2, (target - new_h) // 2))
    if options["enhance_after_padding"]:
        canvas = enhance(canvas)
    return canvas


def encode_image(img, target, quality, options=None):
    options = {**DEFAULT_PREPROCESSING, **(options or {})}
    buffer = BytesIO()
    preprocess_image(img, target, options).save(
        buffer, format="JPEG", quality=quality, optimize=True, progressive=options["progressive"]
    )
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def read_video_frames(path, count, offset):
    """Return ``count`` frames spread evenly over the video, shifted by ``offset`` frames."""
    capture = cv2.VideoCapture(path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Cannot read frame count of video: {path}")
    positions = sorted({(int(p) + offset) % total for p in np.linspace(0, total - 1, num=count, endpoint=False)})
    frames = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = capture.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from video: {path}")
    return frames


def build_inputs(media_dir, images_per_request):
    """
    Build the list of request inputs from the media directory.

    Every video is one input (frames are sampled per request). Images are grouped
    ``images_per_request`` at a time, in file name order, so one request carries one group.
    """
    files = sorted(glob.glob(os.path.join(media_dir, "**", "*"), recursive=True))
    videos = [f for f in files if f.lower().endswith(VIDEO_EXTENSIONS)]
    images = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
    inputs = [("video", [v]) for v in videos]
    for start in range(0, len(images), images_per_request):
        inputs.append(("images", images[start : start + images_per_request]))
    if not inputs:
        raise RuntimeError(f"No images or videos found in media directory: {media_dir}")
    return inputs


def load_request_images(kind, paths, images_per_request, request_index):
    if kind == "video":
        # Shift the sampled frames on every pass so repeated requests do not hit identical inputs
        return read_video_frames(paths[0], images_per_request, offset=request_index * 7)
    return [Image.open(p) for p in paths]


class Recorder:
    def __init__(self, path):
        self._lock = threading.Lock()
        self._file = open(path, "w")  # noqa: SIM115 - held open for the whole run, closed in close()

    def write(self, record):
        with self._lock:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()

    def close(self):
        self._file.close()


def build_request_body(cfg, kind, paths, index):
    """Build the chat-completion body of one request (system prompt, user prompt and images)."""
    images = load_request_images(kind, paths, cfg["images_per_request"], index)
    prompt = cfg["prompt"].replace("${NUM_FRAMES}", str(len(images)))
    content = [{"type": "text", "text": prompt}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image(
                        image, cfg["image_max_size"], cfg["jpeg_quality"], cfg.get("image_preprocessing")
                    )
                },
            }
        )
    messages = []
    if cfg.get("system_prompt"):
        messages.append({"role": "system", "content": cfg["system_prompt"]})
    messages.append({"role": "user", "content": content})
    body = {
        "model": cfg["model"],
        "messages": messages,
        "max_completion_tokens": cfg["max_new_tokens"],
        "temperature": cfg["temperature"],
    }
    # Optional request fields are sent only when the profile sets them, so a request
    # can match exactly what the source application sends
    if cfg.get("enable_thinking") is not None:
        body["chat_template_kwargs"] = {"enable_thinking": cfg["enable_thinking"]}
    if cfg.get("response_format"):
        body["response_format"] = cfg["response_format"]
    return body


def worker_loop(worker_id, cfg, inputs, start_time, measure_start, end_time, recorder, barrier):
    endpoint = cfg["endpoint"]
    index = worker_id
    # With requests_per_worker every worker sends a fixed number of requests and stops, instead of
    # running until the duration ends. All workers start together, so N workers form a burst of N
    # concurrent requests.
    limit = cfg.get("requests_per_worker")
    sent_count = 0
    barrier.wait()
    while time.time() < end_time and (limit is None or sent_count < limit):
        sent_count += 1
        kind, paths = inputs[index % len(inputs)]
        request_id = f"worker{worker_id}_req{index}"
        index += cfg["workers"]
        try:
            body = build_request_body(cfg, kind, paths, index)
            sent = time.time()
            response = requests.post(endpoint, json=body, timeout=cfg["request_timeout"])
            latency = time.time() - sent
            response.raise_for_status()
            result = response.json()
            usage = result.get("usage", {})
            recorder.write(
                {
                    "id": request_id,
                    "ok": True,
                    "warmup": sent < measure_start,
                    "start_time": sent,
                    "latency_sec": latency,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "text": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                }
            )
        except Exception as error:  # noqa: BLE001 - every failure is recorded and counted
            recorder.write({"id": request_id, "ok": False, "warmup": False, "error": str(error)})
            time.sleep(1)


def ttft_probe(cfg, inputs):
    """
    Send ``ttft_requests`` streaming requests one after another and record the time to first token of each.

    Run right after the model server starts: the first request is the cold start, the following ones are
    warm. The requests are the same as the benchmark requests, only streamed, so the benchmark itself stays
    non-streaming like the source applications.
    """
    recorder = Recorder(cfg["output_file"])
    for index in range(cfg["ttft_requests"]):
        kind, paths = inputs[index % len(inputs)]
        request_id = f"ttft_req{index}"
        try:
            body = build_request_body(cfg, kind, paths, index)
            body["stream"] = True
            sent = time.time()
            ttft = None
            completion_chunks = 0
            with requests.post(cfg["endpoint"], json=body, timeout=cfg["request_timeout"], stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                        continue
                    choices = json.loads(line[5:]).get("choices") or [{}]
                    if choices[0].get("delta", {}).get("content"):
                        completion_chunks += 1
                        if ttft is None:
                            ttft = time.time() - sent
            latency = time.time() - sent
            if ttft is None:
                raise RuntimeError("the stream ended before the first token")
            recorder.write(
                {
                    "id": request_id,
                    "ok": True,
                    "probe": "ttft",
                    "cold": index == 0,
                    "ttft_sec": ttft,
                    "latency_sec": latency,
                    "completion_chunks": completion_chunks,
                }
            )
            print(f"TTFT request {index}: {ttft:.3f}s ({'cold' if index == 0 else 'warm'})", flush=True)
        except Exception as error:  # noqa: BLE001 - every failure is recorded and counted
            recorder.write({"id": request_id, "ok": False, "probe": "ttft", "cold": index == 0, "error": str(error)})
    recorder.close()


def send_warmup(cfg):
    """
    Send one small request to trigger the lazy graph and kernel initialization of the model server,
    so the first measured request does not pay for it. ``warmup`` has the prompt, image size and
    token limit of the request.
    """
    warmup = cfg["warmup"]
    size = warmup.get("image_size", 64)
    image = Image.new("RGB", (size, size), (200, 200, 200))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
    body = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": warmup["prompt"]},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        "max_completion_tokens": warmup.get("max_new_tokens", 8),
        "temperature": 0.0,
    }
    if cfg.get("response_format"):
        body["response_format"] = cfg["response_format"]
    started = time.time()
    requests.post(cfg["endpoint"], json=body, timeout=cfg["request_timeout"]).raise_for_status()
    print(f"Warm-up request finished in {time.time() - started:.2f}s", flush=True)


def main():
    # Copy the command-line path character by character and accept only an existing JSON file
    config_path = os.path.realpath("".join(c for c in sys.argv[1]))
    if not config_path.endswith(".json") or not os.path.isfile(config_path):
        sys.exit(f"Config must be an existing .json file: {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    if cfg.get("warmup_only"):
        send_warmup(cfg)
        return
    inputs = build_inputs(cfg["media_dir"], cfg["images_per_request"])
    if cfg.get("ttft_requests"):
        ttft_probe(cfg, inputs)
        return
    print(f"Inputs: {len(inputs)}, workers: {cfg['workers']}, duration: {cfg['duration']}s", flush=True)

    recorder = Recorder(cfg["output_file"])
    barrier = threading.Barrier(cfg["workers"])
    start_time = time.time()
    measure_start = start_time + cfg["init_duration"]
    end_time = measure_start + cfg["duration"]
    threads = [
        threading.Thread(
            target=worker_loop,
            args=(i, cfg, inputs, start_time, measure_start, end_time, recorder, barrier),
            daemon=True,
        )
        for i in range(cfg["workers"])
    ]
    for thread in threads:
        thread.start()
    # A request started before end_time may finish after it, allow it to complete
    for thread in threads:
        thread.join(timeout=cfg["duration"] + cfg["init_duration"] + cfg["request_timeout"] + 30)
    recorder.close()
    print("Benchmark finished", flush=True)


if __name__ == "__main__":
    main()
