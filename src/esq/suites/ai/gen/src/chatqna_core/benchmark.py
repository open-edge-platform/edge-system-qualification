# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark execution for ChatQnA Core application.

Orchestrates:
- Document ingestion into the Q&A corpus
- Query execution (streaming and non-streaming)
- Metric collection (latency, TTFT, throughput, token counts)
"""

import json
import logging
import mimetypes
import time
import uuid
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from typing import Dict, List, Tuple
from urllib import error, request
from urllib.parse import urlparse

from esq.suites.ai.gen.src.chatqna_core.metrics import (
    extract_query_metrics,
    percentile,
)

logger = logging.getLogger(__name__)

SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".txt", ".docx"}


def _as_int(value: object, default: int) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def _build_json_request(url: str, payload: Dict[str, object], timeout_seconds: int) -> request.Request:
    payload_bytes = json.dumps(payload).encode("utf-8")
    return request.Request(
        url=url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _extract_answer_text(body: Dict[str, object]) -> str:
    answer_text = body.get("text") or body.get("answer") or body.get("response") or ""
    if answer_text:
        if isinstance(answer_text, str):
            return answer_text
        return json.dumps(answer_text)

    # OpenAI-style fallback: {"choices":[{"message":{"content":"..."}}]}
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            content = first_choice.get("content")
            if isinstance(content, str):
                return content

    return ""


def _chat_payload_candidates(prompt: str, stream: bool = False) -> List[Dict[str, object]]:
    # Prefer "input" first because current ChatQnA schema explicitly requires it.
    base_candidates: List[Dict[str, object]] = [
        {"input": prompt},
        {"query": prompt},
        {"question": prompt},
        {"text": prompt},
        {
            "messages": [
                {"role": "user", "content": prompt},
            ]
        },
    ]
    if stream:
        return [{**payload, "stream": True} for payload in base_candidates]
    # Keep explicit stream=false first for APIs that default to streaming behavior.
    return [{"input": prompt, "stream": False}, *base_candidates]


def _is_transient_chat_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionResetError, IncompleteRead, RemoteDisconnected)):
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = (
        "incompleteread",
        "connection reset",
        "timed out",
        "remote end closed",
        "temporary failure",
        "infer request is busy",
    )
    return any(marker in message for marker in transient_markers)


def _read_http_error_body(exc: error.HTTPError) -> str:
    try:
        raw_body = exc.read()
        if not raw_body:
            return ""
        return raw_body.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _load_json_response(http_response) -> Dict[str, object]:
    raw_body = http_response.read().decode("utf-8")
    if not raw_body:
        return {}
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return {"raw_body": raw_body}
    if isinstance(payload, dict):
        return payload
    return {"data": payload}


def _string_success_signal(value: object) -> bool:
    normalized = str(value).strip().lower()
    return normalized in {"success", "ok", "healthy", "ready", "running", "passed"}


def _health_response_is_ready(body: Dict[str, object]) -> bool:
    if not body:
        return True

    for key in ("success", "ok", "healthy", "ready"):
        if body.get(key) is True:
            return True

    for key in ("status", "state", "result", "message", "detail", "raw_body"):
        value = body.get(key)
        if value is not None and _string_success_signal(value):
            return True

    return False


def _upload_response_is_success(body: Dict[str, object]) -> bool:
    if not body:
        return False

    for key in ("success", "ok"):
        if body.get(key) is True:
            return True

    for key in ("status", "state", "result", "message"):
        value = body.get(key)
        if value is not None and _string_success_signal(value):
            return True

    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        documents = metadata.get("documents")
        if isinstance(documents, list):
            return True

    documents = body.get("documents")
    if isinstance(documents, list):
        return True

    return False


def _open_url(target, timeout_seconds: int):
    parsed = urlparse(target.full_url if isinstance(target, request.Request) else str(target))
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme '{scheme}'. Only http/https are allowed.")

    hostname = (parsed.hostname or "").lower()
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        return request.build_opener(request.ProxyHandler({})).open(target, timeout=timeout_seconds)
    return request.build_opener().open(target, timeout=timeout_seconds)


def wait_for_service_health(health_url: str, timeout_seconds: int, poll_interval_seconds: int) -> float:
    deadline = time.time() + max(1, int(timeout_seconds))
    poll_interval = max(1, int(poll_interval_seconds))
    start_time = time.perf_counter()
    elapsed_reported = 0

    while time.time() < deadline:
        try:
            with _open_url(health_url, timeout_seconds=poll_interval) as response:
                if 200 <= response.status < 300:
                    body = _load_json_response(response)
                    if _health_response_is_ready(body):
                        elapsed = time.perf_counter() - start_time
                        logger.info(
                            "Chat Q&A Core health check passed at %s after %.1fs",
                            health_url,
                            elapsed,
                        )
                        return elapsed
        except Exception:
            pass

        elapsed = int(time.perf_counter() - start_time)
        if elapsed - elapsed_reported >= 60:
            elapsed_reported = elapsed
            logger.info(
                "Still waiting for health at %s... %ds elapsed / %ds timeout",
                health_url,
                elapsed,
                timeout_seconds,
            )

        time.sleep(poll_interval)

    raise RuntimeError(
        f"Chat Q&A Core health endpoint did not become ready in time: {health_url} "
        f"(waited {int(time.perf_counter() - start_time)}s)"
    )


def _read_prompt_lines(prompt_file: str, prompt_limit: int) -> List[str]:
    prompt_path = Path(prompt_file).expanduser().resolve()
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    prompts: List[str] = []

    if prompt_path.suffix.lower() == ".json":
        payload = json.loads(prompt_text)
        if isinstance(payload, list):
            prompts = [str(item).strip() for item in payload if str(item).strip()]
        elif isinstance(payload, dict):
            prompts = [str(item).strip() for item in payload.get("prompts", []) if str(item).strip()]
    else:
        prompts = [line.strip() for line in prompt_text.splitlines() if line.strip()]

    if not prompts:
        raise RuntimeError(f"No prompts were found in the configured prompt file: {prompt_path}")

    limit = max(1, int(prompt_limit))
    return prompts[:limit]


def _list_corpus_documents(corpus_dir: str) -> List[Path]:
    corpus_root = Path(corpus_dir).expanduser().resolve()
    documents = [
        file_path
        for file_path in sorted(corpus_root.iterdir())
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
    ]
    if not documents:
        raise RuntimeError(
            f"No supported documents were found in benchmark corpus directory: {corpus_root}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))}"
        )
    return documents


def _build_multipart_payload(field_name: str, file_path: Path) -> Tuple[bytes, str]:
    boundary = f"----esq-chatqna-{uuid.uuid4().hex}"
    file_bytes = file_path.read_bytes()
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"

    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        (f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n').encode("utf-8"),
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), boundary


def _delete_documents(api_base_url: str, timeout_seconds: int) -> None:
    delete_url = f"{api_base_url.rstrip('/')}/documents?delete_all=True"
    delete_request = request.Request(url=delete_url, method="DELETE")
    try:
        with _open_url(delete_request, timeout_seconds=timeout_seconds):
            return
    except error.HTTPError as exc:
        if exc.code in {404, 422}:
            return
        raise


def _ingest_documents(api_base_url: str, documents: List[Path], timeout_seconds: int) -> Tuple[float, List[str]]:
    start_time = time.perf_counter()
    ingested_names: List[str] = []

    for document_path in documents:
        request_body, boundary = _build_multipart_payload("files", document_path)
        upload_request = request.Request(
            url=f"{api_base_url.rstrip('/')}/documents",
            data=request_body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with _open_url(upload_request, timeout_seconds=timeout_seconds) as response:
            body = _load_json_response(response)
            if not _upload_response_is_success(body):
                raise RuntimeError(f"Document upload failed for {document_path.name}: {body}")
            metadata = body.get("metadata", {})
            if isinstance(metadata, dict):
                document_names = metadata.get("documents", [])
                if isinstance(document_names, list):
                    ingested_names.extend(str(item) for item in document_names)

    return time.perf_counter() - start_time, ingested_names


def _run_non_stream_query(api_base_url: str, prompt: str, timeout_seconds: int) -> Tuple[float, str]:
    chat_url = f"{api_base_url.rstrip('/')}/chat"
    errors_seen: List[str] = []
    max_attempts_per_payload = 3

    for payload in _chat_payload_candidates(prompt, stream=False):
        for attempt in range(1, max_attempts_per_payload + 1):
            start_time = time.perf_counter()
            chat_request = _build_json_request(
                url=chat_url,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            try:
                with _open_url(chat_request, timeout_seconds=timeout_seconds) as response:
                    body = _load_json_response(response)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                answer_text = _extract_answer_text(body)
                return elapsed_ms, answer_text
            except error.HTTPError as exc:
                body = _read_http_error_body(exc)
                errors_seen.append(f"payload={payload} -> HTTP {exc.code}: {body[:240]}")
                if exc.code != 422:
                    raise
                # 422 is schema mismatch for this payload; try next payload shape.
                break
            except Exception as exc:
                errors_seen.append(f"payload={payload} -> {type(exc).__name__}: {exc}")
                if _is_transient_chat_error(exc) and attempt < max_attempts_per_payload:
                    time.sleep(1)
                    continue
                break

    raise RuntimeError(f"All chat payload candidates failed at {chat_url}. Errors: {' | '.join(errors_seen[:5])}")


def _measure_ttft(api_base_url: str, prompt: str, timeout_seconds: int) -> float:
    """Measure time-to-first-token via streaming chat. Returns -1.0 on failure."""
    chat_url = f"{api_base_url.rstrip('/')}/chat"
    errors_seen: List[str] = []

    for payload in _chat_payload_candidates(prompt, stream=True):
        start_time = time.perf_counter()
        chat_request = _build_json_request(
            url=chat_url,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        try:
            with _open_url(chat_request, timeout_seconds=timeout_seconds) as response:
                while True:
                    chunk = response.readline()
                    if not chunk:
                        logger.warning(
                            "TTFT: streaming response ended before first token for payload=%s at %s",
                            payload,
                            chat_url,
                        )
                        break
                    if chunk.strip():
                        return (time.perf_counter() - start_time) * 1000.0
        except error.HTTPError as exc:
            body = _read_http_error_body(exc)
            errors_seen.append(f"payload={payload} -> HTTP {exc.code}: {body[:240]}")
            if exc.code != 422:
                logger.warning("TTFT measurement failed at %s: HTTP %s", chat_url, exc.code)
                return -1.0
        except Exception as ttft_error:
            errors_seen.append(f"payload={payload} -> {type(ttft_error).__name__}: {ttft_error}")

    logger.warning(
        "TTFT measurement failed for all payload candidates at %s. Errors: %s",
        chat_url,
        " | ".join(errors_seen[:5]),
    )
    return -1.0


def run_chatqna_benchmark(configs: Dict[str, object], api_base_url: str) -> Dict[str, object]:
    request_timeout_seconds = _as_int(configs.get("request_timeout_seconds", 180), default=180)
    prompt_limit = _as_int(configs.get("prompt_limit", 5), default=5)
    corpus_dir = str(configs.get("corpus_dir", "")).strip()
    prompt_file = str(configs.get("prompt_file", "")).strip()
    keep_documents_after_run = bool(configs.get("keep_documents_after_run", False))
    warmup_query = str(configs.get("warmup_query", "")).strip()

    prompts = _read_prompt_lines(prompt_file, prompt_limit=prompt_limit)
    documents = _list_corpus_documents(corpus_dir)

    _delete_documents(api_base_url, timeout_seconds=request_timeout_seconds)
    ingestion_seconds, ingested_documents = _ingest_documents(
        api_base_url,
        documents,
        timeout_seconds=request_timeout_seconds,
    )

    if warmup_query:
        _run_non_stream_query(api_base_url, warmup_query, timeout_seconds=request_timeout_seconds)

    latencies_ms: List[float] = []
    successful_queries = 0
    failed_queries = 0
    output_token_counts: List[int] = []
    answers: List[str] = []

    for prompt in prompts:
        try:
            latency_ms, answer_text = _run_non_stream_query(
                api_base_url,
                prompt,
                timeout_seconds=request_timeout_seconds,
            )
            latencies_ms.append(latency_ms)
            successful_queries += 1
            answers.append(answer_text)

            output_tokens, tps = extract_query_metrics(latency_ms, answer_text)
            output_token_counts.append(output_tokens)
        except Exception as query_error:
            failed_queries += 1
            logger.warning(
                "Chat Q&A Core query failed for prompt '%.80s': %s: %s",
                prompt,
                type(query_error).__name__,
                query_error,
            )

    # Measure TTFT after non-streaming queries so transient streaming errors do not
    # interfere with primary latency collection.
    ttft_ms = _measure_ttft(api_base_url, prompts[0], timeout_seconds=request_timeout_seconds)

    if not keep_documents_after_run:
        _delete_documents(api_base_url, timeout_seconds=request_timeout_seconds)

    total_queries = successful_queries + failed_queries
    query_success_rate = float(successful_queries) / float(total_queries) if total_queries else 0.0

    # Calculate throughput using accurate token counts
    total_latency_sec = sum(latencies_ms) / 1000.0 if latencies_ms else 0.0
    total_output_tokens = sum(output_token_counts)
    estimated_tokens_per_second = total_output_tokens / total_latency_sec if total_latency_sec > 0 else -1.0

    return {
        "corpus_document_count": len(documents),
        "ingested_documents": ingested_documents,
        "prompt_count": len(prompts),
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "query_success_rate": query_success_rate,
        "document_ingestion_seconds": ingestion_seconds,
        "ttft_ms": ttft_ms,
        "p50_query_latency_ms": percentile(latencies_ms, 50.0),
        "p95_query_latency_ms": percentile(latencies_ms, 95.0),
        "estimated_output_tokens_per_second": estimated_tokens_per_second,
        "answers": answers,
        "latencies_ms": latencies_ms,
        "output_token_counts": output_token_counts,
    }
