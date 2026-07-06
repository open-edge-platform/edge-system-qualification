# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Metric calculation and aggregation utilities for ChatQnA Core benchmarking.

Provides functions for:
- Token counting using Llama tokenizer
- Percentile calculation
- Throughput and latency aggregation
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Cached tokenizer instance for efficiency
_tokenizer = None


def _get_tokenizer():
    """
    Get or create a cached tokenizer instance.

    Uses module-level caching to avoid reloading the tokenizer on every call.

    Returns:
        tokenizer: The cached tokenizer instance.

    Raises:
        ImportError: If transformers library is not installed.
    """
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import LlamaTokenizerFast

            _tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer", legacy=False)
        except ImportError as e:
            raise ImportError(
                "transformers library is required for accurate token counting. Install with: pip install transformers"
            ) from e
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in text using Llama tokenizer.

    Args:
        text: Input text to tokenize.

    Returns:
        int: Number of tokens. Returns 0 if tokenization fails.
    """
    if not text:
        return 0

    try:
        tokenizer = _get_tokenizer()
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning("Token counting failed: %s. Falling back to word count.", e)
        return _estimate_tokens_from_words(text)


def _estimate_tokens_from_words(text: str) -> int:
    """
    Fallback: estimate tokens from word count (rough approximation).

    Assumes ~1.3 tokens per word on average for English text.

    Args:
        text: Input text.

    Returns:
        int: Estimated token count.
    """
    word_count = len(text.split())
    return max(1, int(word_count * 1.3))


def percentile(values: List[float], percentile_value: float) -> float:
    """
    Calculate percentile of a list of values.

    Uses linear interpolation between closest ranks.

    Args:
        values: List of numerical values.
        percentile_value: Percentile to calculate (0-100).

    Returns:
        float: The percentile value. Returns -1.0 if list is empty.
    """
    if not values:
        return -1.0

    sorted_values = sorted(float(item) for item in values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (percentile_value / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index

    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def calculate_aggregate_metrics(latencies_ms: List[float], token_counts: List[int]) -> Dict[str, float]:
    """
    Calculate aggregate performance metrics from latency and token data.

    Args:
        latencies_ms: List of query latencies in milliseconds.
        token_counts: List of output token counts per query.

    Returns:
        dict: Dictionary with p50, p95, p99 latencies and throughput metrics.
    """
    if not latencies_ms:
        return {
            "p50_ms": -1.0,
            "p95_ms": -1.0,
            "p99_ms": -1.0,
            "throughput_tokens_per_sec": -1.0,
        }

    total_latency_sec = sum(latencies_ms) / 1000.0
    total_output_tokens = sum(token_counts)

    return {
        "p50_ms": percentile(latencies_ms, 50.0),
        "p95_ms": percentile(latencies_ms, 95.0),
        "p99_ms": percentile(latencies_ms, 99.0),
        "throughput_tokens_per_sec": (total_output_tokens / total_latency_sec if total_latency_sec > 0 else -1.0),
    }


def extract_query_metrics(latency_ms: float, answer_text: str) -> Tuple[int, float]:
    """
    Extract metrics for a single query response.

    Args:
        latency_ms: Query latency in milliseconds.
        answer_text: The LLM response text.

    Returns:
        tuple: (output_token_count, tokens_per_second)
    """
    output_tokens = count_tokens(answer_text)

    if latency_ms > 0 and output_tokens > 0:
        tokens_per_sec = output_tokens / (latency_ms / 1000.0)
    else:
        tokens_per_sec = -1.0

    return output_tokens, tokens_per_sec
