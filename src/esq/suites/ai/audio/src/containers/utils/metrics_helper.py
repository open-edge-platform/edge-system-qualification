# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared text-quality metrics for Whisper ASR benchmarking."""


def _edit_distance(seq_a, seq_b):
    """Levenshtein edit distance between two sequences."""
    m, n = len(seq_a), len(seq_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if seq_a[i - 1] == seq_b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def word_error_rate(reference: str, hypothesis: str) -> float:
    from whisper.normalizers import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()
    ref = normalizer(reference).split()
    hyp = normalizer(hypothesis).split()
    return _edit_distance(ref, hyp) / max(len(ref), 1) * 100


def char_error_rate(reference: str, hypothesis: str) -> float:
    from whisper.normalizers import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer()
    ref = list(normalizer(reference).replace(" ", ""))
    hyp = list(normalizer(hypothesis).replace(" ", ""))
    return _edit_distance(ref, hyp) / max(len(ref), 1) * 100