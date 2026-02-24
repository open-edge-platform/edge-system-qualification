# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Video Analytics (VA) Test Suite - Main Entry Point.

This module serves as the main entry point for video analytics tests,
routing to specific pipeline implementations based on test configuration.

The VA pipeline supports different compute modes:
- Mode 0: CPU/CPU/CPU (all stages on CPU)
- Mode 1: dGPU/dGPU/dGPU (all stages on dGPU)
- Mode 2: iGPU/iGPU/iGPU (all stages on iGPU)
- Mode 3: iGPU/iGPU/NPU (decode+detect on iGPU, classify on NPU)
- Mode 4: iGPU/NPU/NPU (decode on iGPU, detect+classify on NPU)
- Mode 5: dGPU/dGPU/NPU (decode+detect on dGPU, classify on NPU)
- Mode 6: dGPU/NPU/NPU (decode on dGPU, detect+classify on NPU)
- Mode 7: iGPU + NPU concurrent (GPU and NPU pipelines run simultaneously)
- Mode 8: dGPU + NPU concurrent (GPU and NPU pipelines run simultaneously)

Pipeline Variants:
- Light: YOLO11n + ResNet-50 (lightweight, fast inference)
- Medium: YOLOv5m + ResNet-50 + MobileNet-v2 (medium complexity)
- Heavy: YOLO11m + ResNet-v1-50 + MobileNet-v2 (highest accuracy)
"""

import logging

import allure

logger = logging.getLogger(__name__)


@allure.title("Video Analytics Light Pipeline Benchmark")
def test_va_light(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Video Analytics Light Pipeline Test.

    Light Pipeline:
    - Detection: YOLOv11n (INT8, 640x640)
    - Classification: ResNet-50 (INT8)
    - Video: fruit-and-vegetable-detection.mp4 (H264)

    This is the fastest VA pipeline optimized for edge devices.
    """
    from esq.suites.ai.vision.src.va.va_light import test_va_light as va_light_impl

    return va_light_impl(
        request,
        configs,
        cached_result,
        cache_result,
        get_kpi_config,
        validate_test_results,
        summarize_test_results,
        validate_system_requirements_from_configs,
        execute_test_with_cache,
        prepare_test,
    )


@allure.title("Video Analytics Medium Pipeline Benchmark")
def test_va_medium(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Video Analytics Medium Pipeline Test.

    Medium Pipeline:
    - Detection: YOLOv5m (medium model, 640x640)
    - Classification: ResNet-50 + MobileNet-v2 (dual classification)
    - Video: apple.mp4 (H265)

    Balanced pipeline for moderate edge computing workloads.
    """
    from esq.suites.ai.vision.src.va.va_medium import test_va_medium as va_medium_impl

    return va_medium_impl(
        request,
        configs,
        cached_result,
        cache_result,
        get_kpi_config,
        validate_test_results,
        summarize_test_results,
        validate_system_requirements_from_configs,
        execute_test_with_cache,
        prepare_test,
    )


@allure.title("Video Analytics Heavy Pipeline Benchmark")
def test_va_heavy(
    request,
    configs,
    cached_result,
    cache_result,
    get_kpi_config,
    validate_test_results,
    summarize_test_results,
    validate_system_requirements_from_configs,
    execute_test_with_cache,
    prepare_test,
):
    """
    Video Analytics Heavy Pipeline Test.

    Heavy Pipeline:
    - Detection: YOLO11m (medium-weight, higher accuracy)
    - Classification: ResNet-v1-50 + MobileNet-v2 (dual classification)
    - Video: bears.h265 (H265 codec)

    Most demanding VA pipeline for maximum accuracy.
    """
    from esq.suites.ai.vision.src.va.va_heavy import test_va_heavy as va_heavy_impl

    return va_heavy_impl(
        request,
        configs,
        cached_result,
        cache_result,
        get_kpi_config,
        validate_test_results,
        summarize_test_results,
        validate_system_requirements_from_configs,
        execute_test_with_cache,
        prepare_test,
    )
