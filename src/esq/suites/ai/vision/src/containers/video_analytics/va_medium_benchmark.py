# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Video Analytics (VA) Medium Pipeline Benchmark.

Multi-stage video analytics pipeline benchmark with:
- Decode stage: H265 video decoding (H.265/HEVC)
- Detection stage: YOLOv5m object detection (medium model, 640x640)
- Tracking stage: Short-term imageless tracking
- Classification stage: ResNet-50 + MobileNet-v2 dual classification

Medium pipeline is more complex than Light:
- Uses YOLOv5m (medium model)
- Dual classification with ResNet-50 and MobileNet-v2
- Uses H.265 video (1920x1080@30fps bears footage)

Supports multiple compute modes for heterogeneous device usage:
- Mode 0: CPU/CPU/CPU (all stages on CPU)
- Mode 1: dGPU/dGPU/dGPU (all stages on dGPU)
- Mode 2: iGPU/iGPU/iGPU (all stages on iGPU)
- Mode 3: iGPU/iGPU/NPU (decode+detect on iGPU, classify on NPU)
- Mode 4: iGPU/NPU/NPU (decode on iGPU, detect+classify on NPU)
- Mode 5: dGPU/dGPU/NPU (decode+detect on dGPU, classify on NPU)
- Mode 6: dGPU/NPU/NPU (decode on dGPU, detect+classify on NPU)
"""

import argparse
import gc
import os
import re
import sys
import time

# Add consolidated utilities to path
sys.path.insert(0, "/home/dlstreamer")

from base_benchmark import BaseVideoAnalyticsBenchmark
from esq_utils.media.memory_utils import (
    check_available_memory,
    get_memory_based_max_streams,
    log_memory_status,
)

# VA compute modes: (decode_device, detect_device, classify_device)
# Modes 0-6: Standard modes where stages run on specified devices
# Modes 7-8: Concurrent modes where GPU and NPU pipelines run simultaneously
VA_COMPUTE_MODES = {
    "Mode 0": ("CPU", "CPU", "CPU"),
    "Mode 1": ("dGPU", "dGPU", "dGPU"),
    "Mode 2": ("iGPU", "iGPU", "iGPU"),
    "Mode 3": ("iGPU", "iGPU", "NPU"),
    "Mode 4": ("iGPU", "NPU", "NPU"),
    "Mode 5": ("dGPU", "dGPU", "NPU"),
    "Mode 6": ("dGPU", "NPU", "NPU"),
    # Concurrent modes: GPU and NPU pipelines run simultaneously
    # Tuple format: (decode_device, detect_device, classify_device, concurrent_device)
    # concurrent_device indicates this is a concurrent mode with NPU
    "Mode 7": ("iGPU", "iGPU", "iGPU", "NPU_CONCURRENT"),  # iGPU + NPU concurrent
    "Mode 8": ("dGPU", "dGPU", "dGPU", "NPU_CONCURRENT"),  # dGPU + NPU concurrent
}

# Track executed modes to avoid duplicates across device runs
va_medium_executed_modes = set()


class VAMediumBenchmark(BaseVideoAnalyticsBenchmark):
    """
    Video Analytics (VA) Medium Pipeline Benchmark.

    Implements the VA Medium pipeline with YOLOv5m detection, object tracking,
    and dual classification (ResNet-50 + MobileNet-v2) across configurable compute modes.
    """

    def __init__(
        self,
        name,
        device,
        monitor_num,
        is_MTL,
        has_igpu,
        target_fps,
        telemetry_file_prefix,
        log_file,
        result_file_prefix,
        csv_path,
        sink_dir,
        config_file_path=None,
    ):
        super().__init__(
            name,
            device,
            monitor_num,
            is_MTL,
            has_igpu,
            target_fps,
            telemetry_file_prefix,
            log_file,
            result_file_prefix,
            csv_path,
            sink_dir,
            config_file_path,
        )

        # VA video does not need looping - video is already long enough
        self.loop_video_commands = []

        # Model paths for VA medium pipeline
        # Resources mounted from esq_data:
        # - models at: /home/dlstreamer/share/models/medium/...
        # - videos at: /home/dlstreamer/sample_video/medium/video/...
        self.model_path = {
            "yolov5m": {
                "det_model_path": "/home/dlstreamer/share/models/medium/detection/yolov5m_640x640/INT8/yolov5m/INT8/yolov5m.xml",
                "det_proc_json_path": "/home/dlstreamer/model_proc/yolo-v5.json",
            },
            "resnet-50-tf": {
                "cls_model_path": "/home/dlstreamer/share/models/medium/classification/resnet-50-tf/INT8/resnet-50-tf.xml",
                "cls_proc_json_path": "/home/dlstreamer/model_proc/resnet-50.json",
            },
            "mobilenet-v2-pytorch": {
                "cls_model_path": "/home/dlstreamer/share/models/medium/classification/mobilenet-v2-pytorch/INT8/mobilenet-v2-pytorch.xml",
                "cls_proc_json_path": "/home/dlstreamer/model_proc/mobilenet-v2.json",
            },
        }

    def get_sub_elements(self, stage, dev, is_concurrent=False):
        """
        Get GStreamer sub-elements for a specific stage and device.

        Args:
            stage: Pipeline stage ("decode", "detect", "classify")
            dev: Device type ("CPU", "iGPU", "dGPU", "NPU")
            is_concurrent: If True, use concurrent mode parameters (different nireq/batch for NPU)

        Returns:
            str: GStreamer element string for the stage
        """
        if stage == "decode":
            # Decode stage: H264 video decoding (H.264/AVC)
            # Note: Despite .h265 extension, files are MP4 containers with H.264 codec
            return {
                "CPU": "qtdemux ! h264parse ! avdec_h264 ! capsfilter caps=video/x-raw",
                "iGPU": "qtdemux ! h264parse ! vah264dec ! capsfilter caps=video/x-raw\\(memory:VAMemory\\)",
                "dGPU": "qtdemux ! h264parse ! varenderD129h264dec ! capsfilter caps=video/x-raw\\(memory:VAMemory\\)",
                "NPU": "unsupported",  # No decode on NPU
            }.get(dev, "unknown_decode")

        elif stage == "detect":
            # Detection stage: YOLOv5m with CPU/GPU/NPU support
            # GPU: nireq=2, batch-size=8, va-surface-sharing
            # NPU concurrent: nireq=4, batch-size=1, opencv
            # NPU split: nireq=4, batch-size=1
            if dev == "NPU" and is_concurrent:
                return "device=NPU pre-process-backend=opencv nireq=4 batch-size=1 inference-interval=3 threshold=0.5 model-instance-id=yolov5m"
            return {
                "CPU": "device=CPU pre-process-backend=opencv ie-config=NUM_STREAMS=2 batch-size=8 inference-interval=3 threshold=0.5 model-instance-id=yolov5m",
                "iGPU": "device=GPU pre-process-backend=va-surface-sharing nireq=2 ie-config=NUM_STREAMS=2 batch-size=8 inference-interval=3 threshold=0.5 model-instance-id=yolov5m",
                "dGPU": "device=GPU.1 pre-process-backend=va-surface-sharing nireq=2 ie-config=NUM_STREAMS=2 batch-size=8 inference-interval=3 threshold=0.5 model-instance-id=yolov5m",
                "NPU": "device=NPU pre-process-backend=opencv nireq=4 batch-size=1 inference-interval=3 threshold=0.5 model-instance-id=yolov5m",
            }.get(dev, "unknown_detect")

        elif stage == "classify":
            # Classification stage: Used for both ResNet-50 and MobileNet-v2
            # GPU: nireq=2, batch-size=8, va-surface-sharing
            # NPU concurrent: nireq=4, batch-size=1, opencv
            # NPU split: nireq not specified, batch-size=8
            if dev == "NPU" and is_concurrent:
                return (
                    "device=NPU pre-process-backend=opencv nireq=4 batch-size=1 inference-interval=3 inference-region=1"
                )
            return {
                "CPU": "device=CPU pre-process-backend=opencv ie-config=NUM_STREAMS=2 batch-size=8 inference-interval=3 inference-region=1",
                "iGPU": "device=GPU pre-process-backend=va-surface-sharing nireq=2 ie-config=NUM_STREAMS=2 batch-size=8 inference-interval=3 inference-region=1",
                "dGPU": "device=GPU.1 pre-process-backend=va-surface-sharing nireq=2 ie-config=NUM_STREAMS=2 batch-size=8 inference-interval=3 inference-region=1",
                "NPU": "device=NPU pre-process-backend=opencv nireq=4 batch-size=1 inference-interval=3 inference-region=1",
            }.get(dev, "unknown_classify")

        return "unknown_element"

    def get_normalize_device_name(self, dev):
        """
        Normalize device name to canonical form.

        Args:
            dev: Device name (e.g., "dGPU.0", "dGPU.1", "GPU.0", "GPU.1")

        Returns:
            str: Normalized device name (e.g., "dGPU", "iGPU")
        """
        # Remove numeric suffix first
        base_dev = re.sub(r"\.\d+$", "", dev)

        # Map GPU to iGPU/dGPU based on full device name
        device_mapping = {
            "GPU": "iGPU",
            "GPU.0": "iGPU",
            "GPU.1": "dGPU",
        }

        if dev in device_mapping:
            return device_mapping[dev]
        return base_dev

    def get_mode_and_compute_devices(self, available_devices, executed_modes):
        """
        Get compute modes that can run on this device configuration.

        Args:
            available_devices: List of available device types
            executed_modes: Set of already executed mode names

        Returns:
            dict: Mode name -> device tuple for modes to execute
        """
        modes_dict = {}
        normalized_curr_dev = self.get_normalize_device_name(self.device)
        allowed_devices = set(available_devices)
        allowed_devices.add("CPU")  # CPU is always available

        print("[DEBUG] get_mode_and_compute_devices:", file=sys.stderr)
        print(f"[DEBUG]   self.device: {self.device}", file=sys.stderr)
        print(f"[DEBUG]   normalized_curr_dev: {normalized_curr_dev}", file=sys.stderr)
        print(f"[DEBUG]   available_devices: {available_devices}", file=sys.stderr)
        print(f"[DEBUG]   allowed_devices: {allowed_devices}", file=sys.stderr)
        print(f"[DEBUG]   executed_modes: {executed_modes}", file=sys.stderr)

        for mode_name, devices in VA_COMPUTE_MODES.items():
            # Skip if already executed
            if mode_name in executed_modes:
                print(f"[DEBUG]   {mode_name}: SKIPPED (already executed)", file=sys.stderr)
                continue

            # Check for concurrent mode (4-element tuple with NPU_CONCURRENT)
            is_concurrent = len(devices) == 4 and devices[3] == "NPU_CONCURRENT"

            if is_concurrent:
                # Concurrent mode requires: GPU (iGPU or dGPU) AND NPU
                gpu_device = self.get_normalize_device_name(devices[0])
                requires_npu = "NPU" in allowed_devices
                gpu_available = gpu_device in allowed_devices
                dev_in_mode = normalized_curr_dev == gpu_device

                print(
                    f"[DEBUG]   {mode_name}: CONCURRENT mode, gpu={gpu_device}, gpu_available={gpu_available}, npu_available={requires_npu}, dev_in_mode={dev_in_mode}",
                    file=sys.stderr,
                )

                if gpu_available and requires_npu and dev_in_mode:
                    modes_dict[mode_name] = list(devices)
                    executed_modes.add(mode_name)
                    print(f"[DEBUG]   {mode_name}: ADDED (concurrent)", file=sys.stderr)
            else:
                # Standard mode: check all devices are allowed
                normalized_devices = [self.get_normalize_device_name(d) for d in devices]

                all_allowed = all(d in allowed_devices for d in normalized_devices)
                dev_in_mode = normalized_curr_dev in normalized_devices
                print(
                    f"[DEBUG]   {mode_name}: devices={devices}, normalized={normalized_devices}, all_allowed={all_allowed}, dev_in_mode={dev_in_mode}",
                    file=sys.stderr,
                )

                # Check if all devices are allowed AND current device is in the mode
                if all_allowed and dev_in_mode:
                    modes_dict[mode_name] = list(devices)
                    executed_modes.add(mode_name)
                    print(f"[DEBUG]   {mode_name}: ADDED", file=sys.stderr)

        print(f"[DEBUG]   Final modes_dict: {modes_dict}", file=sys.stderr)
        return modes_dict

    def get_available_devices(self):
        """
        Detect available devices on the system.

        Returns:
            list: Available device types (iGPU, dGPU, NPU)
        """
        devices = []

        # Check for iGPU: /dev/dri/renderD128
        if os.path.exists("/dev/dri/renderD128"):
            devices.append("iGPU")

        # Check for dGPU: /dev/dri/renderD129
        if os.path.exists("/dev/dri/renderD129"):
            devices.append("dGPU")

        # Check for NPU: any /dev/accel/accel*
        accel_path = "/dev/accel"
        if os.path.exists(accel_path) and any(name.startswith("accel") for name in os.listdir(accel_path)):
            devices.append("NPU")

        return devices

    def get_config_of_platform(self):
        """
        Get platform-specific configuration based on device type.

        Medium pipeline has lower reference stream counts than Light
        due to larger models (YOLOv5m) and dual classification.
        """
        available_devices = self.get_available_devices()

        # Get device type for config selection
        raw_device = self.device.split(".")[0] if "." in self.device else self.device

        device_mapping = {
            "GPU": "iGPU",
            "GPU.0": "iGPU",
            "GPU.1": "dGPU",
            "iGPU": "iGPU",
            "dGPU": "dGPU",
            "CPU": "CPU",
            "NPU": "NPU",
        }

        if self.device in device_mapping:
            device_type = device_mapping[self.device]
        elif raw_device in device_mapping:
            device_type = device_mapping[raw_device]
        else:
            device_type = raw_device

        self.logger.debug(f"Device mapping: {self.device} -> {device_type}")

        # Medium pipeline reference streams are lower than Light due to:
        # - Larger YOLOv5m model vs YOLO11n
        # - Dual classification (ResNet-50 + MobileNet-v2)
        # - H265 decode overhead

        if device_type == "iGPU":
            if self.VDBOX == 1:
                self.config = {
                    "ref_stream_list": [5],
                    "ref_platform": "i5-12400 (16G Mem)",
                    "ref_gpu_freq": -1,
                    "ref_pkg_power": -1,
                    "models": ["yolov5m+resnet-50-tf+mobilenet-v2-pytorch"],
                    "modes": self.get_mode_and_compute_devices(available_devices, va_medium_executed_modes),
                    "mode_ref_streams": {
                        "Mode 0": 3,  # CPU/CPU/CPU
                        "Mode 2": 5,  # iGPU/iGPU/iGPU
                    },
                }
            else:
                if self.is_MTL:
                    self.config = {
                        "ref_stream_list": [10],
                        "ref_platform": "MTL 165H (32G Mem)",
                        "ref_gpu_freq": -1,
                        "ref_pkg_power": -1,
                        "models": ["yolov5m+resnet-50-tf+mobilenet-v2-pytorch"],
                        "modes": self.get_mode_and_compute_devices(available_devices, va_medium_executed_modes),
                        "mode_ref_streams": {
                            "Mode 0": 4,  # CPU/CPU/CPU
                            "Mode 2": 10,  # iGPU/iGPU/iGPU
                            "Mode 3": 14,  # iGPU/iGPU/NPU
                            "Mode 4": 18,  # iGPU/NPU/NPU
                            "Mode 7": 20,  # iGPU + NPU concurrent
                        },
                    }
                else:
                    self.config = {
                        "ref_stream_list": [8],
                        "ref_platform": "i7-1360p (16G Mem)",
                        "ref_gpu_freq": -1,
                        "ref_pkg_power": -1,
                        "models": ["yolov5m+resnet-50-tf+mobilenet-v2-pytorch"],
                        "modes": self.get_mode_and_compute_devices(available_devices, va_medium_executed_modes),
                        "mode_ref_streams": {
                            "Mode 0": 4,  # CPU/CPU/CPU
                            "Mode 2": 8,  # iGPU/iGPU/iGPU
                        },
                    }
        elif device_type == "dGPU":
            self.config = {
                "ref_stream_list": [22],
                "ref_platform": "ARL Ultra 9 285 + B580",
                "ref_gpu_freq": -1,
                "ref_pkg_power": -1,
                "models": ["yolov5m+resnet-50-tf+mobilenet-v2-pytorch"],
                "modes": self.get_mode_and_compute_devices(available_devices, va_medium_executed_modes),
                "mode_ref_streams": {
                    "Mode 0": 4,  # CPU/CPU/CPU
                    "Mode 1": 22,  # dGPU/dGPU/dGPU
                    "Mode 5": 26,  # dGPU/dGPU/NPU
                    "Mode 6": 30,  # dGPU/NPU/NPU
                    "Mode 8": 40,  # dGPU + NPU concurrent
                },
            }
        elif device_type == "CPU":
            self.config = {
                "ref_stream_list": [10],
                "ref_platform": "Xeon(R) Gold 6414U (128G Mem)",
                "ref_gpu_freq": -1,
                "ref_pkg_power": -1,
                "models": ["yolov5m+resnet-50-tf+mobilenet-v2-pytorch"],
                "modes": self.get_mode_and_compute_devices(available_devices, va_medium_executed_modes),
                "mode_ref_streams": {
                    "Mode 0": 10,  # CPU/CPU/CPU
                },
            }
        else:
            self.logger.error(
                f"Device type '{device_type}' is not supported as primary device. "
                "Supported primary devices: iGPU, dGPU, CPU"
            )
            raise ValueError(f"Unsupported device type: {device_type}")

    def _generate_single_pipeline(
        self,
        video_src,
        decode_elem,
        detect_elem,
        classify_elem,
        det_model_path,
        det_model_proc,
        cls1_model_path,
        cls1_model_proc,
        cls2_model_path,
        cls2_model_proc,
        starting_frame=200,
    ):
        """Generate a single VA Medium pipeline string."""
        pipeline = f"filesrc location={video_src} ! {decode_elem} ! queue ! "

        # Detection with YOLOv5m
        if det_model_proc:
            pipeline += f"gvadetect model={det_model_path} model-proc={det_model_proc} {detect_elem} ! queue ! "
        else:
            pipeline += f"gvadetect model={det_model_path} {detect_elem} ! queue ! "

        # Tracking with tracking-type=1 (short-term-imageless)
        pipeline += "gvatrack tracking-type=1 config=tracking_per_class=false ! queue ! "

        # First classification: ResNet-50
        if cls1_model_proc:
            pipeline += f"gvaclassify model={cls1_model_path} model-proc={cls1_model_proc} {classify_elem} model-instance-id=resnet50 ! queue ! "
        else:
            pipeline += f"gvaclassify model={cls1_model_path} {classify_elem} model-instance-id=resnet50 ! queue ! "

        # Second classification: MobileNet-v2
        if cls2_model_proc:
            pipeline += f"gvaclassify model={cls2_model_path} model-proc={cls2_model_proc} {classify_elem} model-instance-id=mobilenetv2 ! queue ! "
        else:
            pipeline += f"gvaclassify model={cls2_model_path} {classify_elem} model-instance-id=mobilenetv2 ! queue ! "

        # FPS counter
        pipeline += f"gvafpscounter starting-frame={starting_frame} ! "

        if self.enable_mqtt:
            pipeline += (
                f"gvametaconvert json-indent=2 ! "
                f"gvametapublish file-format=json file-path={video_src}_metadata_output.json ! "
            )

        pipeline += "fakesink sync=false async=false "
        return pipeline

    def gen_gst_command(self, stream, resolution=None, codec=None, bitrate=None, model_name=None):
        """
        Generate GStreamer pipeline command for VA Medium benchmark.

        Args:
            stream: Number of parallel streams
            resolution: Tuple of (decode_dev, detect_dev, classify_dev) or
                       (decode_dev, detect_dev, classify_dev, "NPU_CONCURRENT") for concurrent mode
            codec: Video codec (unused, always H265 for VA Medium)
            bitrate: Target bitrate (unused)
            model_name: Model name(s) like "yolov5m+resnet-50-tf+mobilenet-v2-pytorch"

        Returns:
            str: GStreamer command string
        """
        # VA Medium pipeline video source (H265 encoded)
        video_src = "/home/dlstreamer/sample_video/medium/video/apple.h265"

        if model_name is None:
            raise ValueError("model_name cannot be None for VA benchmark")

        # Parse model names: det_model+cls1_model+cls2_model
        model_parts = model_name.split("+")
        if len(model_parts) != 3:
            raise ValueError(f"VA Medium requires 3 models (det+cls1+cls2), got: {model_name}")

        det_model_name = model_parts[0]  # yolo11m
        cls1_model_name = model_parts[1]  # resnet-50-tf
        cls2_model_name = model_parts[2]  # mobilenet-v2-pytorch

        # Get model paths
        det_model_path = self.model_path[det_model_name]["det_model_path"]
        det_model_proc = self.model_path[det_model_name].get("det_proc_json_path", "")

        cls1_model_path = self.model_path[cls1_model_name]["cls_model_path"]
        cls1_model_proc = self.model_path[cls1_model_name].get("cls_proc_json_path", "")

        cls2_model_path = self.model_path[cls2_model_name]["cls_model_path"]
        cls2_model_proc = self.model_path[cls2_model_name].get("cls_proc_json_path", "")

        # Check if this is a concurrent mode (4-element tuple with NPU_CONCURRENT)
        is_concurrent = len(resolution) == 4 and resolution[3] == "NPU_CONCURRENT"

        gst_cmd = " "

        if is_concurrent:
            # Concurrent mode: Run GPU and NPU pipelines simultaneously
            # GPU pipeline uses: decode on GPU, detect/classify on GPU
            # NPU pipeline uses: decode on GPU (NPU can't decode), detect/classify on NPU
            gpu_device = resolution[0]  # iGPU or dGPU

            # Get GPU pipeline elements
            decode_elem_gpu = self.get_sub_elements("decode", gpu_device)
            detect_elem_gpu = self.get_sub_elements("detect", gpu_device, is_concurrent=True)
            classify_elem_gpu = self.get_sub_elements("classify", gpu_device, is_concurrent=True)

            # Get NPU pipeline elements (decode still on GPU since NPU can't decode)
            decode_elem_npu = self.get_sub_elements("decode", gpu_device)  # Use GPU for decode
            detect_elem_npu = self.get_sub_elements("detect", "NPU", is_concurrent=True)
            classify_elem_npu = self.get_sub_elements("classify", "NPU", is_concurrent=True)

            # Distribute streams: half GPU, half NPU (round down for GPU, remainder for NPU)
            gpu_streams = stream // 2
            npu_streams = stream - gpu_streams

            self.logger.info(f"[CONCURRENT] Generating {gpu_streams} GPU + {npu_streams} NPU streams")

            # Generate GPU pipelines (using starting-frame=2000 for concurrent mode)
            for i in range(gpu_streams):
                gst_cmd += self._generate_single_pipeline(
                    video_src,
                    decode_elem_gpu,
                    detect_elem_gpu,
                    classify_elem_gpu,
                    det_model_path,
                    det_model_proc,
                    cls1_model_path,
                    cls1_model_proc,
                    cls2_model_path,
                    cls2_model_proc,
                    starting_frame=200,
                )

            # Generate NPU pipelines
            for i in range(npu_streams):
                gst_cmd += self._generate_single_pipeline(
                    video_src,
                    decode_elem_npu,
                    detect_elem_npu,
                    classify_elem_npu,
                    det_model_path,
                    det_model_proc,
                    cls1_model_path,
                    cls1_model_proc,
                    cls2_model_path,
                    cls2_model_proc,
                    starting_frame=200,
                )
        else:
            # Standard mode: All streams use the same device configuration
            decode_elem = self.get_sub_elements("decode", resolution[0])
            detect_elem = self.get_sub_elements("detect", resolution[1])
            classify_elem = self.get_sub_elements("classify", resolution[2])

            # Determine starting_frame - video has 1080 frames, so use 200 for all modes
            # Original requirement was 200 for CPU, 2000 for GPU, but video is too short
            starting_frame = 200

            for i in range(stream):
                gst_cmd += self._generate_single_pipeline(
                    video_src,
                    decode_elem,
                    detect_elem,
                    classify_elem,
                    det_model_path,
                    det_model_proc,
                    cls1_model_path,
                    cls1_model_proc,
                    cls2_model_path,
                    cls2_model_proc,
                    starting_frame=starting_frame,
                )

        self.logger.debug(gst_cmd)
        return gst_cmd

    def report_csv(self, *args):
        """
        Report VA Medium benchmark results to CSV file.
        """
        tc_name = "VA Medium Pipeline"

        if not self.config_file_path:
            result, avg_fps, refvalue, duration, model_name, mode, devices = args

            # Handle concurrent mode (4-element tuple with NPU_CONCURRENT)
            if len(devices) == 4 and devices[3] == "NPU_CONCURRENT":
                gpu_device = devices[0]
                dev_str = f"{gpu_device}+NPU(Concurrent)"
            else:
                dev_labels = ["Dec", "Det", "Cls"]
                dev_str = "/".join(f"{dev}({lbl})" for dev, lbl in zip(devices, dev_labels))

            # Use mode-specific reference value if available
            ref_platform = self.config.get("ref_platform", "Unknown")
            mode_ref_value = self.config.get("mode_ref_streams", {}).get(mode, refvalue)
            ref_gpu_freq = self.config.get("ref_gpu_freq", -1)
            ref_pkg_power = self.config.get("ref_pkg_power", -1)

            # Telemetry collected
            cpu_freq, cpu_util, sys_mem, gpu_freq, eu_usage, vdbox_usage, pkg_power, gpu_power = self.telemetry_list

            prefix = f"{tc_name},{model_name},{mode},{dev_str},{result},{avg_fps}"
            prefix_esc = f"{tc_name},{model_name},{mode},{dev_str},{result},{avg_fps}"
            prefix_esc = prefix_esc.replace("+", "\\+")

            additional = f"{gpu_freq},{pkg_power},{ref_platform},{mode_ref_value},{ref_gpu_freq},{ref_pkg_power}"
            additional += f",{duration:.2f},No Error"

        else:
            raise RuntimeError(f"Running with config_file_path {self.config_file_path} not supported!")

        self.update_csv(self.csv_path, prefix_esc, prefix + "," + additional)

    def bpl_run_benchmark(self):
        """
        Run VA Medium benchmark for all applicable modes.
        """
        self.get_gst_elements("h265")  # Use H265 for medium pipeline

        # Debug: Show config state
        print(f"[DEBUG] bpl_run_benchmark - device: {self.device}", file=sys.stderr)
        print(
            f"[DEBUG] bpl_run_benchmark - config keys: {list(self.config.keys()) if hasattr(self, 'config') else 'NO CONFIG'}",
            file=sys.stderr,
        )
        print(f"[DEBUG] bpl_run_benchmark - modes: {self.config.get('modes', 'NOT FOUND')}", file=sys.stderr)
        print(f"[DEBUG] bpl_run_benchmark - models: {self.config.get('models', 'NOT FOUND')}", file=sys.stderr)

        # Calculate memory-based max stream limit to prevent OOM
        # Medium pipelines are more memory-intensive with dual classification
        is_igpu = self.device in ["iGPU", "GPU", "gpu"]
        log_memory_status(self.logger, prefix="[VA-Medium] ")
        max_stream = get_memory_based_max_streams(is_igpu=is_igpu, logger=self.logger)
        # Reduce max stream for medium pipeline due to higher memory per stream
        max_stream = max(1, int(max_stream * 0.75))
        self.logger.info(f"[VA-Medium] Using memory-based stream limit: {max_stream} (75% of base)")

        for j, (mode, devices) in enumerate(self.config["modes"].items()):
            # Force garbage collection between modes to prevent memory accumulation
            gc.collect()

            # Re-check memory before each mode to prevent OOM
            log_memory_status(self.logger, prefix=f"[VA-Medium {mode}] ")
            if not check_available_memory(min_available_gb=6.0, logger=self.logger):
                self.logger.warning(f"[VA-Medium] Insufficient memory to run {mode}, skipping remaining modes")
                break

            # Recalculate max streams based on current available memory (apply 75% reduction)
            max_stream = get_memory_based_max_streams(is_igpu=is_igpu, logger=self.logger)
            max_stream = max(1, int(max_stream * 0.75))

            for i, mod in enumerate(self.config["models"]):
                cur_ref_value = self.config["ref_stream_list"][i]
                self.telemetry_file = f"{self.telemetry_file_prefix}_{mod}_{self.device}.result"
                self.result_file = f"{self.result_file_prefix}_{mod}_{self.device}.result"

                self.logger.info(
                    f"Running {self.benchmark_name} using model: {mod}, "
                    f"input resolution: 1080p@{self.target_fps}, device: {self.device}"
                )

                start_time = time.time()
                result = self.run_test_round(
                    resolution=devices,
                    codec="h265",
                    ref_stream=cur_ref_value,
                    model_name=mod,
                    max_stream=max_stream,
                )
                end_time = time.time()
                duration = end_time - start_time

                # Handle result format
                if isinstance(result, str) and "@" in result:
                    avg_str, result_str = result.split("@")
                    avg_fps = float(avg_str)
                    result = int(result_str)
                elif isinstance(result, int):
                    result = result
                    avg_fps = 0.0
                else:
                    raise TypeError(f"Unexpected result type: {type(result)}, value: {result}")

                self.report_csv(result, avg_fps, cur_ref_value, duration, mod, mode, devices)

                self.logger.info(
                    f"{self.benchmark_name} execution [model: {mod}, device: {self.device}] "
                    f"finished in {duration:.2f} seconds"
                )

# Output paths inside container
OUTPUT_DIR = "/home/dlstreamer/output"
DETAIL_LOG_FILE_PREFIX = f"{OUTPUT_DIR}/va_medium_pipeline_runner"
RESULT_FILE_PREFIX = f"{OUTPUT_DIR}/va_medium_pipeline_runner"
TELEMETRY_FILE_PREFIX = f"{OUTPUT_DIR}/va_medium_pipeline_telemetry"

# Use custom CSV filename from environment variable if provided, otherwise default
CSV_FILENAME = os.environ.get("VA_CSV_FILENAME", "va_medium_pipeline.csv")
CSV_FILE_PREFIX = f"{OUTPUT_DIR}/{CSV_FILENAME.replace('.csv', '')}"

SINK_DIR = "/home/dlstreamer/sink"


def run_va_medium_benchmark(device, monitor_num, is_mtl, has_igpu, config_file):
    """
    Run VA Medium benchmark for specified device.

    Args:
        device: Device to run benchmark on
        monitor_num: Number of monitors
        is_mtl: Whether the device is MTL
        has_igpu: Whether the device has iGPU
        config_file: Optional custom config file path
    """
    print(
        f"[DEBUG] VA Medium Parameters: device={device}, monitor_num={monitor_num}, "
        f"is_mtl={is_mtl}, has_igpu={has_igpu}, config_file={config_file}",
        file=sys.stderr,
    )

    if config_file == "none":
        config_file = None
        print("[DEBUG] Config file set to None", file=sys.stderr)

    print("[DEBUG] Creating VAMediumBenchmark instance...", file=sys.stderr)
    va_runner = VAMediumBenchmark(
        name="VA Medium Benchmark",
        device=device,
        monitor_num=monitor_num,
        is_MTL=is_mtl,
        has_igpu=has_igpu,
        target_fps=30,
        telemetry_file_prefix=TELEMETRY_FILE_PREFIX,
        log_file=DETAIL_LOG_FILE_PREFIX,
        result_file_prefix=RESULT_FILE_PREFIX,
        csv_path=CSV_FILE_PREFIX,
        sink_dir=SINK_DIR,
        config_file_path=config_file,
    )
    print("[DEBUG] VAMediumBenchmark instance created", file=sys.stderr)

    va_runner.prepare()
    print("[DEBUG] va_runner.prepare() completed", file=sys.stderr)

    if config_file is None:
        va_runner.bpl_run_benchmark()
    else:
        print("[DEBUG] Running with config: calling run_pipeline_with_config()...", file=sys.stderr)
        va_runner.run_pipeline_with_config()

    print("[DEBUG] bpl_run_benchmark() completed", file=sys.stderr)


if __name__ == "__main__":
    print("=" * 80, file=sys.stderr)
    print("VA MEDIUM BENCHMARK PYTHON SCRIPT STARTED", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Script: {__file__}", file=sys.stderr)
    print(f"Arguments: {sys.argv}", file=sys.stderr)
    print(f"Current directory: {os.getcwd()}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    def str2bool(v):
        return v.lower() in ("true",)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help="Device to run benchmark on")
    parser.add_argument("--monitor_num", type=int, required=True, help="Number of monitors")
    parser.add_argument("--is_mtl", type=str2bool, required=True, help="Whether the device is MTL")
    parser.add_argument("--has_igpu", type=str2bool, required=True, help="Whether the device has iGPU")
    parser.add_argument("--config_file", type=str, required=False, default=None, help="Optional custom config file")

    print("[DEBUG] Parsing arguments...", file=sys.stderr)
    args = parser.parse_args()

    print("[DEBUG] Parsed arguments:", file=sys.stderr)
    print(f"  device: {args.device}", file=sys.stderr)
    print(f"  monitor_num: {args.monitor_num}", file=sys.stderr)
    print(f"  is_mtl: {args.is_mtl}", file=sys.stderr)
    print(f"  has_igpu: {args.has_igpu}", file=sys.stderr)
    print(f"  config_file: {args.config_file}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("[DEBUG] Calling run_va_medium_benchmark...", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    run_va_medium_benchmark(args.device, args.monitor_num, args.is_mtl, args.has_igpu, args.config_file)

    print("=" * 80, file=sys.stderr)
    print("[DEBUG] run_va_medium_benchmark completed", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
