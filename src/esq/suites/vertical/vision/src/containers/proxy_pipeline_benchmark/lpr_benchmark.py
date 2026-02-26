# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import sys

# Add consolidated utilities to path
sys.path.insert(0, "/home/dlstreamer")

from base_plbenchmark import BaseProxyPipelineBenchmark

lpr_compute_modes = {
    "Mode 0": ("CPU", "CPU", "CPU"),
    "Mode 1": ("dGPU", "dGPU", "dGPU"),
    "Mode 2": ("iGPU", "iGPU", "iGPU"),
    "Mode 3": ("iGPU", "iGPU", "NPU"),
    "Mode 4": ("iGPU", "NPU", "NPU"),
    "Mode 5": ("dGPU", "dGPU", "NPU"),
    "Mode 6": ("dGPU", "NPU", "NPU"),
}
lpr_execd_modes = set()


class LPRBenchmark(BaseProxyPipelineBenchmark):
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

        self.loop_video_commands = [["bash", "gst_lpr_loop_mp4.sh", "2", "1280p", "h264", "30"]]

    def get_sub_elemments(self, stage, dev):
        det_opt = "nireq=2 batch-size=4 inference-interval=3 model-instance-id=detect0 "
        cls_opt = "nireq=2 batch-size=4 inference-interval=3 model-instance-id=lpr0 "
        # This maps device to plugin element names
        if stage == "decode":
            return {
                "CPU": "qtdemux ! h264parse ! avdec_h264 ! video/x-raw ",
                "iGPU": "qtdemux ! h264parse ! vah264dec ! video/x-raw\\(memory:VAMemory\\) ",
                "dGPU": "qtdemux ! h264parse ! varenderD129h264dec ! video/x-raw\\(memory:VAMemory\\) ",
                "NPU": "unsupported",  # Typically no decode on NPU
            }.get(dev, "unknown_decode")

        elif stage == "detect":
            # Determine correct OpenVINO device ID for dGPU
            dgpu_device = self._get_dgpu_openvino_device()
            return {
                "CPU": f"device=CPU pre-process-backend=opencv {det_opt}",
                "iGPU": f"device=GPU pre-process-backend=va-surface-sharing {det_opt}",
                "dGPU": f"device={dgpu_device} pre-process-backend=va-surface-sharing {det_opt}",
                "NPU": f"device=NPU pre-process-backend=opencv {det_opt}",
            }.get(dev, "unknown_detect")

        elif stage == "classify":
            # Determine correct OpenVINO device ID for dGPU
            dgpu_device = self._get_dgpu_openvino_device()
            return {
                "CPU": f"device=CPU pre-process-backend=opencv {cls_opt}",
                "iGPU": f"device=GPU pre-process-backend=va-surface-sharing {cls_opt}",
                "dGPU": f"device={dgpu_device} pre-process-backend=va-surface-sharing {cls_opt}",
                "NPU": f"device=NPU pre-process-backend=opencv {cls_opt}",
            }.get(dev, "unknown_classify")
        return "unknown_element"

    def get_normalize_device_name(self, dev):
        """Remove suffix like '.0', '.1' specifically for dGPU.0/1 to get canonical device name."""
        return re.sub(r"\.\d+$", "", dev)

    def _get_dgpu_openvino_device(self):
        """Calculate the correct OpenVINO device ID for discrete GPU."""
        # Extract dGPU index from device name (e.g., dGPU.0 -> 0, dGPU.1 -> 1)
        device_parts = self.device.split(".")
        if len(device_parts) > 1:
            dgpu_idx = int(device_parts[1])
        else:
            dgpu_idx = 0

        # OpenVINO enumeration: GPU/GPU.0 = iGPU (if present), GPU.1+ = dGPUs
        if self.has_igpu:
            # iGPU takes GPU.0, so first dGPU is GPU.1, second is GPU.2, etc.
            openvino_device = f"GPU.{dgpu_idx + 1}"
        else:
            # No iGPU, so first dGPU is GPU.0, second is GPU.1, etc.
            openvino_device = f"GPU.{dgpu_idx}"

        return openvino_device

    def get_mode_and_compute_devices(self, available_devices, lpr_execd_modes):
        modes_dict = {}
        normalized_curr_dev = self.get_normalize_device_name(self.device)
        allowed_devices = set(available_devices)
        allowed_devices.add("CPU")  # TODO: Enabled for testing, it will be removed

        for mode_name, devices in lpr_compute_modes.items():
            # Skip if already executed
            if mode_name in lpr_execd_modes:
                continue

            # Normalize devices for comparison
            normalized_devices = [self.get_normalize_device_name(d) for d in devices]

            # Check if all devices are allowed AND current device is in the mode
            if all(d in allowed_devices for d in normalized_devices) and normalized_curr_dev in normalized_devices:
                modes_dict[mode_name] = list(devices)
                lpr_execd_modes.add(mode_name)  # Mark as executed
        return modes_dict

    def get_available_devices(self):
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
        filtered_modes = []
        available_devices = self.get_available_devices()

        # Device name is already normalized by test runner (iGPU, dGPU, CPU, NPU)
        # Get device type for config selection (strip index for dGPU.0, dGPU.1)
        device_type = self.device.split(".")[0] if "." in self.device else self.device

        if device_type == "iGPU":
            if self.VDBOX == 1:
                self.config = {
                    "ref_stream_list": [4, 7],
                    "ref_platform": "i5-13600 (32G Mem)",
                    "ref_gpu_freq_list": [-1, 1336.37],
                    "ref_pkg_power_list": [-1, 36.1],
                    "models": ["yolov8_license_plate_detector+ch_PP-OCRv4_rec_infer"],
                    "modes": self.get_mode_and_compute_devices(available_devices, lpr_execd_modes),
                    # Mode-specific reference values for graph visualization
                    # Note: i5-12400 does not have NPU, so only CPU and iGPU modes
                    "mode_ref_streams": {
                        "Mode 0": 4,  # CPU/CPU/CPU
                        "Mode 2": 7,  # iGPU/iGPU/iGPU
                    },
                    "mode_ref_gpu_freq": {
                        "Mode 0": -1,  # CPU/CPU/CPU
                        "Mode 2": 1636.37,  # iGPU/iGPU/iGPU
                    },
                    "mode_ref_pkg_power": {
                        "Mode 0": -1,  # CPU/CPU/CPU
                        "Mode 2": 11.96,  # iGPU/iGPU/iGPU
                    },
                }
            else:
                if self.is_MTL:
                    self.config = {
                        "ref_stream_list": [6, 12, 13, 5],
                        "ref_platform": "MTL 165H (32G Mem)",
                        "ref_gpu_freq_list": [-1, 1153.45, 1177.11, 243.96],
                        "ref_pkg_power_list": [-1, 25.41, 26.23, 18.77],
                        "models": ["yolov8_license_plate_detector+ch_PP-OCRv4_rec_infer"],
                        "modes": self.get_mode_and_compute_devices(available_devices, lpr_execd_modes),
                        # Mode-specific reference values for graph visualization
                        # MTL 165H has iGPU and NPU support
                        "mode_ref_streams": {
                            "Mode 0": 6,  # CPU/CPU/CPU
                            "Mode 2": 12,  # iGPU/iGPU/iGPU
                            "Mode 3": 13,  # iGPU/iGPU/NPU
                            "Mode 4": 5,  # iGPU/NPU/NPU
                        },
                        "mode_ref_gpu_freq": {
                            "Mode 0": -1,  # CPU/CPU/CPU
                            "Mode 2": 1153.45,  # iGPU/iGPU/iGPU
                            "Mode 3": 1177.11,  # iGPU/iGPU/NPU
                            "Mode 4": 243.96,  # iGPU/NPU/NPU
                        },
                        "mode_ref_pkg_power": {
                            "Mode 0": -1,  # CPU/CPU/CPU
                            "Mode 2": 25.41,  # iGPU/iGPU/iGPU
                            "Mode 3": 26.23,  # iGPU/iGPU/NPU
                            "Mode 4": 18.77,  # iGPU/NPU/NPU
                        },
                    }
                else:
                    self.config = {
                        "ref_stream_list": [5, 12],
                        "ref_platform": "i7-1360p (16G Mem)",
                        "ref_gpu_freq_list": [-1, 1396.09],
                        "ref_pkg_power_list": [-1, 36.10],
                        "models": ["yolov8_license_plate_detector+ch_PP-OCRv4_rec_infer"],
                        "modes": self.get_mode_and_compute_devices(available_devices, lpr_execd_modes),
                        # Mode-specific reference values for graph visualization
                        # Note: i7-1360p does not have NPU, so only CPU and iGPU modes
                        "mode_ref_streams": {
                            "Mode 0": 5,  # CPU/CPU/CPU
                            "Mode 2": 12,  # iGPU/iGPU/iGPU
                        },
                        "mode_ref_gpu_freq": {
                            "Mode 0": -1,  # CPU/CPU/CPU
                            "Mode 2": 1396.09,  # iGPU/iGPU/iGPU
                        },
                        "mode_ref_pkg_power": {
                            "Mode 0": -1,  # CPU/CPU/CPU
                            "Mode 2": 36.10,  # iGPU/iGPU/iGPU
                        },
                    }
        elif device_type == "dGPU":
            self.config = {
                "ref_stream_list": [6, 14, 15, 14],
                "ref_platform": "ARL Ultra 9 285 + B580",
                "ref_gpu_freq_list": [-1, 1131.87, 1128.87, 258.76],
                "ref_pkg_power_list": [-1, 30.04, 30.27, 29.83],
                "models": ["yolov8_license_plate_detector+ch_PP-OCRv4_rec_infer"],
                "modes": self.get_mode_and_compute_devices(available_devices, lpr_execd_modes),
                # Mode-specific reference values for graph visualization
                # ARL Ultra 9 285 + B580 has dGPU and NPU support
                "mode_ref_streams": {
                    "Mode 0": 6,  # CPU/CPU/CPU
                    "Mode 1": 14,  # dGPU/dGPU/dGPU
                    "Mode 5": 15,  # dGPU/dGPU/NPU
                    "Mode 6": 14,  # dGPU/NPU/NPU
                },
                "mode_ref_gpu_freq": {
                    "Mode 0": -1,  # CPU/CPU/CPU
                    "Mode 1": 1131.87,  # dGPU/dGPU/dGPU
                    "Mode 5": 1128.87,  # dGPU/dGPU/NPU
                    "Mode 6": 258.76,  # dGPU/NPU/NPU
                },
                "mode_ref_pkg_power": {
                    "Mode 0": -1,  # CPU/CPU/CPU
                    "Mode 1": 30.04,  # dGPU/dGPU/dGPU
                    "Mode 5": 30.27,  # dGPU/dGPU/NPU
                    "Mode 6": 29.83,  # dGPU/NPU/NPU
                },
            }
        elif device_type == "CPU":
            self.config = {
                "ref_stream_list": [14],
                "ref_platform": "Xeon(R) Gold 6414U (128G Mem)",
                "ref_gpu_freq_list": [-1],
                "ref_pkg_power_list": [-1],
                "models": ["yolov8_license_plate_detector+ch_PP-OCRv4_rec_infer"],
                "modes": self.get_mode_and_compute_devices(available_devices, lpr_execd_modes),
                # Mode-specific reference values for graph visualization
                "mode_ref_streams": {
                    "Mode 0": 14,  # CPU/CPU/CPU
                },
                "mode_ref_gpu_freq": {
                    "Mode 0": -1,  # CPU/CPU/CPU
                },
                "mode_ref_pkg_power": {
                    "Mode 0": -1,  # CPU/CPU/CPU
                },
            }
        else:
            # NPU or unknown device - not supported as primary device
            self.logger.error(
                f"Device type '{device_type}' is not supported as primary device. "
                "NPU should be used as co-processor with iGPU/dGPU (use is_mtl flag). "
                "Supported primary devices: iGPU, dGPU, CPU"
            )
            raise ValueError(f"Unsupported device type: {device_type}")

    def gen_gst_command(self, stream, resolution=None, codec=None, bitrate=None, model_name=None):
        video_src = "/home/dlstreamer/sample_video/lpr/ParkingVideo_1min.mp4"

        # Handle None model_name
        if model_name is None:
            raise ValueError("model_name cannot be None for LPR benchmark")

        if "+" in model_name:
            det_model_name, cls_model_name = model_name.split("+")
        else:
            det_model_name = model_name
            cls_model_name = None

        decode_elem = self.get_sub_elemments("decode", resolution[0])
        detect_elem = self.get_sub_elemments("detect", resolution[1])
        classify_elem = self.get_sub_elemments("classify", resolution[2])

        det_model_path = self.model_path[det_model_name]["det_model_path"]

        if cls_model_name:
            cls_model_path = self.model_path[cls_model_name]["cls_model_path"]

        gst_cmd = " "

        for i in range(stream):
            gst_cmd += f"filesrc location={video_src} ! {decode_elem} ! gvafpscounter starting-frame=50 ! queue ! gvadetect model={det_model_path} {detect_elem} ! queue ! "
            gst_cmd += "gvatrack tracking-type=short-term-imageless ! "
            if cls_model_name:
                gst_cmd += f"gvaclassify model={cls_model_path} {classify_elem} ! queue ! "
            if self.enable_mqtt:
                gst_cmd += f"gvametaconvert json-indent=2 ! gvametapublish file-format=json file-path={video_src}_metadata_output.json ! "
            gst_cmd += "fakesink sync=false "

        self.logger.debug(gst_cmd)
        return gst_cmd

    def report_csv(self, *arg):
        tc_name = "LPR Pipeline"

        if not self.config_file_path:
            dev_labels = ["Dec", "Det", "Cls"]
            result, avg_fps, refvalue, duration, model_name, mode, devices = arg
            dev_str = "/".join(f"{dev}({lbl})" for dev, lbl in zip(devices, dev_labels))

            # Use mode-specific reference values if available
            ref_platform = self.config.get("ref_platform", "Unknown")
            mode_ref_value = self.config.get("mode_ref_streams", {}).get(mode, refvalue)
            mode_ref_gpu_freq = self.config.get("mode_ref_gpu_freq", {}).get(mode, -1)
            mode_ref_pkg_power = self.config.get("mode_ref_pkg_power", {}).get(mode, -1)

            # Telmetry collected
            cpu_freq, cpu_util, sys_mem, gpu_freq, eu_usage, vdbox_usage, pkg_power, gpu_power = self.telemetry_list

            prefix = f"{tc_name},{model_name},{mode},{dev_str},{result},{avg_fps}"
            prefix_esc = f"{tc_name},{model_name},{mode},{dev_str},{result},{avg_fps}"
            prefix_esc = prefix_esc.replace("+", "\\+")

            additional = (
                f"{gpu_freq},{pkg_power},{ref_platform},{mode_ref_value},{mode_ref_gpu_freq},{mode_ref_pkg_power}"
            )
            additional += f",{duration:.2f},No Error"

        else:
            raise RuntimeError("Running with config_file_path {self.config_file_path} not supported!")

        self.update_csv(self.csv_path, prefix_esc, prefix + "," + additional)


OUTPUT_DIR = "/home/dlstreamer/output"
DETAIL_LOG_FILE_PREFIX = f"{OUTPUT_DIR}/lpr_proxy_pipeline_runner"
RESULT_FILE_PREFIX = f"{OUTPUT_DIR}/lpr_proxy_pipeline_runner"
TELEMETRY_FILE_PREFIX = f"{OUTPUT_DIR}/lpr_proxy_pipeline_telemetry"
CSV_FILE_PREFIX = f"{OUTPUT_DIR}/lpr_proxy_pipeline"

SINK_DIR = "/home/dlstreamer/sink"


def run_lpr_benchmark(device, monitor_num, is_mtl, has_igpu, config_file):
    import sys

    print(
        f"[DEBUG] LPR Parameters: device={device}, monitor_num={monitor_num}, is_mtl={is_mtl}, has_igpu={has_igpu}, config_file={config_file}",
        file=sys.stderr,
    )

    if config_file == "none":
        config_file = None
        print("[DEBUG] Config file set to None", file=sys.stderr)

    print("[DEBUG] Creating LPRBenchmark instance...", file=sys.stderr)
    lpr_runner = LPRBenchmark(
        name="LPR Benchmark",
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
    print("[DEBUG] LPRBenchmark instance created", file=sys.stderr)

    lpr_runner.prepare()
    print("[DEBUG] lpr_runner.prepare() completed", file=sys.stderr)

    if config_file is None:
        lpr_runner.bpl_run_lpr_benchmark()
    else:
        print("[DEBUG] Running with config: calling run_pipeline_with_config()...", file=sys.stderr)
        lpr_runner.run_pipeline_with_config()
    print("[DEBUG] bpl_run_lpr_benchmark() completed", file=sys.stderr)


if __name__ == "__main__":
    import sys

    print("=" * 80, file=sys.stderr)
    print("LPR BENCHMARK PYTHON SCRIPT STARTED", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Script: {__file__}", file=sys.stderr)
    print(f"Arguments: {sys.argv}", file=sys.stderr)
    print(f"Current directory: {os.getcwd()}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    def str2bool(v):
        return v.lower() in ("true")

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help="Device to run benchmark on")
    parser.add_argument("--monitor_num", type=int, required=True, help="Number of monitors")
    parser.add_argument("--is_mtl", type=str2bool, required=True, help="Whether the device is MTL")
    parser.add_argument("--has_igpu", type=str2bool, required=True, help="Whether the device has iGPU")
    parser.add_argument(
        "--config_file", type=str, required=False, default=None, help="Whether to run pipeline with user config"
    )

    print("[DEBUG] Parsing arguments...", file=sys.stderr)
    args = parser.parse_args()

    print("[DEBUG] Parsed arguments:", file=sys.stderr)
    print(f"  device: {args.device}", file=sys.stderr)
    print(f"  monitor_num: {args.monitor_num}", file=sys.stderr)
    print(f"  is_mtl: {args.is_mtl}", file=sys.stderr)
    print(f"  has_igpu: {args.has_igpu}", file=sys.stderr)
    print(f"  config_file: {args.config_file}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("[DEBUG] Calling run_lpr_benchmark...", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    run_lpr_benchmark(args.device, args.monitor_num, args.is_mtl, args.has_igpu, args.config_file)

    print("=" * 80, file=sys.stderr)
    print("[DEBUG] run_lpr_benchmark completed", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
