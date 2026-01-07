# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import time

# Support both installed package and Docker container usage
try:
    from esq.utils.media.pipeline_utils import BaseDLBenchmark, BenchmarkLogger
except ModuleNotFoundError:
    # Inside container, use copied utilities
    from esq_utils.media.pipeline_utils import BaseDLBenchmark, BenchmarkLogger


class MediaBenchmark(BaseDLBenchmark):
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
        task,
        csv_path,
        enc_csv_path,
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
        )
        self.codecs = ["h264", "h265"]
        self.resolutions = ["1080p", "4K"]
        self.bitrates = {"h264": [4000, 16000], "h265": [2000, 8000]}
        self.task = task
        self.enc_csv_path = enc_csv_path
        self.display_width = 3840
        self.display_height = 2160
        self.resolution_dict = {
            "1080p": {
                "width": 1920,
                "height": 1080,
            },
            "4K": {"width": 3840, "height": 2160},
        }

        self.loop_video_commands = [
            ["bash", "gst_loop_mp4.sh", "12", "1080p", "h264"],
            ["bash", "gst_loop_mp4.sh", "12", "1080p", "h265"],
            ["bash", "gst_loop_mp4.sh", "12", "4K", "h264"],
            ["bash", "gst_loop_mp4.sh", "12", "4K", "h265"],
        ]

        BenchmarkLogger.configure(self.log_file)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_config_of_platform(self):
        # Device name is already normalized by test runner (iGPU, dGPU, CPU)
        # Get device type for config selection (strip index for dGPU.0, dGPU.1)
        device_type = self.device.split(".")[0] if "." in self.device else self.device

        if device_type == "iGPU":
            if self.VDBOX == 1:
                self.config = {
                    "Encode": {
                        "ref_stream_list": [10, 2, 14, 3],
                        "ref_gpu_freq_list": [-1, -1, -1],
                        "ref_pkg_power_list": [-1, -1, -1],
                    },
                    "Decode": {
                        "ref_stream_list": [43, 10, 55, 14],
                        "ref_gpu_freq_list": [-1, -1, -1],
                        "ref_pkg_power_list": [-1, -1, -1],
                    },
                    "Decode+Compose": {
                        "ref_stream_list": [10, 9, 16, 8],
                        "ref_gpu_freq_list": [-1, -1, -1],
                        "ref_pkg_power_list": [-1, -1, -1],
                        "compose_size": 4,
                    },
                    "ref_platform": "i5-12400 (16G Mem)",
                }
            else:
                if self.is_MTL:
                    self.config = {
                        "Encode": {
                            "ref_stream_list": [25, 5, 18, 3],
                            "ref_gpu_freq_list": [-1, -1, -1],
                            "ref_pkg_power_list": [-1, -1, -1],
                        },
                        "Decode": {
                            "ref_stream_list": [77, 23, 107, 28],
                            "ref_gpu_freq_list": [-1, -1, -1],
                            "ref_pkg_power_list": [-1, -1, -1],
                        },
                        "Decode+Compose": {
                            "ref_stream_list": [17, 4, 17, 4],
                            "ref_gpu_freq_list": [-1, -1, -1],
                            "ref_pkg_power_list": [-1, -1, -1],
                            "compose_size": 6,
                        },
                        "ref_platform": "MTL 165H (32G Mem)",
                    }
                else:
                    self.config = {
                        "Encode": {
                            "ref_stream_list": [17, 4, 27, 6],
                            "ref_gpu_freq_list": [-1, -1, -1],
                            "ref_pkg_power_list": [-1, -1, -1],
                        },
                        "Decode": {
                            "ref_stream_list": [87, 22, 108, 27],
                            "ref_gpu_freq_list": [-1, -1, -1],
                            "ref_pkg_power_list": [-1, -1, -1],
                        },
                        "Decode+Compose": {
                            "ref_stream_list": [18, 14, 17, 12],
                            "ref_gpu_freq_list": [-1, -1, -1],
                            "ref_pkg_power_list": [-1, -1, -1],
                            "compose_size": 6,
                        },
                        "ref_platform": "i7-1360P (16G Mem)",
                    }
        elif device_type == "dGPU":
            self.config = {
                "Encode": {
                    "ref_stream_list": [29, 6, 36, 8],
                    "ref_gpu_freq_list": [-1, -1, -1],
                    "ref_pkg_power_list": [-1, -1, -1],
                },
                "Decode": {
                    "ref_stream_list": [97, 25, 121, 31],
                    "ref_gpu_freq_list": [-1, -1, -1],
                    "ref_pkg_power_list": [-1, -1, -1],
                },
                "Decode+Compose": {
                    "ref_stream_list": [18, 19, 18, 18],
                    "ref_gpu_freq_list": [-1, -1, -1],
                    "ref_pkg_power_list": [-1, -1, -1],
                    "compose_size": 6,
                },
                "ref_platform": "Arc A380",
            }
        elif device_type == "CPU":
            self.config = {
                "Encode": {
                    "ref_stream_list": [75, 39, 34, 13],
                    "ref_gpu_freq_list": [-1, -1, -1],
                    "ref_pkg_power_list": [-1, -1, -1],
                },
                "Decode": {
                    "ref_stream_list": [458, 124, 325, 84],
                    "ref_gpu_freq_list": [-1, -1, -1],
                    "ref_pkg_power_list": [-1, -1, -1],
                },
                "Decode+Compose": {
                    "ref_stream_list": [36, 36, 36, 36],
                    "ref_gpu_freq_list": [-1, -1, -1],
                    "ref_pkg_power_list": [-1, -1, -1],
                    "compose_size": 6,
                },
                "ref_platform": "Xeon(R) Gold 6430 (512G Mem)",
            }

    def igpu_specified(self, codec):
        if self.device == "iGPU":
            if not self.is_MTL:
                self.enc_ele = f"vaapi{codec}enc"
                self.post_proc_ele = "vaapipostproc"
                os.environ["GST_VAAPI_DRM_DEVICE"] = "/dev/dri/renderD128"

    def gen_gst_command(self, stream, resolution=None, codec=None, bitrate=None, model_name=None):
        video_src = f"/home/dlstreamer/sample_video/car_{resolution}30_120s_{codec}.mp4"
        if self.enc_ele.startswith("vaapi"):
            enc_cmd = f"{self.enc_ele} rate-control=cbr bitrate={bitrate} quality-level=7 tune=low-power"
        else:
            enc_cmd = f"{self.enc_ele}  rate-control=cbr bitrate={bitrate}  target-usage=7 ref-frames=1"

        gst_cmd = ""
        if self.task == "Decode+Compose":
            compose_size = self.config[self.task]["compose_size"]
            cell_width = self.display_width // compose_size
            cell_height = self.display_height // compose_size
            gst_cmd += "compositor name=comp_0 "
            for i in range(stream):
                gst_cmd += f"sink_{i}::xpos={i % compose_size * cell_width} sink_{i}::ypos={i // compose_size * cell_height} sink_{i}::alpha=1 "
            if self.monitor_num == 0:
                if self.device == "CPU":
                    gst_cmd += f"! video/x-raw,format=I420,width={self.resolution_dict[resolution]['width']},height={self.resolution_dict[resolution]['height']} "
                    gst_cmd += f"! {self.enc_ele} bitrate={bitrate} speed-preset=superfast "
                    gst_cmd += f"! {codec}parse ! mp4mux ! filesink location=media_decode_compose_benchamrk_encoded.mp4 async=false sync=false "
                else:
                    gst_cmd += "! fakesink async=false sync=false "
            elif self.monitor_num == 1:
                gst_cmd += "! xvimagesink display=:0 async=false sync=false "
            else:
                gst_cmd += "! xvimagesink display=:0 async=false sync=false "
                gst_cmd += "compositor name=comp_1 "
                for i in range(stream):
                    gst_cmd += f"sink_{i}::xpos={i % compose_size * cell_width} sink_{i}::ypos={i // compose_size * cell_height} sink_{i}::alpha=1 "
                gst_cmd += "! xvimagesink display=:0 async=false sync=false "

        for i in range(stream):
            if self.task == "Encode":
                if self.device == "CPU":
                    enc_pattern = {"1080p": "snow", "4K": "black"}
                    gst_cmd += f"videotestsrc pattern={enc_pattern[resolution]} num-buffers={30 * 120} ! video/x-raw,format=I420,width={self.resolution_dict[resolution]['width']},height={self.resolution_dict[resolution]['height']},framerate=30/1 "
                    gst_cmd += "! gvafpscounter starting-frame=1000 "
                    gst_cmd += (
                        f"! {self.enc_ele} bitrate={bitrate} speed-preset=superfast ! fakesink async=false sync=false "
                    )
                else:
                    if self.post_proc_ele.startswith("vaapi"):
                        gst_cmd += f"videotestsrc pattern=snow num-buffers={30 * 120} ! {self.post_proc_ele} ! video/x-raw(memory:VASurface),width={self.resolution_dict[resolution]['width']},height={self.resolution_dict[resolution]['height']},framerate=30/1 "
                    else:
                        gst_cmd += f"videotestsrc pattern=snow num-buffers={30 * 120} ! {self.post_proc_ele} ! video/x-raw(memory:VAMemory),width={self.resolution_dict[resolution]['width']},height={self.resolution_dict[resolution]['height']},framerate=30/1 "
                    gst_cmd += f"! gvafpscounter starting-frame=1000 ! {enc_cmd} ! fakesink async=false sync=false "
            elif self.task == "Decode":
                if self.device == "CPU":
                    gst_cmd += f"filesrc location={video_src} ! multiqueue ! decodebin "
                    gst_cmd += "! gvafpscounter starting-frame=1000 ! fakesink async=false sync=false "
                else:
                    gst_cmd += f"filesrc location={video_src} ! qtdemux ! {codec}parse ! {self.dec_ele} "
                    gst_cmd += "! gvafpscounter starting-frame=1000 ! fakesink async=false sync=false "
            else:
                # decode + compose
                if self.monitor_num == 2:
                    gst_cmd += f"filesrc location={video_src} ! qtdemux ! {codec}parse ! {self.dec_ele} ! tee name=t{i} ! queue "
                    gst_cmd += f"! gvafpscounter starting-frame=1000 ! {self.post_proc_ele} scale-method=fast ! video/x-raw,width={cell_width},height={cell_height} ! comp_0.sink_{i} "
                    gst_cmd += f"t{i}. ! queue "
                    gst_cmd += f"! gvafpscounter starting-frame=1000 ! {self.post_proc_ele} scale-method=fast ! video/x-raw,width={cell_width},height={cell_height} ! comp_1.sink_{i} "
                else:
                    if self.device == "CPU":
                        gst_cmd += f"filesrc location={video_src} ! multiqueue ! decodebin "
                        gst_cmd += f"! gvafpscounter starting-frame=1000 ! videoscale ! video/x-raw,width={cell_width},height={cell_height} ! queue ! comp_0.sink_{i} "
                    else:
                        gst_cmd += f"filesrc location={video_src} ! qtdemux ! {codec}parse ! {self.dec_ele} "
                        gst_cmd += f"! gvafpscounter starting-frame=1000 ! {self.post_proc_ele} scale-method=fast ! video/x-raw,width={cell_width},height={cell_height} ! queue ! comp_0.sink_{i} "
        self.logger.debug(f"Generated gst command for {stream} streams:")
        self.logger.debug(gst_cmd)
        return gst_cmd

    def report_csv(self, *arg):
        result, resolution, codec, bitrate, ref_stream, ref_gpu_freq, ref_pkg_power, duration = arg

        match self.task:
            case "Decode+Compose":
                compose_size = self.config[self.task]["compose_size"]
                tc_name = f"Media Decode + {compose_size}*{compose_size} Compose Benchmark"
            case "Decode" | "Encode":
                tc_name = f"Media {self.task} Benchmark"
            case _:
                pass
        prefix = f"{tc_name},{self.device},{codec},{int(bitrate / 1000)}Mbps,{resolution}@30, "
        prefix += f"{self.monitor_num}, {result}"
        prefix_esc = prefix.replace("+", "\\+").replace("*", "\\*")

        # Telemetry list contains only 8 values: [cpu_freq, cpu_util, sys_mem, gpu_freq, eu_usage, vdbox_usage, pkg_power, gpu_power]
        cpu_freq, cpu_util, sys_mem, gpu_freq, eu_usage, vdbox_usage, pkg_power, gpu_power = self.telemetry_list

        # Build telemetry columns matching CSV header order:
        # CSV Header: Average FPS, CPU Avg, CPU Peak, Memory Avg, Memory Peak, GPU EU Avg, GPU EU Peak, GPU VDBox Avg, GPU VDBox Peak,
        #             GPU Memory Avg, GPU Memory Peak, SoC Power Avg, SoC Power Peak, GPU Power Avg, GPU Power Peak
        # Note: Average FPS is left as -1.00 (will be extracted from log file by test, not from telemetry)
        # telemetry_cols = f"-1.00,{cpu_util},-1.00,{sys_mem},-1.00,{eu_usage},-1.00,{vdbox_usage},-1.00,-1.00,-1.00,{pkg_power},-1.00,{gpu_power},-1.00"

        # Build additional columns: telemetry + reference platform info + duration + errors
        # additional = f"{telemetry_cols},{self.config['ref_platform']},{refvalue},{ref_gpu_freq},{ref_pkg_power},{duration:.2f},No Error"

        ref_platform = self.config.get("ref_platform", "Unknown")

        additional = f"{gpu_freq},{pkg_power},{ref_platform},{ref_stream},{ref_gpu_freq},{ref_pkg_power}"
        additional += f",{duration:.2f},No Error"

        if self.task == "Encode":
            self.update_csv(self.enc_csv_path, prefix_esc, prefix + "," + additional)
        else:
            self.update_csv(self.csv_path, prefix_esc, prefix + "," + additional)

    def run_benchmark(self, filter_codec=None, filter_bitrate=None, filter_resolution=None):
        """Run benchmark, optionally filtered to specific codec/bitrate/resolution.

        Args:
            filter_codec: If provided, only run tests for this codec (e.g., 'H.264')
            filter_bitrate: If provided, only run tests for this bitrate (e.g., '4Mbps')
            filter_resolution: If provided, only run tests for this resolution (e.g., '1080p30')
        """
        if self.task == "Decode+Compose":
            max_streams = self.config[self.task]["compose_size"] ** 2
        else:
            max_streams = -1
        ref_id = 0

        # Filter codecs if specified
        codecs_to_test = [filter_codec] if filter_codec else self.codecs

        for codec in codecs_to_test:
            # Skip if codec not in configured list
            if codec not in self.codecs:
                self.logger.warning(f"Codec {codec} not in configured codecs {self.codecs}, skipping")
                continue

            self.get_gst_elements(codec)
            self.igpu_specified(codec)

            # Filter resolutions if specified
            resolutions_to_test = [filter_resolution] if filter_resolution else self.resolutions

            for res in resolutions_to_test:
                # Skip if resolution not in configured list
                if res not in self.resolutions:
                    self.logger.warning(f"Resolution {res} not in configured resolutions {self.resolutions}, skipping")
                    continue

                # Get index for this resolution to look up bitrate
                res_index = self.resolutions.index(res)
                cur_bitrate = self.bitrates[codec][res_index]

                # Convert bitrate to string format for comparison
                # self.bitrates values are in kbps (e.g., 4000 kbps)
                # Convert to Mbps string: 4000 / 1000 = 4Mbps
                # Or keep as kbps string if filter is in kbps format
                if filter_bitrate and filter_bitrate.endswith("Mbps"):
                    # Filter is in Mbps format like "4Mbps"
                    bitrate_str = f"{int(cur_bitrate / 1000)}Mbps"
                else:
                    # Filter is in kbps format like "4000"
                    bitrate_str = str(cur_bitrate)

                # Skip if bitrate filter specified and doesn't match
                if filter_bitrate and bitrate_str != filter_bitrate:
                    self.logger.debug(
                        f"Skipping {codec} {res} - bitrate {bitrate_str} doesn't match filter {filter_bitrate}"
                    )
                    continue

                cur_ref_value = self.config[self.task]["ref_stream_list"][ref_id]
                # Get reference values from config (will be overwritten with actual telemetry data)
                cur_ref_gpu_freq = self.config[self.task]["ref_gpu_freq_list"][ref_id]
                cur_ref_pkg_power = self.config[self.task]["ref_pkg_power_list"][ref_id]

                self.telemetry_file = (
                    f"{self.telemetry_file_prefix}_{self.task}_{codec}_{cur_bitrate}_{res}_{self.device}.result"
                )
                self.result_file = (
                    f"{self.result_file_prefix}_{self.task}_{codec}_{cur_bitrate}_{res}_{self.device}.result"
                )
                self.logger.info(
                    f"Running Media {self.task} Benchmark, Input Codec: {codec}, Bitrate: {cur_bitrate}, Resolution: {res}, Device: {self.device}"
                )
                start_time = time.time()
                result = self.run_test_round(
                    resolution=res, codec=codec, bitrate=cur_bitrate, ref_stream=cur_ref_value, max_stream=max_streams
                )
                end_time = time.time()
                duration = end_time - start_time

                # Extract actual GPU Freq and Pkg Power from telemetry data collected during test
                # Telemetry list order: [0]=CPU Freq, [1]=CPU Usage, [2]=Mem Usage,
                #                        [3]=GPU Freq, [4]=EU Usage, [5]=VDBox Usage,
                #                        [6]=Pkg Power, [7]=GPU Power
                try:
                    if len(self.telemetry_list) >= 8:
                        actual_gpu_freq = float(self.telemetry_list[3])
                        actual_pkg_power = float(self.telemetry_list[6])
                        self.logger.info(
                            f"Using actual telemetry data - GPU Freq: {actual_gpu_freq:.2f} MHz, Pkg Power: {actual_pkg_power:.2f} W"
                        )
                    else:
                        self.logger.warning(
                            f"Insufficient telemetry data ({len(self.telemetry_list)} entries), using config values"
                        )
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to extract telemetry data: {e}, using config values")

                self.report_csv(
                    result, res, codec, cur_bitrate, cur_ref_value, cur_ref_gpu_freq, cur_ref_pkg_power, duration
                )
                self.logger.info(
                    f"Media {self.task} Benchmark execution [Input Codec: {codec}], Bitrate: {cur_bitrate}, Resolution: {res}, Device: {self.device} finished in {duration} seconds"
                )
                ref_id += 1


OUTPUT_DIR = "/home/dlstreamer/output"
DETAIL_LOG_FILE = f"{OUTPUT_DIR}/media_performance_benchmark_runner.log"
RESULT_FILE_PREFIX = f"{OUTPUT_DIR}/media_performance_benchmark_runner"
TELEMETRY_FILE_PREFIX = f"{OUTPUT_DIR}/media_performance_benchmark_telemetry"
CSV_FILE = f"{OUTPUT_DIR}/media_performance_benchmark.csv"
ENC_CSV_FILE = f"{OUTPUT_DIR}/media_encode_performance_benchmark.csv"

SINK_DIR = "/home/dlstreamer/sink"


def run_media_benchmark(device, monitor_num, is_mtl, has_igpu, operation, codec, bitrate, resolution):
    """Run media benchmark for a specific operation/codec/bitrate/resolution combination.

    Args:
        device: Device identifier (e.g., 'iGPU', 'dGPU.0')
        monitor_num: Number of monitors (0 for headless)
        is_mtl: Whether the device is MTL
        has_igpu: Whether the device has iGPU
        operation: Operation type ('encode', 'decode', or 'decode + compose')
        codec: Codec type (e.g., 'H.264', 'H.265')
        bitrate: Bitrate (e.g., '4Mbps', '16Mbps')
        resolution: Resolution (e.g., '1080p30', '4k@30')
    """
    print(
        f"run_media_benchmark(), device={device} operation={operation} codec={codec} bitrate={bitrate} resolution={resolution}"
    )

    # Map operation string to task name
    operation_lower = operation.lower()
    if operation_lower == "encode":
        task = "Encode"
    elif operation_lower == "decode":
        task = "Decode"
    elif operation_lower in ["decode + compose", "decode+compose", "compose"]:
        task = "Decode+Compose"
    else:
        raise ValueError(f"Unknown operation: {operation}")

    print(f"Running {task} benchmark for {codec} {bitrate} {resolution} on {device}")

    media_benchmark = MediaBenchmark(
        name="Media Benchmark",
        device=device,
        monitor_num=monitor_num,
        is_MTL=is_mtl,
        has_igpu=has_igpu,
        target_fps=30,
        telemetry_file_prefix=TELEMETRY_FILE_PREFIX,
        log_file=DETAIL_LOG_FILE,
        result_file_prefix=RESULT_FILE_PREFIX,
        task=task,
        csv_path=CSV_FILE,
        enc_csv_path=ENC_CSV_FILE,
    )
    media_benchmark.prepare()

    # Run benchmark for the specific codec/bitrate/resolution combination
    # Pass filter parameters to run only the requested test case
    media_benchmark.run_benchmark(filter_codec=codec, filter_bitrate=bitrate, filter_resolution=resolution)


if __name__ == "__main__":

    def str2bool(v):
        return v.lower() in ("true")

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help="Device to run benchmark on (e.g., GPU.0, GPU.1)")
    parser.add_argument("--monitor_num", type=int, required=True, help="Number of monitors (0 for headless)")
    parser.add_argument("--is_mtl", type=str2bool, required=True, help="Whether the device is MTL")
    parser.add_argument("--has_igpu", type=str2bool, required=True, help="Whether the device has iGPU")
    parser.add_argument("--operation", type=str, required=True, help="Operation type (encode/decode)")
    parser.add_argument("--codec", type=str, required=True, help="Codec type (H.264/H.265)")
    parser.add_argument("--bitrate", type=str, required=True, help="Bitrate (e.g., 4Mbps, 16Mbps)")
    parser.add_argument("--resolution", type=str, required=True, help="Resolution (e.g., 1080p30, 4k30)")
    args = parser.parse_args()
    print(
        f"main(), device={args.device} operation={args.operation} codec={args.codec} bitrate={args.bitrate} resolution={args.resolution}"
    )
    run_media_benchmark(
        args.device,
        args.monitor_num,
        args.is_mtl,
        args.has_igpu,
        args.operation,
        args.codec,
        args.bitrate,
        args.resolution,
    )
