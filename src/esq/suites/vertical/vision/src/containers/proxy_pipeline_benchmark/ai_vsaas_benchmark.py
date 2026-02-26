import argparse
import sys

# Add consolidated utilities to path
sys.path.insert(0, "/home/dlstreamer")

from base_plbenchmark import BaseProxyPipelineBenchmark

class AIVSaaSBenchmark(BaseProxyPipelineBenchmark):
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

        self.loop_video_commands = [["bash", "gst_loop_mp4.sh", "18", "1080p", "h264", "30"]]

    def get_config_of_platform(self):
        # Device name is already normalized by test runner (iGPU, dGPU, CPU, NPU)
        # Get device type for config selection (strip index for dGPU.0, dGPU.1)
        device_type = self.device.split(".")[0] if "." in self.device else self.device

        if device_type == "iGPU":
            if self.VDBOX == 1:
                # Reference values from i5-12400 benchmarking (1 VDBox)
                self.config = {
                    "ref_stream_list": [63,6,2],
                    "ref_gpu_freq_list": [1251.51,1361.6,1437.98],
                    "ref_pkg_power_list": [32.1,34.65,29.14],
                    "ref_platform": "i5-13600 (32G Mem)",
                    "models": ["yolov5s-416", "yolov5m-416"],
                    "enc_flag": "rate-control=cbr bitrate=2000 target-usage=7",
                    "preproc_backend": "pre-process-backend=va-surface-sharing scale-method=fast",
                    "max_binary_search_start": 6,
                }
            else:
                if self.is_MTL:
                    # Reference values from MTL 165H benchmarking (for CSV comparison reports)
                    # Note: Binary search uses max_binary_search_start to cap initial high value
                    # to prevent timeout issues, while keeping full ref values for reporting
                    self.config = {
                        "ref_stream_list": [18, 19, 5],
                        "ref_gpu_freq_list": [1421.87, 1419.27, 852.81],
                        "ref_pkg_power_list": [27.80, 28.11, 23.50],
                        "ref_platform": "MTL 165H (32G Mem)",
                        "models": ["yolov5s-416", "yolov5m-416", "yolov5m-416+efficientnet-b0"],
                        "enc_flag": "rate-control=cbr bitrate=2000 target-usage=7",
                        "preproc_backend": "pre-process-backend=va-surface-sharing scale-method=fast",
                        # Cap binary search starting point to avoid timeout on slower platforms
                        # Binary search starts at min(ref_stream, max_binary_search_start)
                        "max_binary_search_start": 8,
                    }
                else:
                    self.config = {
                        "ref_stream_list": [14, 13, 6],
                        "ref_gpu_freq_list": [1251.51, 1361.60, 1437.98],
                        "ref_pkg_power_list": [32.10, 34.65, 29.14],
                        "ref_platform": "i7-1360p (16G Mem)",
                        "models": ["yolov5s-416", "yolov5m-416", "yolov5m-416+efficientnet-b0"],
                        "enc_flag": "rate-control=cbr bitrate=2000 target-usage=7",
                        "preproc_backend": "pre-process-backend=va-surface-sharing scale-method=fast",
                        "max_binary_search_start": 8,
                    }
        elif device_type == "dGPU":
            self.config = {
                "ref_stream_list": [35, 10, 3],
                "ref_gpu_freq_list": [1020.50, 1208.22, 1150.00],
                "ref_pkg_power_list": [31.51, 35.62, 38.00],
                "ref_platform": "Arc™ B-Series B580",
                "models": ["yolov5s-416", "yolov5m-416", "yolov5m-416+efficientnet-b0"],
                "enc_flag": "rate-control=cbr bitrate=2000 target-usage=7",
                "preproc_backend": "pre-process-backend=va-surface-sharing scale-method=fast",
                "max_binary_search_start": 12,  # dGPU can handle higher starting point
            }
        elif device_type == "CPU":
            self.config = {
                "ref_stream_list": [22, 19, 7],
                "ref_gpu_freq_list": [-1, -1, -1],
                "ref_pkg_power_list": [-1, -1, -1],
                "ref_platform": "Xeon(R) Gold 6430 (512G Mem)",
                "models": ["yolov5s-416", "yolov5m-416", "yolov5m-416+efficientnet-b0"],
                "enc_flag": "bitrate=2000 speed-preset=superfast",
                "preproc_backend": "",
                "max_binary_search_start": 10,  # CPU has many cores, can start higher
            }
        else:
            # NPU or unknown device - not supported as primary device
            self.logger.error(
                f"Device type '{device_type}' is not supported as primary device. "
                "NPU should be used as co-processor with iGPU (use is_mtl flag). "
                "Supported primary devices: iGPU, dGPU, CPU"
            )
            raise ValueError(f"Unsupported device type: {device_type}")

    def gen_gst_command(self, stream, resolution=None, codec=None, bitrate=None, model_name=None):
        video_src = f"/home/dlstreamer/sample_video/car_{resolution}30_180s_{codec}.mp4"
        gst_cmd = ""

        preproc_backend = self.config["preproc_backend"]
        enc_flag = self.config["enc_flag"]

        if "+" in model_name:
            det_model_name, cls_model_name = model_name.split("+")
        else:
            det_model_name = model_name
            cls_model_name = None

        # Device name is already normalized (iGPU, dGPU, CPU, NPU)
        # Get device type for config selection (strip index for dGPU.0, dGPU.1)
        device_type = self.device.split(".")[0] if "." in self.device else self.device

        if device_type == "CPU":
            inf_device = "CPU"
        elif device_type == "iGPU":
            inf_device = "GPU"
        elif device_type == "dGPU":
            if self.has_igpu:
                inf_device = f"GPU.{self.dgpu_idx + 1}"
            else:
                inf_device = f"GPU.{self.dgpu_idx}"
        else:
            # NPU or unknown device type - not supported as primary device
            self.logger.error(
                f"Device type '{device_type}' not supported as primary device. "
                "NPU should be used as co-processor with iGPU (use is_mtl flag)."
            )
            raise ValueError(f"Unsupported device type: {device_type}")

        det_model_path = self.model_path[det_model_name]["det_model_path"]
        det_proc_json_path = self.model_path[det_model_name]["det_proc_json_path"]
        if cls_model_name:
            cls_model_path = self.model_path[cls_model_name]["cls_model_path"]
            cls_proc_json_path = self.model_path[cls_model_name]["cls_proc_json_path"]

        # Optimize inference parameters based on device type and model complexity
        # For combo models (detect+classify), use conservative parameters to avoid GPU memory conflicts
        if device_type == "dGPU":
            if cls_model_name:
                detect_params = "batch-size=1 nireq=2 ie-config=NUM_STREAMS=2"
            else:
                detect_params = "batch-size=2 nireq=4 ie-config=NUM_STREAMS=4"
        else:
            detect_params = "batch-size=1 nireq=2 ie-config=NUM_STREAMS=2"

        for i in range(stream):
            gst_cmd += f"filesrc location={video_src} ! qtdemux ! h264parse ! tee name=t{i} ! queue ! mp4mux ! filesink async=false sync=false location={self.sink_dir}/vsaas_gateway_with_storage_and_ai_proxy_pipeline_local_storage_stream{i}.mp4 "

            # Build per-stream detection and tracking commands
            # Each stream needs its own model instance for parallel processing
            detect_cmd = f"gvadetect model={det_model_path} {preproc_backend} model-proc={det_proc_json_path} {detect_params} inference-interval=3 model-instance-id=detect{i} device={inf_device} threshold=0.5 "
            tracking_cmd = "gvatrack tracking-type=short-term-imageless "

            if device_type == "CPU":
                gst_cmd += f"t{i}. ! queue ! decodebin ! tee name=inf{i} "
                gst_cmd += f"! queue ! videoscale ! video/x-raw,format=I420,width=1920,height=1080 ! {self.enc_ele} {enc_flag} ! h265parse ! mp4mux ! filesink location={self.sink_dir}/vsaas_gateway_with_storage_and_ai_proxy_pipeline_encode_storage_stream{i}.mp4 sync=false async=false "
                gst_cmd += f"inf{i}. ! queue ! {detect_cmd} ! {tracking_cmd} "
            else:
                # For dual-model combinations on GPU, skip encode storage sink to avoid VA-API buffer conflicts
                # This reduces pipeline complexity and prevents "invalid VABufferID" errors
                if cls_model_name:
                    # Simplified pipeline: decode -> inference only (no encode storage)
                    gst_cmd += f"t{i}. ! queue ! {self.dec_ele} ! video/x-raw(memory:VAMemory) ! {detect_cmd} ! {tracking_cmd} "
                else:
                    # Standard pipeline: decode -> encode storage + inference
                    gst_cmd += f"t{i}. ! queue ! {self.dec_ele} ! tee name=inf{i} "
                    gst_cmd += f"! multiqueue ! video/x-raw(memory:VAMemory),width=1920,height=1080 ! {self.enc_ele} {enc_flag} ! h265parse ! mp4mux ! filesink async=false sync=false location={self.sink_dir}/vsaas_gateway_with_storage_and_ai_proxy_pipeline_encode_storage_stream{i}.mp4 "
                    gst_cmd += f"inf{i}. ! multiqueue ! video/x-raw(memory:VAMemory) ! {detect_cmd} ! {tracking_cmd} "

            if cls_model_name:
                if self.is_MTL and device_type == "iGPU":
                    # NPU has limited throughput - use higher inference-interval to reduce load
                    # This allows more concurrent streams while maintaining 30 FPS video processing
                    gst_cmd += f"! gvaclassify model={cls_model_path} {preproc_backend} model-proc={cls_proc_json_path} batch-size=1 nireq=2 inference-interval=15 inference-region=roi-list model-instance-id=classify{i} device=NPU "
                else:
                    # Use conservative classification parameters to avoid GPU memory conflicts
                    classify_params = "batch-size=1 nireq=2 ie-config=NUM_STREAMS=2"
                    gst_cmd += f"! gvaclassify model={cls_model_path} {preproc_backend} model-proc={cls_proc_json_path} {classify_params} inference-interval=3 inference-region=roi-list model-instance-id=classify{i} device={inf_device} "

            if self.enable_mqtt:
                gst_cmd += f"! gvametaconvert format=json json-indent=4 source={video_src} add-empty-results=true ! gvametapublish method=mqtt address={self.mqtt_address} mqtt-client-id=client{i} topic={self.mqtt_topic} "
            else:
                gst_cmd += f"! gvametaconvert format=json json-indent=4 source={video_src} add-empty-results=true ! gvametapublish method=file file-path=/dev/null "

            # Use lower starting-frame for complex pipelines to prevent hang/timeout issues
            # - Dual-model combinations: Always use 300 (original fix for A750)
            # - High stream count (>5): Use 300 as iGPU may struggle with many concurrent streams
            # - Single-model with encode storage: Use 500 (encode adds latency)
            # - Simple single-model: Use 1000 (original behavior)
            if cls_model_name:
                starting_frame = 300  # Dual-model warmup takes longer
            elif stream > 5:
                starting_frame = 300  # High stream count on iGPU needs faster feedback
            elif device_type != "CPU" and not cls_model_name:
                starting_frame = 500  # Single-model with encode storage adds latency
            else:
                starting_frame = 1000  # CPU or simple cases
            gst_cmd += f"! gvafpscounter starting-frame={starting_frame} ! fakesink async=false sync=false "

        self.logger.debug(f"GStreamer command: {gst_cmd}")

        return gst_cmd

    def report_csv(self, *arg):
        tc_name = "AI VSaaS Gateway Pipeline"

        if not self.config_file_path:
            result, ref_value, actual_gpu_freq, actual_pkg_power, duration, model_name = arg
            ref_platform = self.config.get("ref_platform", "Unknown")

            # Get reference values from config for comparison
            model_index = self.config["models"].index(model_name)
            ref_gpu_freq = self.config["ref_gpu_freq_list"][model_index]
            ref_pkg_power = self.config["ref_pkg_power_list"][model_index]

            # Log the actual result for debugging
            self.logger.info("=" * 60)
            self.logger.info("=== Test Result Summary (report_csv) ===")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Model: {model_name}")
            self.logger.info(f"Result AI Channels: {result} (type: {type(result).__name__})")
            self.logger.info(f"Reference AI Channels: {ref_value}")
            self.logger.info(f"Test Duration: {duration:.2f}s")
            self.logger.info(f"CSV prefix: {tc_name}, {self.device}, H264 (4Mbps), 1080p@30, {model_name}, {result}")

            # Telemetry collected
            cpu_freq, cpu_util, sys_mem, gpu_freq, eu_usage, vdbox_usage, pkg_power, gpu_power = self.telemetry_list

            prefix = f"{tc_name}, {self.device}, H264 (4Mbps), 1080p@30, {model_name}, {result}"
            prefix_esc = f"{tc_name},{self.device},H264 \\(4Mbps\\), 1080p@30, {model_name}, {result}"
            prefix_esc = prefix_esc.replace("+", "\\+")

            # Use actual telemetry values in CSV output (from telemetry_list)
            # Format: gpu_freq, pkg_power, ref_platform, ref_value, ref_gpu_freq, ref_pkg_power, duration, status
            additional = (
                f"{actual_gpu_freq}, {actual_pkg_power}, {ref_platform}, {ref_value}, {ref_gpu_freq}, {ref_pkg_power}"
            )
            additional += f", {duration:.2f}, No Error"
        else:
            raise RuntimeError("Running with config_file_path {self.config_file_path} not supported!")

        self.update_csv(self.csv_path, prefix_esc, prefix + "," + additional)


OUTPUT_DIR = "/home/dlstreamer/output"
DETAIL_LOG_FILE_PREFIX = f"{OUTPUT_DIR}/ai_vsaas_proxy_pipeline_runner"
RESULT_FILE_PREFIX = f"{OUTPUT_DIR}/ai_vsaas_proxy_pipeline_runner"
TELEMETRY_FILE_PREFIX = f"{OUTPUT_DIR}/ai_vsaas_proxy_pipeline_telemetry"
CSV_FILE_PREFIX = f"{OUTPUT_DIR}/ai_vsaas_proxy_pipeline"

SINK_DIR = "/home/dlstreamer/sink"


def run_ai_vsaas_benchmark(device, monitor_num, is_mtl, has_igpu, config_file):
    if config_file == "none":
        config_file = None

    ai_vsaas_benchmark_runner = AIVSaaSBenchmark(
        name="AI VSaaS Gateway Benchmark",
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

    ai_vsaas_benchmark_runner.prepare()

    if config_file is None:
        ai_vsaas_benchmark_runner.run_benchmark()
    else:
        ai_vsaas_benchmark_runner.run_pipeline_with_config()


if __name__ == "__main__":

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

    args = parser.parse_args()
    run_ai_vsaas_benchmark(args.device, "N/A", args.is_mtl, args.has_igpu, args.config_file)
