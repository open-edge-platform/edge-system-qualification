import argparse
import os
import sys

# Add consolidated utilities to path
sys.path.insert(0, "/home/dlstreamer")

from base_plbenchmark import BaseProxyPipelineBenchmark


class SmartNVRBenchmark(BaseProxyPipelineBenchmark):
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

        self.loop_video_commands = [["bash", "gst_loop_mp4.sh", "18", "1080p", "h264", "20"]]

    def get_config_of_platform(self):
        # Device name is already normalized by test runner (iGPU, dGPU, CPU, NPU)
        # Get device type for config selection (strip index for dGPU.0, dGPU.1)
        device_type = self.device.split(".")[0] if "." in self.device else self.device

        if device_type == "iGPU":
            if self.VDBOX == 1:
                self.config = {
                    "compose_size": 4,
                    "ref_stream_list": [10],
                    "ref_gpu_freq_list": [-1],
                    "ref_pkg_power_list": [-1],
                    "ref_platform": "i5-12400 (16G Mem)",
                    "output_width": 1920,
                    "output_height": 1080,
                    "models": ["yolov5s-416"],
                    "enc_flag": "rate-control=cbr bitrate=4000 target-usage=7",
                    "preproc_backend": "pre-process-backend=vaapi-surface-sharing scale-method=fast",
                }
            else:
                if self.is_MTL:
                    self.config = {
                        "compose_size": 5,
                        "ref_stream_list": [23, 12, 3],
                        "ref_gpu_freq_list": [-1, -1, -1],
                        "ref_pkg_power_list": [-1, -1, -1],
                        "ref_platform": "MTL 165H (32G Mem)",
                        "output_width": 3840,
                        "output_height": 2160,
                        "models": ["yolov5s-416", "yolov5m-416", "yolov5m-416+efficientnet-b0"],
                        "enc_flag": "rate-control=cbr bitrate=16000 target-usage=7",
                        "preproc_backend": "pre-process-backend=vaapi-surface-sharing scale-method=fast",
                    }
                else:
                    self.config = {
                        "compose_size": 5,
                        "ref_stream_list": [15, 7, 2],
                        "ref_gpu_freq_list": [-1, -1, -1],
                        "ref_pkg_power_list": [-1, -1, -1],
                        "ref_platform": "i7-1360p (16G Mem)",
                        "output_width": 3840,
                        "output_height": 2160,
                        "models": ["yolov5s-416", "yolov5m-416", "yolov5m-416+efficientnet-b0"],
                        "enc_flag": "rate-control=cbr bitrate=16000 target-usage=7",
                        "preproc_backend": "pre-process-backend=vaapi-surface-sharing scale-method=fast",
                    }
        elif device_type == "dGPU":
            self.config = {
                "compose_size": 6,
                "ref_stream_list": [23, 6],
                "ref_gpu_freq_list": [-1, -1],
                "ref_pkg_power_list": [-1, -1],
                "ref_platform": "Arc A380",
                "output_width": 3840,
                "output_height": 2160,
                "models": ["yolov5m-416", "yolov5m-416+efficientnet-b0"],
                "enc_flag": "rate-control=cbr bitrate=16000 target-usage=7",
                "preproc_backend": "pre-process-backend=vaapi-surface-sharing scale-method=fast",
            }
        elif device_type == "CPU":
            self.config = {
                "compose_size": 6,
                "ref_stream_list": [26, 5],
                "ref_gpu_freq_list": [-1, -1],
                "ref_pkg_power_list": [-1, -1],
                "ref_platform": "Xeon(R) Gold 6430 (512G Mem)",
                "output_width": 3840,
                "output_height": 2160,
                "models": ["yolov5m-416", "yolov5m-416+efficientnet-b0"],
                "enc_flag": "bitrate=16000 speed-preset=superfast",
                "preproc_backend": "",
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
        video_src = f"/home/dlstreamer/sample_video/car_{resolution}20_180s_{codec}.mp4"
        gst_cmd = "compositor name=comp "

        total_stream = self.config["compose_size"] ** 2
        compose_size = self.config["compose_size"]
        cell_width = self.config["output_width"] // compose_size
        cell_height = self.config["output_height"] // compose_size
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
            inf_device = "GPU.0"
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

        if self.monitor_num == 0:
            display_sink = "fakesink async=false sync=false"
        else:
            # Use DISPLAY environment variable (e.g., :1 for NoMachine, :0 for native X)
            display = os.getenv("DISPLAY", ":0")
            display_sink = f"xvimagesink display={display} async=false sync=false"

        for i in range(total_stream):
            gst_cmd += f"sink_{i}::xpos={i % compose_size * cell_width} sink_{i}::ypos={i // compose_size * cell_height} sink_{i}::alpha=1 "

        compose_file_sink_cmd = f"h264parse ! mp4mux ! filesink location={self.sink_dir}/smart_nvr_composed_video.mp4 sync=false async=false "
        if device_type == "CPU":
            gst_cmd += f"! {self.enc_ele} {enc_flag} ! {compose_file_sink_cmd}"
        else:
            gst_cmd += f"! tee name=t ! queue ! {self.enc_ele} {enc_flag} ! {compose_file_sink_cmd}"
            gst_cmd += f"t. ! queue ! {display_sink} "

        for i in range(stream):
            gst_cmd += f"filesrc location={video_src} ! qtdemux ! h264parse ! tee name=t{i} ! queue ! mp4mux ! filesink async=false sync=false location={self.sink_dir}/smart_nvr_proxy_pipeline_filesink_stream{i}.mp4 "
            if device_type == "CPU":
                gst_cmd += f"t{i}. ! queue ! decodebin "
            else:
                gst_cmd += f"t{i}. ! queue ! {self.dec_ele} ! video/x-raw(memory:VAMemory) "
            gst_cmd += f"! gvadetect model={det_model_path} {preproc_backend} model-proc={det_proc_json_path} batch-size=1 nireq=2 ie-config=NUM_STREAMS=2 inference-interval=2 model-instance-id=detect{i} device={inf_device} threshold=0.5 "
            gst_cmd += "! gvatrack tracking-type=short-term-imageless "
            if cls_model_name:
                if self.is_MTL and device_type == "iGPU":
                    gst_cmd += f"! gvaclassify model={cls_model_path} {preproc_backend} model-proc={cls_proc_json_path} batch-size=1 nireq=2 inference-interval=2 inference-region=roi-list model-instance-id=classify{i} device=NPU "
                else:
                    gst_cmd += f"! gvaclassify model={cls_model_path} {preproc_backend} model-proc={cls_proc_json_path} batch-size=1 nireq=2 ie-config=NUM_STREAMS=2 inference-interval=2 inference-region=roi-list model-instance-id=classify{i} device={inf_device} "

            if self.enable_mqtt:
                gst_cmd += f"! gvametaconvert format=json json-indent=4 source={video_src} add-empty-results=true ! gvametapublish method=mqtt address={self.mqtt_address} mqtt-client-id=client{i} topic={self.mqtt_topic} "
            else:
                gst_cmd += f"! gvametaconvert format=json json-indent=4 source={video_src} add-empty-results=true ! gvametapublish method=file file-path=/dev/null "

            if device_type == "CPU":
                gst_cmd += f"! videoscale ! video/x-raw,width={cell_width},height={cell_height} ! gvafpscounter starting-frame=1000 ! comp.sink_{i} "
            else:
                gst_cmd += f"! {self.post_proc_ele} scale-method=fast ! video/x-raw,width={cell_width},height={cell_height} ! gvafpscounter starting-frame=1000 ! comp.sink_{i} "

        for i in range(total_stream - stream):
            idx = i + stream
            gst_cmd += f"filesrc location={video_src} ! qtdemux ! h264parse ! tee name=t{idx} ! queue ! mp4mux ! filesink async=false sync=false location={self.sink_dir}/smart_nvr_proxy_pipeline_filesink_stream{idx}.mp4 "
            if device_type == "CPU":
                gst_cmd += f"t{idx}. ! queue ! decodebin ! videoscale ! video/x-raw,width={cell_width},height={cell_height} ! gvafpscounter starting-frame=1500 ! queue ! comp.sink_{idx} "
            else:
                gst_cmd += f"t{idx}. ! queue ! {self.dec_ele} ! {self.post_proc_ele} scale-method=fast ! video/x-raw,width={cell_width},height={cell_height} ! gvafpscounter starting-frame=1000 ! comp.sink_{idx} "

        self.logger.debug(f"Genreated gst command for {stream} streams: ")
        self.logger.debug(gst_cmd)
        return gst_cmd

    def report_csv(self, *arg):
        tc_name = "Smart NVR Pipeline"
        compose_size = self.config["compose_size"]
        total_stream = compose_size**2

        if not self.config_file_path:
            result, ref_value, ref_gpu_freq, ref_pkg_power, duration, model_name = arg
            ref_platform = self.config.get("ref_platform", "Unknown")

            # Telmetry collected
            cpu_freq, cpu_util, sys_mem, gpu_freq, eu_usage, vdbox_usage, pkg_power, gpu_power = self.telemetry_list

            prefix = (
                f"{tc_name}, {self.device}, H264 (4Mbps), 1080p@20, {model_name}, "
                f"{compose_size}x{compose_size}, {self.monitor_num}, {total_stream - result}, {result}"
            )
            prefix_esc = (
                f"{tc_name}, {self.device}, H264 \\(4Mbps\\), 1080p@20, {model_name}, "
                f"{compose_size}x{compose_size}, {self.monitor_num}, {total_stream - result}, {result}"
            )
            prefix_esc = prefix_esc.replace("+", "\\+")

            additional = f"{gpu_freq}, {pkg_power}, {ref_platform}, {ref_value}, {ref_gpu_freq}, {ref_pkg_power}"
            additional += f", {duration:.2f}, No Error"

        else:
            raise RuntimeError("Running with config_file_path {self.config_file_path} not supported!")

        self.update_csv(self.csv_path, prefix_esc, prefix + "," + additional)


OUTPUT_DIR = "/home/dlstreamer/output"
DETAIL_LOG_FILE_PREFIX = f"{OUTPUT_DIR}/smart_nvr_proxy_pipeline_runner"
RESULT_FILE_PREFIX = f"{OUTPUT_DIR}/smart_nvr_proxy_pipeline_runner"
TELEMETRY_FILE_PREFIX = f"{OUTPUT_DIR}/smart_nvr_proxy_pipeline_telemetry"
CSV_FILE_PREFIX = f"{OUTPUT_DIR}/smart_nvr_proxy_pipeline"

SINK_DIR = "/home/dlstreamer/sink"


def run_smart_nvr_benchmark(device, monitor_num, is_mtl, has_igpu, config_file):
    if config_file == "none":
        config_file = None

    smart_nvr_runner = SmartNVRBenchmark(
        name="Smart NVR Benchmark",
        device=device,
        monitor_num=monitor_num,
        is_MTL=is_mtl,
        has_igpu=has_igpu,
        target_fps=20,
        telemetry_file_prefix=TELEMETRY_FILE_PREFIX,
        log_file=DETAIL_LOG_FILE_PREFIX,
        result_file_prefix=RESULT_FILE_PREFIX,
        csv_path=CSV_FILE_PREFIX,
        sink_dir=SINK_DIR,
        config_file_path=config_file,
    )
    smart_nvr_runner.prepare()

    if config_file is None:
        smart_nvr_runner.run_benchmark()
    else:
        smart_nvr_runner.run_pipeline_with_config()


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
    run_smart_nvr_benchmark(args.device, args.monitor_num, args.is_mtl, args.has_igpu, args.config_file)
