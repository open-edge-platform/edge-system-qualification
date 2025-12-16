import os
import time
import logging
import configparser
import re
import sys

# Add consolidated utilities to path
sys.path.insert(0, "/home/dlstreamer")

from esq_utils.media.pipeline_utils import BaseDLBenchmark, configure_logging


class BaseProxyPipelineBenchmark(BaseDLBenchmark):
    def __init__(self, name, device, monitor_num, is_MTL, has_igpu, target_fps, telemetry_file_prefix, log_file, result_file_prefix, csv_path, sink_dir, config_file_path=None):
        super().__init__(name, device, monitor_num, is_MTL, has_igpu, target_fps, telemetry_file_prefix, log_file, result_file_prefix, csv_path)
        self.sink_dir = sink_dir
        self.config_file_path = config_file_path

        if self.config_file_path:
            self.log_file = log_file + "_with_config.log"
            self.csv_path = csv_path + "_with_config.csv"
        else:
            self.log_file = log_file + ".log"
            self.csv_path = csv_path + ".csv"

        configure_logging(self.log_file)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_path = {
            "yolov5s-416": {
                "det_model_path": "/home/dlstreamer/share/models/yolov5s-416_INT8/FP16-INT8/yolov5s.xml",
                "det_proc_json_path": "/home/dlstreamer/share/models/yolov5s-416_INT8/yolo-v5.json",
            },
            "yolov5m-416": {
                "det_model_path": "/home/dlstreamer/share/models/yolov5m-416_INT8/FP16-INT8/yolov5m-416_INT8.xml",
                "det_proc_json_path": "/home/dlstreamer/share/models/yolov5m-416_INT8/yolo-v5.json",
            },
            "efficientnet-b0": {
                "cls_model_path": "/home/dlstreamer/share/models/efficientnet-b0_INT8/FP16-INT8/efficientnet-b0.xml",
                "cls_proc_json_path": "/home/dlstreamer/share/models/efficientnet-b0_INT8/efficientnet-b0.json",
            },
            "yolov8_license_plate_detector": {
                "det_model_path": "/home/dlstreamer/share/models/lpr/models/yolov8n/yolov8n_retrained.xml"
            },
            "ch_PP-OCRv4_rec_infer": {
                "cls_model_path": "/home/dlstreamer/share/models/lpr/models/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.xml"
            },
        }

        self.enable_mqtt = False
        self.mqtt_address = None
        self.mqtt_topic = None

        self.loop_video_commands = ""

    def parse_config(self):
        config_parser = configparser.ConfigParser()
        config_parser.read(self.config_file_path)

        stream = int(config_parser["AI_Stream"]["ai_stream"])

        det_model_name = config_parser["Detection_Model"]["det_model_name"]
        if not det_model_name:
            raise ValueError("Please provide the name of the detection model")
        if det_model_name.lower() not in ("yolov5s-416", "yolov5m-416", "yolov8_license_plate_detector"):
            det_model_path = config_parser["Detection_Model"]["det_model_path"]
            det_model_proc_path = config_parser["Detection_Model"]["det_model_proc_path"]
            if det_model_path == "" or not os.path.exists(det_model_path):
                raise ValueError("Please provide the correct path of the detection model")
            if (
                det_model_name.lower() != "yolov8_license_plate_detector"
                and det_model_proc_path == ""
                or not os.path.exists(det_model_proc_path)
            ):
                raise ValueError("Please provide the correct path of the proc json file of the detection model")
            self.model_path[det_model_name] = {
                "det_model_path": det_model_path,
                "det_proc_json_path": det_model_proc_path,
            }

        cls_model_name = config_parser["Classification_Model"]["cls_model_name"]
        if cls_model_name.lower() not in ("efficientnet-b0", "ch_PP-OCRv4_rec_infer"):
            cls_model_path = config_parser["Classification_Model"]["cls_model_path"]
            cls_model_proc_path = config_parser["Classification_Model"]["cls_model_proc_path"]
            if cls_model_path == "" or not os.path.exists(cls_model_path):
                raise ValueError("Please provide the correct path of the classification model")
            if (
                cls_model_name.lower() != "ch_PP-OCRv4_rec_infer"
                and cls_model_proc_path == ""
                or not os.path.exists(cls_model_proc_path)
            ):
                raise ValueError("Please provide the correct path of the proc json file of the classification model")
            self.model_path[cls_model_name] = {
                "cls_model_path": cls_model_path,
                "cls_proc_json_path": cls_model_proc_path,
            }

        if config_parser["MQTT"]["enable_mqtt"] == "true":
            self.enable_mqtt = True
            self.mqtt_address = config_parser["MQTT"]["mqtt_address"]
            self.mqtt_topic = config_parser["MQTT"]["mqtt_topic"]
            if self.mqtt_address == "" or self.mqtt_topic == "":
                raise ValueError("Please provide the correct mqtt address and topic")

        if not cls_model_name:
            return det_model_name, stream
        else:
            return det_model_name + "+" + cls_model_name, stream

    def bpl_run_lpr_benchmark(self):
        self.get_gst_elements("h264")
        max_stream = -1

        for j, (mode, devices) in enumerate(self.config["modes"].items()):
            for i, mod in enumerate(self.config["models"]):
                cur_ref_value = self.config["ref_stream_list"][i]
                self.telemetry_file = f"{self.telemetry_file_prefix}_{mod}_{self.device}.result"
                self.result_file = f"{self.result_file_prefix}_{mod}_{self.device}.result"
                self.logger.info(
                    f"Running {self.benchmark_name} using model: {mod}, input resolution: 1080p@{self.target_fps}, device: {self.device}"
                )
                start_time = time.time()
                result = self.run_test_round(
                    resolution=devices, codec="h264", ref_stream=cur_ref_value, model_name=mod, max_stream=max_stream
                )
                end_time = time.time()
                duration = end_time - start_time

                # Handle both old format (string with "@") and new format (int)
                if isinstance(result, str) and "@" in result:
                    avg_str, result_str = result.split("@")
                    avg_fps = float(avg_str)
                    result = int(result_str)
                elif isinstance(result, int):
                    # New format: run_test_round returns just the result code
                    # avg_fps needs to be extracted from telemetry or logs
                    result = result
                    avg_fps = 0.0  # Placeholder - will be populated from logs/telemetry
                else:
                    raise TypeError(f"Unexpected result type from run_test_round: {type(result)}, value: {result}")
                self.report_csv(result, avg_fps, cur_ref_value, duration, mod, mode, devices)
                self.logger.info(
                    f"{self.benchmark_name} execution [model: {mod}, input resolution: 1080p@{self.target_fps}, device: {self.device}] finished in {duration:.2f} seconds"
                )

    def run_benchmark(self):
        self.get_gst_elements("h264")
        if self.benchmark_name == "AI VSaaS Gateway Benchmark":
            # replace h264 encoding with h265 encoding
            self.enc_ele = self.enc_ele.replace("264", "265")
            max_stream = -1
        else:
            max_stream = self.config["compose_size"] ** 2

        for i, mod in enumerate(self.config["models"]):
            cur_ref_value = self.config["ref_stream_list"][i]
            # Get reference values from config (will be overwritten with actual telemetry data)
            cur_ref_gpu_freq = self.config["ref_gpu_freq_list"][i]
            cur_ref_pkg_power = self.config["ref_pkg_power_list"][i]

            self.telemetry_file = f"{self.telemetry_file_prefix}_{mod}_{self.device}.result"
            self.result_file = f"{self.result_file_prefix}_{mod}_{self.device}.result"
            self.logger.info(
                f"Running {self.benchmark_name} using model: {mod}, input resolution: 1080p@{self.target_fps}, device: {self.device}"
            )
            start_time = time.time()
            result = self.run_test_round(
                resolution="1080p", codec="h264", ref_stream=cur_ref_value, model_name=mod, max_stream=max_stream
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
                    self.logger.info(f"Using actual telemetry data - GPU Freq: {actual_gpu_freq:.2f} MHz, Pkg Power: {actual_pkg_power:.2f} W")
                    # Use actual values instead of config defaults
                    cur_ref_gpu_freq = actual_gpu_freq
                    cur_ref_pkg_power = actual_pkg_power
                else:
                    self.logger.warning(f"Insufficient telemetry data ({len(self.telemetry_list)} entries), using config values")
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Failed to extract telemetry data: {e}, using config values")

            self.report_csv(result, cur_ref_value, cur_ref_gpu_freq, cur_ref_pkg_power, duration, mod)
            self.logger.info(
                f"{self.benchmark_name} execution [model: {mod}, input resolution: 1080p@{self.target_fps}, device: {self.device}] finished in {duration:.2f} seconds"
            )

    def run_test_round_with_config(self, stream, resolution=None, codec=None, bitrate=None, model_name=None):
        self.logger.info(f"Start to run the pipeline with {stream} streams")
        gst_cmd = self.gen_gst_command(stream, resolution, codec, bitrate, model_name)
        avg_fps, status = self.run_gst_pipeline(gst_cmd)
        self.logger.info(f"Average fps is {avg_fps}")
        if status != 0:
            self.logger.error(f"Failed to run the pipeline with {stream} streams")
        self.update_telemetry()
        return avg_fps

    def run_pipeline_with_config(self):
        self.get_gst_elements("h264")
        if self.benchmark_name == "AI VSaaS Gateway Benchmark":
            # replace h264 encoding with h265 encoding
            self.enc_ele = self.enc_ele.replace("264", "265")

        model_name, stream = self.parse_config()
        self.telemetry_file = f"{self.telemetry_file_prefix}_{model_name}_{self.device}_with_config.result"
        self.result_file = f"{self.result_file_prefix}_{model_name}_{self.device}_with_config.result"
        self.logger.info(
            f"Running {self.benchmark_name} with config using model: {model_name}, input resolution: 1080p@{self.target_fps}, device: {self.device}"
        )
        start_time = time.time()
        fps = self.run_test_round_with_config(stream=stream, resolution="1080p", codec="h264", model_name=model_name)
        end_time = time.time()
        duration = end_time - start_time
        # TODO: Add logic for LPR Pipeline
        self.report_csv(stream, fps, duration, model_name)
        self.logger.info(
            f"{self.benchmark_name} execution with config [model: {model_name}, input resolution: 1080p@{self.target_fps}, device: {self.device}] finished in {duration:.2f} seconds"
        )
