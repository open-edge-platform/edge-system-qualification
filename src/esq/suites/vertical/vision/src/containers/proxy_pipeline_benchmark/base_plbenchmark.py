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
        # Track best result across all modes to preserve correct telemetry
        best_result = -1
        best_telemetry_list = None

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

                # Track best result and save its telemetry
                if result > best_result:
                    best_result = result
                    # Deep copy telemetry list to preserve it
                    best_telemetry_list = list(self.telemetry_list) if self.telemetry_list else None
                    self.logger.info(
                        f"New best result: {best_result} streams (mode: {mode}, model: {mod}). "
                        f"Telemetry saved."
                    )

                self.report_csv(result, avg_fps, cur_ref_value, duration, mod, mode, devices)
                self.logger.info(
                    f"{self.benchmark_name} execution [model: {mod}, input resolution: 1080p@{self.target_fps}, device: {self.device}] finished in {duration:.2f} seconds"
                )

        # Restore best telemetry after all modes complete
        if best_telemetry_list is not None:
            self.telemetry_list = best_telemetry_list
            self.logger.info(
                f"Restored telemetry from best result ({best_result} streams). "
                f"Final telemetry will reflect the highest performing mode."
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

            self.logger.info("="*80)
            self.logger.info(f"[MODEL {i+1}/{len(self.config['models'])}] STARTING TEST")
            self.logger.info(f"Model: {mod}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Resolution: 1080p@{self.target_fps}")
            self.logger.info(f"Reference streams: {cur_ref_value}")
            self.logger.info("="*80)

            start_time = time.time()
            result = self.run_test_round(
                resolution="1080p", codec="h264", ref_stream=cur_ref_value, model_name=mod, max_stream=max_stream
            )

            self.logger.info("="*80)
            self.logger.info(f"[MODEL {i+1}/{len(self.config['models'])}] TEST ROUND COMPLETED")
            self.logger.info(f"Result: {result} streams achieved")
            self.logger.info("[PROGRESS] Processing results and updating telemetry...")
            self.logger.info("="*80)

            end_time = time.time()
            duration = end_time - start_time

            # Log the average FPS for the best result
            if hasattr(self, 'best_avg_fps'):
                self.logger.info(f"  Best Average FPS: {self.best_avg_fps:.2f} (for {result} streams)")
            else:
                self.logger.info("  Average FPS: Not available")

            # Ensure telemetry is read from the current model's telemetry file
            # run_test_round should have updated telemetry, but verify the file exists
            if not os.path.isfile(self.telemetry_file):
                self.logger.warning(f"Telemetry file not found: {self.telemetry_file}")
                self.telemetry_list = [-1] * 8
            elif result == 0:
                self.logger.warning(f"No successful streams for model {mod}, setting telemetry to -1")
                self.telemetry_list = [-1] * 8
            else:
                # Re-read telemetry to ensure we have the latest data for this model
                try:
                    self.logger.info(f"[PROGRESS] Calling update_telemetry() for {mod}...")
                    self.update_telemetry()
                    self.logger.info(f"[PROGRESS] Telemetry updated successfully for {mod}: {self.telemetry_list}")
                except Exception as e:
                    self.logger.error(f"Failed to update telemetry for {mod}: {e}")
                    self.telemetry_list = [-1] * 8

            # Extract actual GPU Freq and Pkg Power from telemetry data collected during test
            # Telemetry list order: [0]=CPU Freq, [1]=CPU Usage, [2]=Mem Usage, 
            #                        [3]=GPU Freq, [4]=EU Usage, [5]=VDBox Usage, 
            #                        [6]=Pkg Power, [7]=GPU Power
            # Initialize with reference config values (will be overwritten if telemetry is valid)
            actual_gpu_freq = cur_ref_gpu_freq
            actual_pkg_power = cur_ref_pkg_power
            try:
                if len(self.telemetry_list) >= 8:
                    telemetry_gpu_freq = float(self.telemetry_list[3])
                    telemetry_pkg_power = float(self.telemetry_list[6])
                    if telemetry_gpu_freq > 0:
                        actual_gpu_freq = telemetry_gpu_freq
                        actual_pkg_power = telemetry_pkg_power
                        self.logger.info(f"Using actual telemetry data - GPU Freq: {actual_gpu_freq:.2f} MHz, Pkg Power: {actual_pkg_power:.2f} W")
                    else:
                        self.logger.warning("GPU Freq is 0, telemetry may not be available. Using config values.")
                else:
                    self.logger.warning(f"Insufficient telemetry data ({len(self.telemetry_list)} entries), using config values")
            except (ValueError, IndexError, TypeError) as e:
                self.logger.warning(f"Failed to extract telemetry data: {e}, using config values")

            self.logger.info(f"[PROGRESS] Writing results to CSV for {mod}...")
            self.report_csv(result, cur_ref_value, cur_ref_gpu_freq, cur_ref_pkg_power, duration, mod)
            self.logger.info("[PROGRESS] CSV report written successfully")

            self.logger.info("="*80)
            self.logger.info(f"[MODEL {i+1}/{len(self.config['models'])}] COMPLETED")
            self.logger.info(f"{self.benchmark_name} execution finished in {duration:.2f} seconds")
            self.logger.info(f"Model: {mod}, Result: {result} streams")
            self.logger.info("="*80)

            # Add small delay to ensure logs are flushed
            import time as time_module
            time_module.sleep(0.5)

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
