# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base Video Analytics Pipeline Benchmark.

This module provides the base class for video analytics benchmarks
including VA multi-stage pipelines (decode/detect/track/classify).
"""

import configparser
import logging
import os
import sys
import time

# Add consolidated utilities to path
sys.path.insert(0, "/home/dlstreamer")

from esq_utils.media.pipeline_utils import BaseDLBenchmark, configure_logging


class BaseVideoAnalyticsBenchmark(BaseDLBenchmark):
    """
    Base class for video analytics pipeline benchmarks.

    Extends BaseDLBenchmark with:
    - Multi-stage pipeline support (decode/detect/track/classify)
    - Compute mode configuration (CPU/iGPU/dGPU/NPU combinations)
    - MQTT metadata publishing support
    - Model path management for detection and classification
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
        """
        Initialize video analytics benchmark.

        Args:
            name: Benchmark name
            device: Primary target device (CPU, iGPU, dGPU.0, dGPU.1)
            monitor_num: Number of monitoring streams
            is_MTL: Whether platform is Meteor Lake
            has_igpu: Whether platform has integrated GPU
            target_fps: Target FPS for scaling test
            telemetry_file_prefix: Prefix for telemetry output files
            log_file: Path to log file (without extension)
            result_file_prefix: Prefix for result files
            csv_path: Path to CSV results file (without extension)
            sink_dir: Directory for sink outputs
            config_file_path: Optional path to custom config file
        """
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

        # Default model paths (can be overridden by subclasses)
        self.model_path = {}

        # MQTT configuration
        self.enable_mqtt = False
        self.mqtt_address = None
        self.mqtt_topic = None

        # Video looping commands
        self.loop_video_commands = []

    def parse_config(self):
        """
        Parse configuration file for custom pipeline settings.

        Returns:
            tuple: (model_name, stream_count)
        """
        config_parser = configparser.ConfigParser()
        config_parser.read(self.config_file_path)

        stream = int(config_parser["AI_Stream"]["ai_stream"])

        det_model_name = config_parser["Detection_Model"]["det_model_name"]
        if not det_model_name:
            raise ValueError("Please provide the name of the detection model")

        # Handle custom detection model
        if det_model_name.lower() not in self.model_path:
            det_model_path = config_parser["Detection_Model"]["det_model_path"]
            det_model_proc_path = config_parser["Detection_Model"].get("det_model_proc_path", "")
            if det_model_path == "" or not os.path.exists(det_model_path):
                raise ValueError("Please provide the correct path of the detection model")
            self.model_path[det_model_name] = {
                "det_model_path": det_model_path,
                "det_proc_json_path": det_model_proc_path,
            }

        cls_model_name = config_parser["Classification_Model"]["cls_model_name"]
        # Handle custom classification model
        if cls_model_name and cls_model_name.lower() not in self.model_path:
            cls_model_path = config_parser["Classification_Model"]["cls_model_path"]
            cls_model_proc_path = config_parser["Classification_Model"].get("cls_model_proc_path", "")
            if cls_model_path == "" or not os.path.exists(cls_model_path):
                raise ValueError("Please provide the correct path of the classification model")
            self.model_path[cls_model_name] = {
                "cls_model_path": cls_model_path,
                "cls_proc_json_path": cls_model_proc_path,
            }

        # MQTT configuration
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

    def run_benchmark_with_modes(self, modes_dict):
        """
        Run benchmark for each compute mode.

        Args:
            modes_dict: Dictionary of mode_name -> (decode_dev, detect_dev, classify_dev)
        """
        self.get_gst_elements("h264")

        for mode_name, devices in modes_dict.items():
            for model in self.config["models"]:
                cur_ref_value = self.config["ref_stream_list"][0]  # Use first ref value
                self.telemetry_file = f"{self.telemetry_file_prefix}_{model}_{self.device}.result"
                self.result_file = f"{self.result_file_prefix}_{model}_{self.device}.result"

                self.logger.info(
                    f"Running {self.benchmark_name} - Mode: {mode_name}, Model: {model}, Device: {self.device}"
                )

                start_time = time.time()
                result = self.run_test_round(
                    resolution=devices,
                    codec="h264",
                    ref_stream=cur_ref_value,
                    model_name=model,
                    max_stream=-1,
                )
                end_time = time.time()
                duration = end_time - start_time

                # Parse result
                if isinstance(result, str) and "@" in result:
                    avg_str, result_str = result.split("@")
                    avg_fps = float(avg_str)
                    result = int(result_str)
                elif isinstance(result, int):
                    avg_fps = 0.0
                else:
                    raise TypeError(f"Unexpected result type: {type(result)}, value: {result}")

                self.report_csv(result, avg_fps, cur_ref_value, duration, model, mode_name, devices)

                self.logger.info(
                    f"{self.benchmark_name} execution [Mode: {mode_name}, Model: {model}] "
                    f"finished in {duration:.2f} seconds"
                )

    def run_pipeline_with_config(self):
        """
        Run pipeline with custom configuration file.
        """
        self.get_gst_elements("h264")

        model_name, stream = self.parse_config()
        self.telemetry_file = f"{self.telemetry_file_prefix}_{model_name}_{self.device}_with_config.result"
        self.result_file = f"{self.result_file_prefix}_{model_name}_{self.device}_with_config.result"

        self.logger.info(
            f"Running {self.benchmark_name} with config using model: {model_name}, "
            f"input resolution: 1080p@{self.target_fps}, device: {self.device}"
        )

        start_time = time.time()
        fps = self.run_test_round_with_config(
            stream=stream,
            resolution="1080p",
            codec="h264",
            model_name=model_name,
        )
        end_time = time.time()
        duration = end_time - start_time

        self.report_csv(stream, fps, duration, model_name)

        self.logger.info(
            f"{self.benchmark_name} execution with config [model: {model_name}] finished in {duration:.2f} seconds"
        )

    def run_test_round_with_config(self, stream, resolution=None, codec=None, bitrate=None, model_name=None):
        """
        Run single test round with configuration.

        Args:
            stream: Number of streams
            resolution: Video resolution or device tuple
            codec: Video codec
            bitrate: Target bitrate
            model_name: Model name(s)

        Returns:
            float: Average FPS
        """
        self.logger.info(f"Start to run the pipeline with {stream} streams")
        gst_cmd = self.gen_gst_command(stream, resolution, codec, bitrate, model_name)
        avg_fps, status = self.run_gst_pipeline(gst_cmd)
        self.logger.info(f"Average fps is {avg_fps}")
        if status != 0:
            self.logger.error(f"Failed to run the pipeline with {stream} streams")
        self.update_telemetry()
        return avg_fps

    def get_sub_elements(self, stage, dev):
        """
        Get GStreamer sub-elements for a specific stage and device.

        Must be implemented by subclasses.

        Args:
            stage: Pipeline stage ("decode", "detect", "classify")
            dev: Device type ("CPU", "iGPU", "dGPU", "NPU")

        Returns:
            str: GStreamer element string for the stage
        """
        raise NotImplementedError("Subclass must implement get_sub_elements")

    def gen_gst_command(self, stream, resolution=None, codec=None, bitrate=None, model_name=None):
        """
        Generate GStreamer pipeline command.

        Must be implemented by subclasses.

        Args:
            stream: Number of streams
            resolution: Video resolution or device tuple
            codec: Video codec
            bitrate: Target bitrate
            model_name: Model name(s)

        Returns:
            str: GStreamer command string
        """
        raise NotImplementedError("Subclass must implement gen_gst_command")

    def report_csv(self, *args):
        """
        Report results to CSV file.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement report_csv")
