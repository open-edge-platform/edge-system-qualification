# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline utilities for media benchmarks.

Consolidates base_dlbenchmark.py and benchmark_log.py from media/proxy containers.
"""

import fcntl
import logging
import multiprocessing
import os
import re
import signal
import subprocess  # nosec B404 - Popen needed for background pipeline execution
import time

# Support both installed package and Docker container usage
try:
    from sysagent.utils.core.process import run_command
    # For Popen cases, we need to use subprocess with proper validation
    # FW API doesn't expose Popen directly, so use container_utils wrapper
    from esq.utils.media.container_utils import secure_popen
except ModuleNotFoundError:
    # Inside Docker container, use the lightweight container utilities
    from .container_utils import run_command, secure_popen


def configure_logging(log_file_name):
    """
    Configure logging for benchmark execution.

    Args:
        log_file_name: Path to log file
    """
    # Use INFO level by default for production
    # Can be overridden via LOG_LEVEL environment variable if needed
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_name),
        ],
    )


class BenchmarkLogger:
    """Wrapper for benchmark logging configuration."""

    @staticmethod
    def configure(log_file_name):
        """Configure logging to file with INFO level."""
        configure_logging(log_file_name)


class BaseDLBenchmark:
    """
    Base class for DLStreamer benchmark execution.

    Provides common functionality for media decode/encode/compose benchmarks:
    - Pipeline execution and monitoring
    - Telemetry collection
    - Result validation and CSV reporting
    - Multi-stream scaling tests
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
    ):
        """
        Initialize benchmark.

        Args:
            name: Benchmark name (e.g., "Decode Benchmark")
            device: Target device (CPU, iGPU, dGPU.0, dGPU.1)
            monitor_num: Number of monitoring streams
            is_MTL: Whether platform is Meteor Lake
            has_igpu: Whether platform has integrated GPU
            target_fps: Target FPS for scaling test
            telemetry_file_prefix: Prefix for telemetry output files
            log_file: Path to log file
            result_file_prefix: Prefix for result files
            csv_path: Path to CSV results file
        """
        self.benchmark_name = name
        self.device = device
        self.monitor_num = monitor_num
        self.is_MTL = is_MTL
        self.has_igpu = has_igpu
        self.target_fps = target_fps
        self.telemetry_file_prefix = telemetry_file_prefix
        self.result_file_prefix = result_file_prefix

        self.log_file = log_file
        self.csv_path = csv_path

        self.gpu_render = 0
        self.telemetry_list = []
        self.loop_video_commands = []
        self.available_va_plugins = ""

        self.logger = logging.getLogger(self.__class__.__name__)

    def run_command(self, command):
        """
        Run shell command and return stdout.

        Args:
            command: Command as list of strings

        Returns:
            Command stdout as string
        """
        result = run_command(command, capture_output=True)
        return result.stdout

    def get_dGPU_Info(self):
        """Extract dGPU index and calculate render device number."""
        self.dgpu_idx = int(self.device.split(".")[1])
        if self.has_igpu:
            self.gpu_render = 129 + self.dgpu_idx
        else:
            self.gpu_render = 128 + self.dgpu_idx

    def get_gst_elements(self, codec):
        """
        Get GStreamer element names based on device and codec.

        Args:
            codec: Codec name (h264, h265, av1, vp9)

        Sets self.enc_ele, self.dec_ele, self.post_proc_ele
        """
        self.logger.debug(self.available_va_plugins)
        if self.device == "CPU":
            self.enc_ele = f"x{codec[1:]}enc"
            self.dec_ele = "decodebin"
        else:
            if self.gpu_render == 128:
                if self.device == "iGPU":
                    if self.is_MTL:
                        self.enc_ele = f"va{codec}enc"
                    else:
                        self.enc_ele = f"va{codec}lpenc"
                else:
                    self.enc_ele = f"qsv{codec}enc"
                self.dec_ele = f"va{codec}dec"
                self.post_proc_ele = "vapostproc"
            else:
                self.dec_ele = f"varenderD{self.gpu_render}{codec}dec"
                if self.is_MTL:
                    self.enc_ele = f"va{codec}enc"
                else:
                    self.enc_ele = f"va{codec}lpenc"
                self.post_proc_ele = f"varenderD{self.gpu_render}postproc"

    def check_VDBox(self):
        """Check number of video decode boxes (VDBox) available on GPU."""
        if self.device == "CPU":
            self.VDBOX = 0
        else:
            if self.device == "iGPU":
                self.gpu_render = 128

            self.logger.info(f"Start to check number of VDBox in {self.device}")
            vdbox_output = self.run_command(["lsgpu", "-p"])
            if "vcs0" in vdbox_output and "vcs1" in vdbox_output:
                self.VDBOX = 2
            else:
                self.VDBOX = 1
            self.logger.info(f"This {self.device} has {self.VDBOX} VDBox")

    def get_config_of_platform(self):
        """Get platform configuration. Must be implemented by subclass."""
        raise NotImplementedError

    def report_csv(self, *arg):
        """Report results to CSV. Must be implemented by subclass."""
        raise NotImplementedError

    def update_csv(self, csv_path, prefix, cur_record):
        """
        Update CSV file with new result record.

        Uses file locking to ensure thread-safe updates.

        Args:
            csv_path: Path to CSV file
            prefix: Record identifier prefix
            cur_record: Full CSV record line to write
        """
        with open(csv_path, "r+") as f:
            # Acquire lock to ensure only 1 process can modify file
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                pattern = re.compile(f"{prefix},.*")
                content = f.read()
                if pattern.search(content):
                    self.logger.info(f"Existed result for {prefix}, update")
                    updated_content = pattern.sub(f"{cur_record}", content)

                    f.seek(0)
                    f.write(updated_content)
                    f.truncate()
                else:
                    self.logger.info(f"No record for {prefix}, add one")
                    # Move to end of file
                    f.seek(0, os.SEEK_END)
                    f.write(f"{cur_record}\n")
            finally:
                # Release the lock
                fcntl.flock(f, fcntl.LOCK_UN)

    def collect_telemetry(self):
        """Start telemetry collection in separate process."""
        # Import here to avoid circular dependency
        try:
            from esq.utils.media.telemetry import TelemetryCollector
        except ModuleNotFoundError:
            # Inside Docker container, use relative import
            from .telemetry import TelemetryCollector

        self.telemetry_collector = TelemetryCollector(self.device, self.gpu_render, self.telemetry_file)
        self.tele_process = multiprocessing.Process(
            target=self.telemetry_collector.collect_telemetries,
        )
        self.tele_process.start()

    def stop_telemetry(self):
        """Stop telemetry collection process."""
        if self.tele_process.is_alive():
            try:
                os.kill(self.tele_process.pid, signal.SIGUSR1)
                self.tele_process.join(timeout=5)
            except Exception as e:
                self.logger.warning(f"Failed to stop telemetry gracefully: {e}")

            # Force terminate if still alive after timeout
            if self.tele_process.is_alive():
                self.logger.warning("Telemetry process didn't stop, forcing termination")
                self.tele_process.terminate()
                self.tele_process.join(timeout=2)

            # Force kill if still alive
            if self.tele_process.is_alive():
                self.logger.error("Telemetry process didn't terminate, forcing kill")
                self.tele_process.kill()
                self.tele_process.join(timeout=1)

    def update_telemetry(self):
        """Read telemetry results from file."""
        self.telemetry_list = []
        with open(self.telemetry_file, "r") as f:
            for line in f:
                self.logger.debug(f"Telemetry line: {line.strip()} from {self.telemetry_file}")
                self.telemetry_list.append(line.split(":")[1].strip())

    def gen_gst_command(self, stream, resolution=None, codec=None, bitrate=None, model_name=None):
        """
        Generate GStreamer pipeline command. Must be implemented by subclass.

        Args:
            stream: Number of streams
            resolution: Video resolution
            codec: Codec name
            bitrate: Target bitrate
            model_name: Model name for inference

        Returns:
            GStreamer command string
        """
        raise NotImplementedError

    def run_gst_pipeline(self, gst_cmd):
        """
        Run GStreamer pipeline and collect metrics.

        Args:
            gst_cmd: GStreamer command string

        Returns:
            tuple: (avg_fps, status)
                avg_fps: Average FPS achieved
                status: 0 on success, 1 on failure
        """
        gst_process = secure_popen(
            ["bash", "./run_pipeline.sh", self.device, gst_cmd],
            text=True,
            stdout=open(self.result_file, "w"),
            stderr=subprocess.STDOUT,
        )

        time.sleep(1)

        self.collect_telemetry()

        status = 0
        avg_fps = 0.0

        try:
            while gst_process.poll() is None:
                # Kill pipeline if no output for 60 seconds
                if os.path.isfile(self.result_file) and (time.time() - os.path.getmtime(self.result_file)) > 60:
                    with open(self.result_file, "r") as f:
                        if "overall" not in f.read():
                            status = 1
                    gst_process.terminate()
                    gst_process.wait()
                    self.logger.info("Pipeline killed after no output for 60s")
                    break
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        finally:
            self.logger.debug("pipeline finish, kill process and stop telemetry")
            # Only wait if process is still running (poll() returns None)
            # Avoid double-wait which causes hang when process already exited
            if gst_process.poll() is None:
                try:
                    gst_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Process didn't terminate gracefully, forcing kill")
                    gst_process.kill()
                    gst_process.wait()
            self.stop_telemetry()

        # Parse results from output file
        with open(self.result_file, "r") as f:
            for line in f:
                if "ERROR: from element" in line:
                    status = 1
                    break
                if "overall" in line:
                    match = re.search(r"per-stream=\s*([\d]+\.\d+)\s+fps", line)
                    if match:
                        avg_fps = float(match.group(1))
                    break
        return avg_fps, status

    def _meet_target_fps(self, fps):
        """
        Check if FPS meets target threshold.

        Args:
            fps: Achieved FPS

        Returns:
            bool: True if target met
        """
        return fps >= self.target_fps

    def run_test_round(
        self, resolution=None, codec=None, bitrate=None, ref_stream=None, model_name=None, max_stream=-1
    ):
        """
        Run binary search to find maximum concurrent streams meeting target FPS.

        Args:
            resolution: Video resolution
            codec: Codec name
            bitrate: Target bitrate
            ref_stream: Reference stream count for binary search
            model_name: Model name for inference
            max_stream: Maximum streams to test (-1 for unlimited)

        Returns:
            int or str: Maximum concurrent streams, or formatted result string
        """
        quick_search = True
        low = 1
        high = ref_stream
        current_stream = 0
        result = 0

        while True:
            if quick_search:
                current_stream = (low + high) // 2
            else:
                current_stream += 1
                if max_stream > 0 and current_stream > max_stream:
                    break

            self.logger.info(f"Start to run the pipeline with {current_stream} streams")

            if self.benchmark_name == "LPR Benchmark":
                gst_cmd = self.gen_gst_command(current_stream, resolution=resolution, model_name=model_name)
            else:
                gst_cmd = self.gen_gst_command(current_stream, resolution, codec, bitrate, model_name)

            avg_fps, status = self.run_gst_pipeline(gst_cmd)
            self.logger.info(f"Average fps is {avg_fps}")

            if status != 0:
                self.logger.error(f"Failed to run the pipeline with {current_stream} streams")

            if self._meet_target_fps(avg_fps):
                if current_stream < ref_stream:
                    low = current_stream + 1
                else:
                    quick_search = False
                result = current_stream
                self.update_telemetry()
            else:
                if current_stream < ref_stream:
                    high = current_stream - 1
                else:
                    break

            if low > high:
                break

        if result == 0:
            self.telemetry_list = [-1] * 8
        else:
            if self.device == "CPU":
                if float(self.telemetry_list[1]) / os.cpu_count() < 25:
                    result = f"{result}(cpu_usage < 25%)"

        if self.benchmark_name == "LPR Benchmark":
            gst_cmd = self.gen_gst_command(result, resolution=resolution, model_name=model_name)
            avg_fps, status = self.run_gst_pipeline(gst_cmd)
            self.logger.info(f"{self.benchmark_name} Average fps is {avg_fps}")
            result = f"{avg_fps}@{result}"

        return result

    def run_benchmark(self):
        """Run complete benchmark suite. Must be implemented by subclass."""
        raise NotImplementedError

    def loop_video(self):
        """Execute video looping commands for preparation."""
        with open(self.log_file, "a") as f:
            for command in self.loop_video_commands:
                # Use run_command but redirect output to file
                result = run_command(command, capture_output=True)
                f.write(result.stdout)
                f.write(result.stderr)

    def prepare(self):
        """Prepare benchmark environment (loop videos, check plugins, etc.)."""
        self.loop_video()

        if self.device != "CPU":
            self.available_va_plugins = self.run_command(["bash", "check_va_plugins.sh"])

        if "dGPU" in self.device:
            self.get_dGPU_Info()

        self.check_VDBox()
        self.get_config_of_platform()
