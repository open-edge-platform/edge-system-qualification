# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System telemetry collection for media benchmarks.

Consolidates telemetry.py from media/proxy containers.
"""

import logging
import os
import signal
import subprocess  # nosec B404
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


class TelemetryCollector:
    """
    Collect system telemetry during benchmark execution.

    Monitors CPU, GPU, memory, and power metrics using system tools:
    - top: CPU usage and memory
    - intel_gpu_top: GPU frequency, EU usage, VDBox usage, power
    - xpu-smi: dGPU power consumption
    """

    def __init__(self, device, gpu_render, telemetry_file):
        """
        Initialize telemetry collector.

        Args:
            device: Target device (CPU, iGPU, dGPU.0, dGPU.1)
            gpu_render: Render device number (128 for iGPU, 129+ for dGPU)
            telemetry_file: Output file path for telemetry results
        """
        self.device = device
        self.gpu_render = gpu_render
        self.telemetry_file = telemetry_file

        self.got_gpu_top_header = False
        self.stop_collecting = False

        # Telemetry accumulators
        self.total_cnt = 0
        self.total_cpu_freq = 0
        self.total_cpu_usage = 0
        self.total_system_memory_usage = 0
        self.total_gpu_freq = 0
        self.total_eu_usage = 0
        self.total_vdbox_usage = 0
        self.total_package_power_usage = 0
        self.total_gpu_power = 0

        # Skip counters for invalid samples
        self.skip_cpu_usage = 0
        self.skip_system_memory_usage = 0
        self.skip_gpu_freq = 0
        self.skip_eu_usage = 0
        self.skip_vdbox_usage = 0
        self.skip_package_power_usage = 0
        self.skip_gpu_power = 0

        # Average results
        self.average_cpu_freq = 0
        self.avergae_cpu_usage = 0
        self.average_system_memory_usage = 0
        self.average_gpu_freq = 0
        self.average_eu_usage = 0
        self.average_vdbox_usage = 0
        self.average_package_power_usage = 0
        self.average_gpu_power = 0

        self.logger = logging.getLogger(self.__class__.__name__)

        # Process handles
        self.top_process = None
        self.gpu_top_process = None
        self.xpu_xmi_process = None

    def signal_handler(self, signum, frame):
        """Handle SIGUSR1 to stop collection."""
        self.stop_collecting = True

    def start_telemetries(self):
        """
        Start telemetry collection processes.

        Waits for GStreamer PID file to exist, then starts:
        - top for CPU monitoring
        - intel_gpu_top for GPU monitoring (non-CPU devices)
        - xpu-smi for dGPU power monitoring
        """
        signal.signal(signal.SIGUSR1, self.signal_handler)

        # Wait for GStreamer process PID file
        device_src = "/tmp/gst_pid_"  # nosec
        while not os.path.exists(f"{device_src}{self.device}.txt"):
            time.sleep(1)

        with open(f"{device_src}{self.device}.txt", "r") as f:
            gst_pid = int(f.read().strip())

        # Start CPU monitoring using secure_popen wrapper
        self.top_process = secure_popen(
            ["top", "-p", str(gst_pid), "-b", "-d", "1"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # Start GPU monitoring
        if self.device != "CPU":
            # Use secure_popen wrapper
            self.gpu_top_process = secure_popen(
                ["sudo", "intel_gpu_top", "-d", f"drm:/dev/dri/renderD{self.gpu_render}", "-l"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

        # Start dGPU power monitoring
        if "dGPU" in self.device:
            # Use secure_popen wrapper
            self.xpu_xmi_process = secure_popen(
                ["sudo", "xpu-smi", "dump", "-d", str(self.gpu_render - 129), "-m", "1"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

    def collect_telemetries(self):
        """
        Main telemetry collection loop.

        Runs until SIGUSR1 received, collecting metrics every second.
        """
        self.start_telemetries()
        time.sleep(1)

        while not self.stop_collecting:
            self.collect_cpu_freq()
            self.collect_top_output()
            if self.device != "CPU":
                self.collect_gpu_top_output()
                if "dGPU" in self.device:
                    self.collect_xpu_smi_output()

            self.total_cnt += 1
            time.sleep(1)

        self.calculate_averages()
        self.write_result_to_file()
        self.cleanup()

    def _get_average_cpu_freq(self):
        """
        Get average CPU frequency across all cores.

        Returns:
            float: Average frequency in MHz
        """
        freq = run_command(["cat", "/proc/cpuinfo"], capture_output=True)
        cpu_freqs = [float(line.split(":")[1].strip()) for line in freq.stdout.splitlines() if "MHz" in line]
        return sum(cpu_freqs) / len(cpu_freqs)

    def collect_cpu_freq(self):
        """Collect current CPU frequency."""
        cur_cpu_freq = self._get_average_cpu_freq()
        self.total_cpu_freq += cur_cpu_freq

    def collect_top_output(self):
        """
        Collect CPU usage and memory from top output.

        Parses top output to extract CPU% and MEM% for tracked process.
        """
        header_found = False
        cur_top_result = ""

        while self.top_process.poll() is None:
            if header_found:
                cur_top_result = self.top_process.stdout.readline()
                break
            else:
                cur_line = self.top_process.stdout.readline().strip().split()
                if "%CPU" in cur_line:
                    header_found = True

        try:
            cur_cpu_usage = cur_top_result.strip().split()[8]
            cur_system_memory_usage = cur_top_result.strip().split()[9]
        except IndexError:
            cur_cpu_usage = 0
            cur_system_memory_usage = 0
            self.logger.debug(f"IndexError, current top result: {cur_top_result.strip()}")
            self.logger.debug(f"total cpu usage: {self.total_cpu_usage}")
            self.logger.debug(f"total cnt: {self.total_cnt}")

        try:
            self.total_cpu_usage += float(cur_cpu_usage)
        except ValueError:
            self.skip_cpu_usage += 1

        try:
            self.total_system_memory_usage += float(cur_system_memory_usage)
        except ValueError:
            self.skip_system_memory_usage += 1

    def collect_gpu_top_output(self):
        """
        Collect GPU metrics from intel_gpu_top output.

        Parses intel_gpu_top to extract:
        - GPU frequency
        - EU (Execution Unit) usage
        - VDBox (Video Decode Box) usage
        - Package power
        - GPU power (iGPU only)
        """
        latest_gpu_top_output = None
        if self.gpu_top_process.poll() is None:
            if not self.got_gpu_top_header:
                self.gpu_top_title = self.gpu_top_process.stdout.readline().strip().split()
                self.gpu_top_subtitle = self.gpu_top_process.stdout.readline().strip().split()
                self.got_gpu_top_header = True

            ignore_line = self.gpu_top_process.stdout.readline()
            latest_gpu_top_output = self.gpu_top_process.stdout.readline().strip().split()

            # Skip header lines if encountered
            if "Freq" in latest_gpu_top_output or "req" in latest_gpu_top_output:
                ignore_line = self.gpu_top_process.stdout.readline()
                ignore_line = self.gpu_top_process.stdout.readline()
                latest_gpu_top_output = self.gpu_top_process.stdout.readline().strip().split()

        if latest_gpu_top_output is not None:
            # GPU frequency
            cur_gpu_freq = latest_gpu_top_output[1]
            try:
                self.total_gpu_freq += float(cur_gpu_freq)
            except ValueError:
                self.skip_gpu_freq += 1

            # Power metrics
            power_cnt = 0
            if "gpu" in self.gpu_top_subtitle:
                power_cnt += 1
                if self.device == "iGPU":
                    cur_gpu_power = latest_gpu_top_output[4]
                    try:
                        self.total_gpu_power += float(cur_gpu_power)
                    except ValueError:
                        self.skip_gpu_power += 1

            if "pkg" in self.gpu_top_subtitle:
                power_cnt += 1
                cur_package_power_usage = latest_gpu_top_output[5]
                try:
                    self.total_package_power_usage += float(cur_package_power_usage)
                except ValueError:
                    self.skip_package_power_usage += 1

            # EU usage (RCS for iGPU, CCS for dGPU)
            cur_rcs_usage = latest_gpu_top_output[4 + power_cnt]
            try:
                if "dGPU" in self.device:
                    cur_ccs_usage = latest_gpu_top_output[16 + power_cnt]
                    self.total_eu_usage += float(cur_ccs_usage)
                else:
                    self.total_eu_usage += float(cur_rcs_usage)
            except ValueError:
                self.skip_eu_usage += 1

            # VDBox usage
            cur_vdbox_usage = latest_gpu_top_output[10 + power_cnt]
            try:
                self.total_vdbox_usage += float(cur_vdbox_usage)
            except ValueError:
                self.skip_vdbox_usage += 1

    def collect_xpu_smi_output(self):
        """
        Collect dGPU power from xpu-smi output.

        Parses xpu-smi dump for total GPU power consumption.
        """
        latest_xpu_xmi_output = None
        if self.xpu_xmi_process.poll() is None:
            header = self.xpu_xmi_process.stdout.readline()
            latest_xpu_xmi_output = self.xpu_xmi_process.stdout.readline().strip().split(",")

        if latest_xpu_xmi_output is not None:
            cur_total_gpu_power_usage = latest_xpu_xmi_output[2]
            try:
                self.total_gpu_power += float(cur_total_gpu_power_usage)
            except ValueError:
                self.skip_gpu_power += 1

    def calculate_averages(self):
        """Calculate average values from collected telemetry."""
        try:
            self.average_cpu_freq = self.total_cpu_freq / self.total_cnt
            self.avergae_cpu_usage = self.total_cpu_usage / (self.total_cnt - self.skip_cpu_usage)
            self.average_system_memory_usage = self.total_system_memory_usage / (
                self.total_cnt - self.skip_system_memory_usage
            )

            if self.device != "CPU":
                # Calculate average only if we have valid samples (not all skipped)
                valid_gpu_freq = self.total_cnt - self.skip_gpu_freq
                self.average_gpu_freq = self.total_gpu_freq / valid_gpu_freq if valid_gpu_freq > 0 else -1

                valid_eu_usage = self.total_cnt - self.skip_eu_usage
                if valid_eu_usage > 0:
                    self.average_eu_usage = self.total_eu_usage / valid_eu_usage
                    # Convert 0.0 to -1 (indicates metric not available/collected)
                    if self.average_eu_usage == 0.0:
                        self.average_eu_usage = -1
                else:
                    self.average_eu_usage = -1

                valid_vdbox_usage = self.total_cnt - self.skip_vdbox_usage
                if valid_vdbox_usage > 0:
                    self.average_vdbox_usage = self.total_vdbox_usage / valid_vdbox_usage
                    # Convert 0.0 to -1 (indicates metric not available/collected)
                    if self.average_vdbox_usage == 0.0:
                        self.average_vdbox_usage = -1
                else:
                    self.average_vdbox_usage = -1

                valid_package_power = self.total_cnt - self.skip_package_power_usage
                if valid_package_power > 0:
                    self.average_package_power_usage = self.total_package_power_usage / valid_package_power
                    # Convert 0.0 to -1 (indicates metric not available/collected)
                    if self.average_package_power_usage == 0.0:
                        self.average_package_power_usage = -1
                else:
                    self.average_package_power_usage = -1

                valid_gpu_power = self.total_cnt - self.skip_gpu_power
                self.average_gpu_power = self.total_gpu_power / valid_gpu_power if valid_gpu_power > 0 else -1
            else:
                self.average_gpu_freq = -1
                self.average_eu_usage = -1
                self.average_vdbox_usage = -1
                self.average_package_power_usage = -1
                self.average_gpu_power = -1
        except ZeroDivisionError as e:
            self.logger.debug(f"Divided by zero error: {e}")
            self.logger.debug(f"total cnt: {self.total_cnt}")

    def write_result_to_file(self):
        """Write average telemetry results to output file."""
        with open(self.telemetry_file, "w") as f:
            f.write(f"Average CPU Frequency: {self.average_cpu_freq:.2f}\n")
            f.write(f"Average CPU Usage: {self.avergae_cpu_usage:.2f}\n")
            f.write(f"Average System Memory Usage: {self.average_system_memory_usage:.2f}\n")

            if self.device != "CPU":
                f.write(f"Average GPU Frequency: {self.average_gpu_freq:.2f}\n")
                f.write(f"Average EU Usage: {self.average_eu_usage:.2f}\n")
                f.write(f"Average VDBox Usage: {self.average_vdbox_usage:.2f}\n")
                f.write(f"Average Package Power Usage: {self.average_package_power_usage:.2f}\n")
                f.write(f"Average GPU Power: {self.average_gpu_power:.2f}\n")
            else:
                f.write(f"Average GPU Frequency: {self.average_gpu_freq:.2f}\n")
                f.write(f"Average EU Usage: {self.average_eu_usage:.2f}\n")
                f.write(f"Average VDBox Usage: {self.average_vdbox_usage:.2f}\n")
                f.write(f"Average Package Power Usage: {self.average_package_power_usage:.2f}\n")
                f.write(f"Average GPU Power: {self.average_gpu_power:.2f}\n")

    def cleanup(self):
        """Terminate telemetry collection processes."""
        if self.top_process and self.top_process.poll() is None:
            self.top_process.terminate()
            self.top_process.wait()

        if self.gpu_top_process and self.gpu_top_process.poll() is None:
            self.gpu_top_process.terminate()
            self.gpu_top_process.wait()

        if "dGPU" in self.device:
            if self.xpu_xmi_process and self.xpu_xmi_process.poll() is None:
                self.xpu_xmi_process.terminate()
                self.xpu_xmi_process.wait()


Telemetry = TelemetryCollector
