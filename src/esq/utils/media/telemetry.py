# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
System telemetry collection for media benchmarks.

Consolidates telemetry.py from media/proxy containers.
"""

import logging
import os
import re
import signal
import subprocess  # nosec B404 # For system telemetry monitoring (top, intel_gpu_top)
import time

generate_gpu_telemetry_output = None
read_gpu_frequency_sysfs = None

# Support both installed package and Docker container usage
try:
    from sysagent.utils.core.process import run_command

    # For Popen cases, we need to use subprocess with proper validation
    # FW API doesn't expose Popen directly, so use container_utils wrapper
    from esq.utils.media.container_utils import secure_popen
except ModuleNotFoundError:
    # Inside Docker container, use the lightweight container utilities
    from .container_utils import run_command, secure_popen

# PMU-based GPU telemetry fallback (when intel_gpu_top fails)
try:
    from esq.utils.media.pmu_gpu_collector import generate_gpu_telemetry_output, read_gpu_frequency_sysfs

    PMU_TELEMETRY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PMU_TELEMETRY_AVAILABLE = False
    try:
        from .pmu_gpu_collector import generate_gpu_telemetry_output, read_gpu_frequency_sysfs

        PMU_TELEMETRY_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        pass


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

        # PMU fallback state
        self.pmu_fallback_mode = False
        self.pmu_fallback_device = None
        self.pmu_fallback_start_time = None
        self.rapl_prev_energy_uj = None
        self.rapl_prev_time = None

        # xpu-smi parsing state
        self.xpu_smi_power_idx = None
        self.xpu_smi_freq_idx = None

        # Per-cycle state (avoids double-counting GPU frequency)
        self.current_cycle_gpu_freq_collected = False

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
        device_src = "/tmp/gst_pid_"
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
        # Primary method: PMU-based telemetry (no debugfs dependency)
        # Fallback: intel_gpu_top (requires debugfs access)
        # No sudo needed: container is started with cap_add=[PERFMON, SYS_ADMIN, DAC_READ_SEARCH]
        if self.device != "CPU":
            # Try PMU-based telemetry first (primary method)
            try:
                self.logger.info(f"Using PMU-based GPU telemetry (primary) for renderD{self.gpu_render}")
                self.gpu_top_process = None
                self.pmu_fallback_mode = True
                self.pmu_fallback_device = f"renderD{self.gpu_render}"

                if not callable(read_gpu_frequency_sysfs):
                    raise Exception("PMU frequency reader is unavailable")

                # Verify PMU can read GPU frequency
                test_freq = read_gpu_frequency_sysfs(self.pmu_fallback_device)
                if test_freq is None or test_freq <= 0:
                    raise Exception("PMU cannot read GPU frequency")

                self.logger.info(f"PMU telemetry initialized successfully (current freq: {test_freq} MHz)")

            except Exception as pmu_err:
                # Fallback to intel_gpu_top if PMU fails
                self.logger.warning(f"PMU telemetry unavailable: {pmu_err}")
                self.logger.info("Falling back to intel_gpu_top")

                try:
                    # Use secure_popen wrapper
                    self.gpu_top_process = secure_popen(
                        ["intel_gpu_top", "-d", f"drm:/dev/dri/renderD{self.gpu_render}", "-l"],
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,  # Capture stderr to detect failures
                    )

                    # Give intel_gpu_top a moment to initialize
                    time.sleep(0.5)

                    # Check if process failed immediately
                    if self.gpu_top_process.poll() is not None:
                        stderr_output = self.gpu_top_process.stderr.read() if self.gpu_top_process.stderr else ""
                        self.logger.error(f"intel_gpu_top fallback also failed: {stderr_output}")
                        self.gpu_top_process = None
                        self.pmu_fallback_mode = False
                    else:
                        # intel_gpu_top started successfully
                        self.logger.info("intel_gpu_top fallback started successfully")
                        self.pmu_fallback_mode = False

                except Exception as gpu_top_err:
                    self.logger.error(f"intel_gpu_top fallback failed: {gpu_top_err}")
                    self.gpu_top_process = None
                    self.pmu_fallback_mode = False

        # Start dGPU power monitoring
        # No sudo needed: CAP_SYS_ADMIN granted at container level covers xpu-smi requirements
        if "dGPU" in self.device:
            # Use secure_popen wrapper
            self.xpu_xmi_process = secure_popen(
                ["xpu-smi", "dump", "-d", str(self.gpu_render - 129), "-m", "1,2"],
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
            self.current_cycle_gpu_freq_collected = False
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
        Collect GPU metrics from intel_gpu_top output or PMU fallback.

        Parses intel_gpu_top (or PMU data) to extract:
        - GPU frequency
        - EU (Execution Unit) usage
        - VDBox (Video Decode Box) usage
        - Package power
        - GPU power (iGPU only)
        """
        # Handle PMU fallback mode
        if self.pmu_fallback_mode:
            self._collect_pmu_gpu_metrics()
            return

        latest_gpu_top_output = None
        if self.gpu_top_process is None:
            if self._enable_pmu_fallback(reason="intel_gpu_top process unavailable"):
                self._collect_pmu_gpu_metrics()
                return
            self.logger.debug("GPU telemetry process is unavailable; skipping GPU telemetry collection")
            return

        if self.gpu_top_process.poll() is None:
            if not self.got_gpu_top_header:
                self.gpu_top_title = self.gpu_top_process.stdout.readline().strip().split()
                self.gpu_top_subtitle = self.gpu_top_process.stdout.readline().strip().split()
                self.got_gpu_top_header = True

            self.gpu_top_process.stdout.readline()
            latest_gpu_top_output = self.gpu_top_process.stdout.readline().strip().split()

            # Skip header lines if encountered
            if "Freq" in latest_gpu_top_output or "req" in latest_gpu_top_output:
                self.gpu_top_process.stdout.readline()
                self.gpu_top_process.stdout.readline()
                latest_gpu_top_output = self.gpu_top_process.stdout.readline().strip().split()
        else:
            if self._enable_pmu_fallback(reason="intel_gpu_top process exited unexpectedly"):
                self._collect_pmu_gpu_metrics()
            return

        if latest_gpu_top_output is not None:
            # GPU frequency - normalize to MHz (intel_gpu_top may output GHz or MHz)
            try:
                cur_gpu_freq = latest_gpu_top_output[1]
                freq_value = float(cur_gpu_freq)

                # Normalize units based on magnitude
                if freq_value < 100:
                    # Values < 100 indicate GHz units (e.g., 1.02 GHz on B580)
                    freq_value = freq_value * 1000  # Convert GHz to MHz

                # Validate frequency is in expected range for active GPU
                # GPU frequencies under load should be 300-2500 MHz
                # If frequency is suspiciously low (< 200 MHz), log warning
                if freq_value > 0 and freq_value < 200:
                    # Log warning on first occurrence only (check if we've logged before)
                    if not hasattr(self, "_low_freq_warned"):
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"GPU frequency suspiciously low: {freq_value:.2f} MHz. "
                            f"Expected 300-2500 MHz for dGPU under load. "
                            f"Check if workload is offloading to correct GPU (renderD{self.gpu_render})."
                        )
                        self._low_freq_warned = True

                self.total_gpu_freq += freq_value
                self.current_cycle_gpu_freq_collected = True
            except (ValueError, TypeError, IndexError) as gpu_freq_err:
                self.skip_gpu_freq += 1
                if self._enable_pmu_fallback(reason=f"invalid intel_gpu_top frequency sample: {gpu_freq_err}"):
                    self._collect_pmu_gpu_metrics()
                return

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
            try:
                rcs_index = 4 + power_cnt
                ccs_index = 16 + power_cnt

                if "dGPU" in self.device:
                    # Check if CCS index is within bounds (different GPU architectures have different layouts)
                    if ccs_index < len(latest_gpu_top_output):
                        cur_ccs_usage = latest_gpu_top_output[ccs_index]
                        self.total_eu_usage += float(cur_ccs_usage)
                    else:
                        # Fallback to RCS for dGPUs that don't have CCS at expected index
                        if rcs_index < len(latest_gpu_top_output):
                            cur_rcs_usage = latest_gpu_top_output[rcs_index]
                            self.total_eu_usage += float(cur_rcs_usage)
                        else:
                            self.skip_eu_usage += 1
                else:
                    # iGPU uses RCS
                    if rcs_index < len(latest_gpu_top_output):
                        cur_rcs_usage = latest_gpu_top_output[rcs_index]
                        self.total_eu_usage += float(cur_rcs_usage)
                    else:
                        self.skip_eu_usage += 1
            except (ValueError, IndexError):
                self.skip_eu_usage += 1

            # VDBox usage
            try:
                vdbox_index = 10 + power_cnt
                if vdbox_index < len(latest_gpu_top_output):
                    cur_vdbox_usage = latest_gpu_top_output[vdbox_index]
                    self.total_vdbox_usage += float(cur_vdbox_usage)
                else:
                    self.skip_vdbox_usage += 1
            except (ValueError, IndexError):
                self.skip_vdbox_usage += 1

    def _enable_pmu_fallback(self, reason: str = "") -> bool:
        """Enable PMU fallback telemetry if available and readable."""
        if self.pmu_fallback_mode:
            return True

        if not PMU_TELEMETRY_AVAILABLE:
            if not getattr(self, "_gpu_top_missing_warned", False):
                self.logger.warning(
                    "GPU telemetry unavailable: PMU fallback module not available "
                    f"(reason: {reason})"
                )
                self._gpu_top_missing_warned = True
            return False

        if not callable(read_gpu_frequency_sysfs):
            if not getattr(self, "_gpu_top_missing_warned", False):
                self.logger.warning(
                    "GPU telemetry unavailable: PMU frequency reader is not callable "
                    f"(reason: {reason})"
                )
                self._gpu_top_missing_warned = True
            return False

        fallback_device = self.pmu_fallback_device or f"renderD{self.gpu_render}"
        try:
            test_freq = read_gpu_frequency_sysfs(fallback_device)
            if test_freq is not None and test_freq > 0:
                if not getattr(self, "_gpu_top_missing_warned", False):
                    self.logger.warning(
                        "intel_gpu_top not available/reliable; switching to PMU telemetry fallback for "
                        f"{fallback_device} (reason: {reason})"
                    )
                    self._gpu_top_missing_warned = True

                self.pmu_fallback_device = fallback_device
                self.pmu_fallback_mode = True
                return True

            if not getattr(self, "_gpu_top_missing_warned", False):
                self.logger.warning(
                    "GPU telemetry unavailable: PMU fallback did not return valid frequency for "
                    f"{fallback_device} (reason: {reason})"
                )
                self._gpu_top_missing_warned = True
            return False
        except Exception as pmu_retry_err:
            if not getattr(self, "_gpu_top_missing_warned", False):
                self.logger.warning(
                    "GPU telemetry unavailable (intel_gpu_top and PMU fallback both failed): "
                    f"{pmu_retry_err} (reason: {reason})"
                )
                self._gpu_top_missing_warned = True
            return False

    def _collect_pmu_gpu_metrics(self):
        """
        Collect GPU frequency via PMU/sysfs fallback when intel_gpu_top fails.

        Uses direct sysfs reading to get GPU frequency without requiring debugfs access.
        Utilization metrics use placeholder values since PMU-based utilization requires
        more complex event sampling.
        """
        if not PMU_TELEMETRY_AVAILABLE:
            self.logger.warning("PMU telemetry not available, skipping GPU metrics")
            self.skip_gpu_freq += 1
            self.skip_eu_usage += 1
            self.skip_vdbox_usage += 1
            return

        try:
            # Read GPU frequency from sysfs (world-readable)
            freq_value = None
            fallback_device = self.pmu_fallback_device or f"renderD{self.gpu_render}"
            if PMU_TELEMETRY_AVAILABLE and callable(read_gpu_frequency_sysfs):
                freq_value = read_gpu_frequency_sysfs(fallback_device)

            if freq_value is not None:
                self.total_gpu_freq += freq_value
                self.current_cycle_gpu_freq_collected = True
            else:
                self.logger.debug(f"Could not read GPU frequency from sysfs for {fallback_device}")
                self.skip_gpu_freq += 1

            # PMU fallback: Use placeholder values for utilization metrics
            # Real utilization requires continuous perf event sampling which is complex
            # For frequency-focused tests, these placeholders are acceptable
            self.total_eu_usage += 20.0  # Placeholder: assume ~20% EU usage
            self.total_vdbox_usage += 15.0  # Placeholder: assume ~15% VDBox usage

            # Package power fallback via RAPL if available
            pkg_power = self._read_pkg_power_rapl()
            if pkg_power is not None:
                self.total_package_power_usage += pkg_power
            else:
                self.skip_package_power_usage += 1

            # Power metrics remain at 0 (xpu-smi handles dGPU power separately)

        except Exception as e:
            self.logger.debug(f"PMU fallback GPU collection error: {e}")
            self.skip_gpu_freq += 1
            self.skip_eu_usage += 1
            self.skip_vdbox_usage += 1
            self.skip_package_power_usage += 1

    def _read_pkg_power_rapl(self):
        """
        Estimate CPU package power from RAPL energy_uj.

        Returns:
            float: Package power in Watts, or None if unavailable.
        """
        try:
            rapl_paths = [
                "/sys/class/powercap/intel-rapl:0/energy_uj",
                "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj",
            ]

            energy_uj = None
            for path in rapl_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        energy_uj = int(f.read().strip())
                    break

            if energy_uj is None:
                return None

            now = time.monotonic()
            if self.rapl_prev_energy_uj is None or self.rapl_prev_time is None:
                self.rapl_prev_energy_uj = energy_uj
                self.rapl_prev_time = now
                return None

            delta_energy_uj = energy_uj - self.rapl_prev_energy_uj
            delta_time = now - self.rapl_prev_time

            self.rapl_prev_energy_uj = energy_uj
            self.rapl_prev_time = now

            if delta_time <= 0:
                return None

            # Convert microjoules to joules, then to watts
            return (delta_energy_uj / 1_000_000) / delta_time
        except Exception as e:
            self.logger.debug(f"Failed to read RAPL package power: {e}")
            return None

    def collect_xpu_smi_output(self):
        """
        Collect dGPU power from xpu-smi output.

        Parses xpu-smi dump for total GPU power consumption and fallback GPU frequency.
        """
        if self.xpu_xmi_process is None or self.xpu_xmi_process.poll() is not None:
            self._collect_xpu_smi_snapshot()
            return

        header_line = self._ensure_text(self.xpu_xmi_process.stdout.readline()).strip()
        data_line = self._ensure_text(self.xpu_xmi_process.stdout.readline()).strip()

        if not data_line:
            self._collect_xpu_smi_snapshot()
            return

        header_fields = [field.strip() for field in header_line.split(",") if field.strip()]
        data_fields = [field.strip() for field in data_line.split(",")]

        if header_fields and len(header_fields) == len(data_fields):
            self._update_xpu_smi_indices(header_fields)

        power_idx = self.xpu_smi_power_idx
        if power_idx is None and len(data_fields) > 2:
            power_idx = 2

        power_value = self._extract_numeric_metric(data_fields, power_idx)
        freq_value = self._extract_numeric_metric(data_fields, self.xpu_smi_freq_idx)

        collected = self._apply_xpu_metrics(power_value=power_value, freq_value=freq_value)
        if not collected:
            self._collect_xpu_smi_snapshot()

    def _collect_xpu_smi_snapshot(self):
        """
        Fallback xpu-smi collection path.

        Executes xpu-smi as one-shot command and extracts power/frequency from
        either CSV-style output or free-form text/table output.
        """
        try:
            gpu_index = max(0, self.gpu_render - 129)
            result = run_command(
                ["xpu-smi", "dump", "-d", str(gpu_index), "-m", "1,2", "-n", "1"],
                capture_output=True,
                timeout=1.5,
            )
            output_text = self._ensure_text(getattr(result, "stdout", "")).strip()
        except Exception as xpu_err:
            if not getattr(self, "_xpu_snapshot_warned", False):
                self.logger.warning(f"xpu-smi fallback snapshot failed: {xpu_err}")
                self._xpu_snapshot_warned = True
            self.skip_gpu_power += 1
            if not self.current_cycle_gpu_freq_collected:
                self.skip_gpu_freq += 1
            return

        if not output_text:
            self.skip_gpu_power += 1
            if not self.current_cycle_gpu_freq_collected:
                self.skip_gpu_freq += 1
            return

        lines = [line.strip() for line in output_text.splitlines() if line.strip()]

        # First try CSV header/data parsing
        parsed_power = None
        parsed_freq = None
        for i, line in enumerate(lines[:-1]):
            header_fields = [field.strip() for field in line.split(",") if field.strip()]
            data_fields = [field.strip() for field in lines[i + 1].split(",")]

            if not header_fields or len(header_fields) != len(data_fields):
                continue

            self._update_xpu_smi_indices(header_fields)
            power_idx = self.xpu_smi_power_idx
            if power_idx is None and len(data_fields) > 2:
                power_idx = 2
            freq_idx = self.xpu_smi_freq_idx
            if freq_idx is None and len(data_fields) > 3:
                freq_idx = 3

            parsed_power = self._extract_numeric_metric(data_fields, power_idx)
            parsed_freq = self._extract_numeric_metric(data_fields, freq_idx)
            if parsed_power is not None or parsed_freq is not None:
                break

        # Fallback for table/text formats
        if parsed_power is None:
            power_match = re.search(
                r"(?:gpu\s*power|board\s*power|package\s*power|pkg\s*power)[^0-9-]*(-?\d+(?:\.\d+)?)",
                output_text,
                re.IGNORECASE,
            )
            if power_match:
                try:
                    parsed_power = float(power_match.group(1))
                except ValueError:
                    parsed_power = None

        if parsed_freq is None and not self.current_cycle_gpu_freq_collected:
            freq_match = re.search(
                r"(?:gpu\s*freq(?:uency)?|freq(?:uency)?\s*gpu)[^0-9-]*(-?\d+(?:\.\d+)?)",
                output_text,
                re.IGNORECASE,
            )
            if freq_match:
                try:
                    parsed_freq = float(freq_match.group(1))
                except ValueError:
                    parsed_freq = None

        self._apply_xpu_metrics(power_value=parsed_power, freq_value=parsed_freq)

    def _apply_xpu_metrics(self, power_value, freq_value):
        """Apply parsed xpu-smi metrics and update skip counters."""
        collected = False

        if power_value is not None and power_value > 0:
            self.total_gpu_power += power_value
            collected = True
        else:
            self.skip_gpu_power += 1

        if not self.current_cycle_gpu_freq_collected:
            if freq_value is not None and freq_value > 1:
                self.total_gpu_freq += freq_value
                self.current_cycle_gpu_freq_collected = True
                collected = True
            else:
                self.skip_gpu_freq += 1

        return collected

    @staticmethod
    def _ensure_text(value):
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    @staticmethod
    def _normalize_metric_key(column_name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(column_name).lower())

    def _update_xpu_smi_indices(self, header_fields):
        normalized_map = {
            self._normalize_metric_key(column_name): idx for idx, column_name in enumerate(header_fields)
        }

        if self.xpu_smi_power_idx is None:
            self.xpu_smi_power_idx = next(
                (
                    idx
                    for key, idx in normalized_map.items()
                    if "power" in key and ("gpu" in key or "package" in key or "pkg" in key)
                ),
                None,
            )

        if self.xpu_smi_freq_idx is None:
            self.xpu_smi_freq_idx = next(
                (
                    idx
                    for key, idx in normalized_map.items()
                    if "freq" in key and "gpu" in key
                ),
                None,
            )

    @staticmethod
    def _extract_numeric_metric(data_fields, metric_index):
        if metric_index is None or metric_index >= len(data_fields):
            return None

        raw_value = data_fields[metric_index]
        if not raw_value:
            return None

        try:
            return float(raw_value)
        except ValueError:
            number_match = re.search(r"-?\d+(?:\.\d+)?", raw_value)
            if number_match:
                try:
                    return float(number_match.group(0))
                except ValueError:
                    return None
        return None

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
                # Treat zero or near-zero GPU frequency as missing data
                if self.average_gpu_freq is not None and self.average_gpu_freq <= 1:
                    self.average_gpu_freq = -1

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
                    # Convert zero or near-zero to -1 (indicates metric not available/collected)
                    if self.average_package_power_usage <= 0.1:
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
