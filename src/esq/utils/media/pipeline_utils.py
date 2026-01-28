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
import subprocess  # nosec B404 # Popen needed for background pipeline execution
import time

# Support both installed package and Docker container usage
try:
    from sysagent.utils.core.process import run_command

    # Import adaptive search for FPS-guided stream count optimization
    from esq.utils.media.adaptive_search import (
        AdaptiveSearchConfig,
        fps_guided_adaptive_search,
        get_initial_stream_count,
    )

    # For Popen cases, we need to use subprocess with proper validation
    # FW API doesn't expose Popen directly, so use container_utils wrapper
    from esq.utils.media.container_utils import secure_popen

    ADAPTIVE_SEARCH_AVAILABLE = True
except ModuleNotFoundError:
    # Inside Docker container, use the lightweight container utilities
    from .container_utils import run_command, secure_popen

    # Import adaptive search from container path
    try:
        from .adaptive_search import (
            AdaptiveSearchConfig,
            fps_guided_adaptive_search,
            get_initial_stream_count,
        )

        ADAPTIVE_SEARCH_AVAILABLE = True
    except ImportError:
        ADAPTIVE_SEARCH_AVAILABLE = False


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

    def get_gst_elements(self, codec, use_generic_encoder=False):
        """
        Get GStreamer element names based on device and codec.

        Args:
            codec: Codec name (h264, h265, av1, vp9)
            use_generic_encoder: If True, use generic VA encoder (vah264enc) for dGPU
                instead of device-specific encoder. This is useful when the encoder
                receives system memory input (e.g., from compositor) where the generic
                encoder performs better than device-specific encoder.

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
                # dGPU: Use device-specific elements for decoder and postproc
                self.dec_ele = f"varenderD{self.gpu_render}{codec}dec"
                self.post_proc_ele = f"varenderD{self.gpu_render}postproc"
                # For encoder: use generic encoder when input is system memory
                # (e.g., from compositor), otherwise use device-specific encoder
                if use_generic_encoder:
                    self.enc_ele = f"va{codec}enc"
                else:
                    self.enc_ele = f"varenderD{self.gpu_render}{codec}enc"

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
        self.logger.info(f"[PROGRESS] Opening CSV file: {csv_path}")
        with open(csv_path, "r+") as f:
            # Acquire lock to ensure only 1 process can modify file
            self.logger.info("[PROGRESS] Acquiring file lock...")
            fcntl.flock(f, fcntl.LOCK_EX)
            self.logger.info("[PROGRESS] File lock acquired")
            try:
                pattern = re.compile(f"{prefix},.*")
                content = f.read()
                if pattern.search(content):
                    self.logger.info(f"Existed result for {prefix}, updating")
                    updated_content = pattern.sub(f"{cur_record}", content)

                    f.seek(0)
                    f.write(updated_content)
                    f.truncate()
                else:
                    self.logger.info(f"No record for {prefix}, adding new")
                    # Move to end of file
                    f.seek(0, os.SEEK_END)
                    f.write(f"{cur_record}\n")
                self.logger.info("[PROGRESS] CSV write complete")
            finally:
                # Release the lock
                fcntl.flock(f, fcntl.LOCK_UN)
                self.logger.info("[PROGRESS] File lock released")

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
                self.logger.info(f"[PROGRESS] Sending SIGUSR1 to telemetry process (PID: {self.tele_process.pid})...")
                os.kill(self.tele_process.pid, signal.SIGUSR1)
                self.logger.info("[PROGRESS] Waiting for telemetry process to join (timeout: 5s)...")
                self.tele_process.join(timeout=5)
                self.logger.info("[PROGRESS] Telemetry process joined successfully")
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
        else:
            self.logger.info("[PROGRESS] Telemetry process already stopped")

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

    def run_gst_pipeline(self, gst_cmd, expected_stream_count=None, iteration_timeout=600):
        """
        Execute GStreamer pipeline and parse FPS results.

        Args:
            gst_cmd: GStreamer command string
            expected_stream_count: Number of streams we expect in this test (for validation)
            iteration_timeout: Maximum time in seconds for a single iteration (default: 600s = 10 min)
                               If exceeded, pipeline is killed and failure status is returned
                               with captured CPU/GPU utilization for diagnostics.

        Returns:
            tuple: (avg_fps, status)
                avg_fps: Average FPS achieved
                status: 0 on success, 1 on failure
        """
        # Store expected stream count for parsing validation
        self._expected_stream_count = expected_stream_count

        # Only use unique filenames for VSaaS benchmark
        # VSaaS uses gvafpscounter which has state persistence issues across test runs
        # LPR/SmartNVR don't need this workaround
        use_unique_filename = expected_stream_count is not None and self.benchmark_name == "AI VSaaS Gateway Benchmark"
        if use_unique_filename:
            import time as time_module

            timestamp = int(time_module.time() * 1000)  # Milliseconds
            unique_result_file = f"{self.result_file}.{timestamp}"
            self.logger.debug(f"Using unique result file for VSaaS: {unique_result_file}")
            current_result_file = unique_result_file
        else:
            current_result_file = self.result_file
            self.logger.debug(f"Using standard result file: {current_result_file}")

        try:
            # Create/truncate file
            result_file_handle = open(current_result_file, "w")
            result_file_handle.flush()
            os.fsync(result_file_handle.fileno())

            file_size_before = os.path.getsize(current_result_file)
            self._last_file_size_before = file_size_before
            self._current_result_file = current_result_file  # Store for later use

            if file_size_before != 0:
                self.logger.error(f"ERROR: Newly created file is not empty! Size: {file_size_before} bytes")

            self.logger.debug(f"Created unique result file, size: {file_size_before} bytes")
        except Exception as e:
            self.logger.error(f"Failed to create result file: {e}")
            return 0.0, 1

        gst_process = secure_popen(
            ["bash", "./run_pipeline.sh", self.device, gst_cmd],
            text=True,
            stdout=result_file_handle,
            stderr=subprocess.STDOUT,
        )

        time.sleep(1)

        self.collect_telemetry()

        status = 0
        avg_fps = 0.0
        start_wait_time = time.time()
        last_progress_log = start_wait_time

        iteration_timeout_triggered = False
        try:
            while gst_process.poll() is None:
                current_time = time.time()
                elapsed = current_time - start_wait_time

                # Log progress every 30 seconds to show pipeline is still running
                if current_time - last_progress_log >= 30:
                    self.logger.info(f"[PROGRESS] Pipeline running... elapsed: {elapsed:.0f}s")
                    last_progress_log = current_time

                # Check for iteration timeout - prevents indefinite hangs on overloaded GPU
                if elapsed > iteration_timeout:
                    iteration_timeout_triggered = True
                    self.logger.error(f"[TIMEOUT] Iteration timeout ({iteration_timeout}s) exceeded!")
                    self.logger.error(f"[TIMEOUT] Streams: {expected_stream_count}, Elapsed: {elapsed:.0f}s")

                    # Capture current utilization from telemetry for diagnostics
                    try:
                        self.stop_telemetry()  # Stop and flush telemetry data
                        if os.path.isfile(self.telemetry_file):
                            with open(self.telemetry_file, "r") as tf:
                                telemetry_lines = tf.readlines()
                            # Parse telemetry: CPU Freq, CPU Usage, Mem Usage, GPU Freq, EU Usage, VDBox Usage, Pkg Power, GPU Power
                            telemetry_data = {}
                            for line in telemetry_lines:
                                if ":" in line:
                                    key, value = line.strip().split(":", 1)
                                    telemetry_data[key.strip()] = value.strip()
                            cpu_util = telemetry_data.get("CPU Usage", "N/A")
                            gpu_freq = telemetry_data.get("GPU Freq", "N/A")
                            eu_usage = telemetry_data.get("EU Usage", "N/A")
                            vdbox_usage = telemetry_data.get("VDBox Usage", "N/A")
                            self.logger.error(
                                f"[TIMEOUT] Utilization at timeout - CPU: {cpu_util}%, GPU Freq: {gpu_freq} MHz, EU: {eu_usage}%, VDBox: {vdbox_usage}%"
                            )
                        else:
                            self.logger.warning(f"[TIMEOUT] Telemetry file not found: {self.telemetry_file}")
                    except Exception as telem_err:
                        self.logger.warning(f"[TIMEOUT] Failed to capture utilization: {telem_err}")

                    # Kill the pipeline
                    self.logger.error(f"[TIMEOUT] Killing pipeline after {elapsed:.0f}s without valid FPS output")
                    gst_process.terminate()
                    try:
                        gst_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.logger.warning("[TIMEOUT] Process didn't terminate gracefully, forcing kill")
                        gst_process.kill()
                        gst_process.wait()
                    status = 1
                    break

                # Kill pipeline if no output for 120 seconds
                if os.path.isfile(self.result_file) and (time.time() - os.path.getmtime(self.result_file)) > 120:
                    with open(self.result_file, "r") as f:
                        if "overall" not in f.read():
                            status = 1
                    gst_process.terminate()
                    gst_process.wait()
                    self.logger.info("Pipeline killed after no output for 120s")
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

            # Close result file handle to ensure all data is flushed
            # and file can be properly truncated in next iteration
            try:
                result_file_handle.flush()  # Force flush before close
                os.fsync(result_file_handle.fileno())  # Sync to filesystem
                result_file_handle.close()
                self.logger.debug(f"Result file handle flushed, synced, and closed: {self.result_file}")
                # Longer delay to ensure filesystem and Docker volume sync complete
                time.sleep(0.5)
            except Exception as e:
                self.logger.warning(f"Error closing result file handle: {e}")

            # Only stop telemetry if not already stopped due to iteration timeout
            if not iteration_timeout_triggered:
                self.stop_telemetry()

        # Copy result file content before potential issues with next iteration
        # Read from the unique file we created for this iteration
        result_file_content = None
        current_file = getattr(self, "_current_result_file", self.result_file)

        try:
            # Verify file exists and is readable
            if not os.path.exists(current_file):
                self.logger.error(f"ERROR: Result file does not exist after pipeline: {current_file}")
                return 0.0, 1

            with open(current_file, "r") as f:
                result_file_content = f.read()
            file_size_after = os.path.getsize(current_file)

            # Quick sanity check on content
            num_overall_lines = result_file_content.count("FpsCounter(overall")
            self.logger.debug(
                f"Read {len(result_file_content)} bytes from result file for parsing "
                f"(file: {os.path.basename(current_file)}, size: {file_size_after} bytes, {num_overall_lines} overall lines)"
            )

            # Clean up old unique result files (only for VSaaS which uses unique filenames)
            try:
                if self.benchmark_name == "AI VSaaS Gateway Benchmark":
                    import glob

                    pattern = f"{self.result_file}.*"
                    old_files = glob.glob(pattern)
                    for old_file in old_files:
                        if old_file != current_file:
                            os.remove(old_file)
                            self.logger.debug(f"Cleaned up old VSaaS result file: {os.path.basename(old_file)}")
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to cleanup old result files: {cleanup_error}")

            # Sanity check: File should have grown from the truncated size
            if hasattr(self, "_last_file_size_before"):
                if file_size_after <= self._last_file_size_before:
                    self.logger.warning(
                        f"WARNING: Result file did not grow! Before: {self._last_file_size_before}, "
                        f"After: {file_size_after}. Truncation may have failed!"
                    )
        except Exception as e:
            self.logger.error(f"Failed to read result file: {e}")
            return 0.0, 1

        # Parse results from in-memory content (already read above)
        # For multi-stream tests, each stream has its own gvafpscounter that outputs:
        # - Periodic "last X.XXsec" updates
        # - Final "overall X.XXsec" summary at end
        # Format: "FpsCounter(overall X.XXsec): total=XXX.XX fps, number-streams=N, per-stream=XX.XX fps"
        # File may contain "overall" lines from previous stream counts during ramp-up.
        # We must ONLY collect lines where number-streams matches the MAXIMUM value (current test).

        fps_measurements = []
        overall_lines_by_stream_count = {}

        for line in result_file_content.splitlines():
            if "ERROR: from element" in line:
                status = 1
                break
            # Collect all "overall" lines and group by number-streams count
            if "FpsCounter(overall" in line and "number-streams=" in line:
                # Extract number-streams value to identify which test iteration this is from
                stream_match = re.search(r"number-streams=(\d+)", line)
                if stream_match:
                    stream_count = int(stream_match.group(1))
                    if stream_count not in overall_lines_by_stream_count:
                        overall_lines_by_stream_count[stream_count] = []
                    overall_lines_by_stream_count[stream_count].append(line)

        # Debug: Log what we found BEFORE filtering
        if overall_lines_by_stream_count:
            all_stream_counts = sorted(overall_lines_by_stream_count.keys())
            expected = getattr(self, "_expected_stream_count", "unknown")
            self.logger.debug(
                f"BEFORE filtering - Found stream counts in file: {all_stream_counts}, "
                f"Expected: {expected}, Total lines: {sum(len(v) for v in overall_lines_by_stream_count.values())}"
            )

            # Stale data filtering: Only applies to VSaaS benchmark.
            # VSaaS uses binary search with varying stream counts and gvafpscounter state
            # can persist between runs causing stale data issues.
            # Other benchmarks (SmartNVR, Headed Visual AI, LPR) use fixed compose grids
            # where the FPS counter reports total grid size (e.g., 25 for 5x5, 36 for 6x6),
            # which should NOT be filtered as stale data.
            is_vsaas = self.benchmark_name == "AI VSaaS Gateway Benchmark"
            if is_vsaas and hasattr(self, "_expected_stream_count") and self._expected_stream_count:
                # Stale data handling: Filter out any stream counts HIGHER than expected.
                # Higher counts indicate leftover data from a previous run with more streams.
                # Instead of failing hard, we filter them out and use the remaining valid data.
                max_found = max(all_stream_counts)
                if max_found > self._expected_stream_count:
                    # Found stale data - filter it out instead of failing
                    stale_counts = [c for c in all_stream_counts if c > self._expected_stream_count]
                    valid_counts = [c for c in all_stream_counts if c <= self._expected_stream_count]
                    self.logger.warning(
                        f"Detected stale data: counts {stale_counts} exceed expected "
                        f"{self._expected_stream_count}. Using valid counts: {valid_counts}"
                    )
                    # Remove stale entries from the dictionary
                    for stale_count in stale_counts:
                        del overall_lines_by_stream_count[stale_count]

                    # After filtering, check if we have any valid data left
                    if not overall_lines_by_stream_count:
                        self.logger.error(
                            "No valid data remaining after filtering stale stream counts. "
                            "All data was from previous runs with higher stream counts."
                        )
                        return 0.0, 1

                    # Update all_stream_counts after filtering
                    all_stream_counts = sorted(overall_lines_by_stream_count.keys())
                    self.logger.info(f"After filtering stale data, valid stream counts: {all_stream_counts}")

                # Check if expected stream count is now available
                if self._expected_stream_count not in all_stream_counts:
                    # Expected stream count not found. This means some streams haven't finished
                    # warmup (starting-frame not reached). Use highest available instead of failing.
                    max_valid = max(all_stream_counts)
                    self.logger.warning(
                        f"Expected stream count {self._expected_stream_count} not found in result file. "
                        f"Found: {all_stream_counts}. Some streams may not have reached starting-frame threshold."
                    )
                    self.logger.info(f"Using highest available stream count: {max_valid}")
        else:
            self.logger.warning("No FpsCounter(overall) lines found in result file!")

        # Use the EXPECTED stream count if available (VSaaS), otherwise fall back to MAX (LPR/SmartNVR)
        if overall_lines_by_stream_count:
            # Use expected stream count if we have it, otherwise use max
            if (
                hasattr(self, "_expected_stream_count")
                and self._expected_stream_count
                and self._expected_stream_count in overall_lines_by_stream_count
            ):
                target_stream_count = self._expected_stream_count
                self.logger.debug(f"Using EXPECTED stream count: {target_stream_count}")
            else:
                target_stream_count = max(overall_lines_by_stream_count.keys())
                self.logger.debug(f"Using MAX stream count: {target_stream_count}")

            overall_lines = overall_lines_by_stream_count[target_stream_count]

            # Debug: Show all stream counts found in file for troubleshooting
            all_stream_counts = sorted(overall_lines_by_stream_count.keys())
            self.logger.debug(
                f"Found {len(overall_lines_by_stream_count)} different stream counts in result file: {all_stream_counts}. "
                f"Using stream count: {target_stream_count} with {len(overall_lines)} counters"
            )

            # Extract per-stream FPS from each overall line for the current stream count
            # Line format: "... total=404.56 fps, number-streams=5, per-stream=80.91 fps ..."
            for line in overall_lines:
                # Match specifically: "per-stream= <number> fps"
                match = re.search(r"per-stream=\s*([\d]+\.?\d*)\s+fps", line)
                if match:
                    fps_value = float(match.group(1))
                    fps_measurements.append(fps_value)

        # Calculate average FPS across all per-stream measurements
        if fps_measurements:
            avg_fps = sum(fps_measurements) / len(fps_measurements)
            self.logger.debug(
                f"FPS calculation: {len(fps_measurements)} counter(s), "
                f"measurements={fps_measurements}, average={avg_fps:.2f}"
            )
        else:
            # No valid FPS measurements found
            self.logger.warning("No FPS measurements found in result file")

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
        self,
        resolution=None,
        codec=None,
        bitrate=None,
        ref_stream=None,
        model_name=None,
        max_stream=-1,
        max_binary_search_start=None,
    ):
        """
        Run binary search to find maximum concurrent streams meeting target FPS.

        Args:
            resolution: Video resolution
            codec: Codec name
            bitrate: Target bitrate
            ref_stream: Reference stream count for binary search and CSV comparison
            model_name: Model name for inference
            max_stream: Maximum streams to test (-1 for unlimited)
            max_binary_search_start: Cap for binary search starting point to prevent timeout
                                     on slower platforms. If None, uses ref_stream directly.
                                     This allows keeping high ref_stream for CSV comparison
                                     while starting binary search at a lower, safer value.

        Returns:
            int or str: Maximum concurrent streams, or formatted result string
        """
        quick_search = True
        low = 1
        # Cap the binary search starting point to avoid timeout on slower platforms
        # If max_binary_search_start is set in config, use min(ref_stream, max_binary_search_start)
        # This keeps ref_stream for CSV comparison while starting search lower
        if max_binary_search_start is not None and max_binary_search_start > 0:
            high = min(ref_stream, max_binary_search_start)
            if high < ref_stream:
                self.logger.info(
                    f"[BINARY_SEARCH] Capping starting point from {ref_stream} to {high} "
                    f"(max_binary_search_start={max_binary_search_start})"
                )
        else:
            high = ref_stream
        current_stream = 0
        result = 0
        best_avg_fps = 0.0  # Track best FPS for reporting

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

            self.logger.info(f"[PROGRESS] Executing pipeline for {current_stream} streams...")
            avg_fps, status = self.run_gst_pipeline(gst_cmd, expected_stream_count=current_stream)
            self.logger.info(f"[PROGRESS] Pipeline completed. Average fps is {avg_fps}")

            if status != 0:
                self.logger.error(f"Failed to run the pipeline with {current_stream} streams")

            if self._meet_target_fps(avg_fps):
                if current_stream < ref_stream:
                    low = current_stream + 1
                else:
                    quick_search = False
                result = current_stream
                best_avg_fps = avg_fps  # Store FPS for best result
                self.logger.info(f"[PROGRESS] Target FPS met with {result} streams. Updating telemetry...")
                self.update_telemetry()
                self.logger.info("[PROGRESS] Telemetry updated successfully")
            else:
                if current_stream < ref_stream:
                    high = current_stream - 1
                else:
                    break

            if low > high:
                break

        self.logger.info(f"[PROGRESS] Binary search complete. Optimal stream count: {result}")

        if result == 0:
            self.telemetry_list = [-1] * 8
        else:
            if self.device == "CPU":
                if float(self.telemetry_list[1]) / os.cpu_count() < 25:
                    result = f"{result}(cpu_usage < 25%)"

        if self.benchmark_name == "LPR Benchmark":
            self.logger.info(f"[PROGRESS] Running final LPR verification with {result} streams...")
            gst_cmd = self.gen_gst_command(result, resolution=resolution, model_name=model_name)
            avg_fps, status = self.run_gst_pipeline(gst_cmd)
            self.logger.info(f"{self.benchmark_name} Average fps is {avg_fps}")
            best_avg_fps = avg_fps  # Update best FPS for LPR
            result = f"{avg_fps}@{result}"

        # Expose best_avg_fps as instance variable for reporting
        self.best_avg_fps = best_avg_fps

        self.logger.info(f"[PROGRESS] run_test_round returning result: {result}")
        return result

    def run_test_round_adaptive(
        self,
        resolution,
        codec,
        bitrate,
        ref_stream,
        model_name,
        max_stream,
        previous_model_streams=None,
        safety_factor=0.8,
        min_jump=2,
        min_start=5,
        iteration_timeout=600,
    ):
        """
        Run a test round using FPS-Guided Adaptive Search algorithm.

        This algorithm uses FPS headroom to calculate intelligent jump sizes,
        reducing iterations from O(n) to O(log n) for high-performance GPUs.

        Args:
            resolution: Video resolution string
            codec: Video codec
            bitrate: Video bitrate
            ref_stream: Reference stream count for initial binary search
            model_name: Name of the model being tested
            max_stream: Maximum stream count limit (-1 for unlimited, uses default 100)
            previous_model_streams: Result from previous model (for smart starting point)
            safety_factor: Multiplier for jump calculation (0.0-1.0)
            min_jump: Minimum streams to jump when passing
            min_start: Minimum starting stream count (skip warmup zone)
            iteration_timeout: Maximum time in seconds for a single pipeline iteration

        Returns:
            Result stream count or formatted string (e.g., "36" or "36(cpu_usage < 25%)")
        """
        if not ADAPTIVE_SEARCH_AVAILABLE:
            self.logger.warning("Adaptive search not available, falling back to linear search")
            return self.run_test_round(resolution, codec, bitrate, ref_stream, model_name, max_stream)

        self.logger.info(f"[PROGRESS] Starting adaptive search for model: {model_name}")
        self.logger.info(
            f"[PROGRESS] Parameters: max_stream={max_stream}, ref_stream={ref_stream}, "
            f"safety_factor={safety_factor}, min_start={min_start}"
        )

        if previous_model_streams:
            self.logger.info(f"[PROGRESS] Using previous model result as hint: {previous_model_streams} streams")

        best_avg_fps = 0.0

        def run_pipeline_func(stream_count):
            """
            Wrapper function for adaptive search algorithm.

            Args:
                stream_count: Number of streams to test

            Returns:
                Tuple of (fps, status) where status is 0 for success
            """
            nonlocal best_avg_fps

            self.logger.info(f"[PROGRESS] Executing pipeline for {stream_count} streams...")

            if self.benchmark_name == "LPR Benchmark":
                gst_cmd = self.gen_gst_command(stream_count, resolution=resolution, model_name=model_name)
            else:
                gst_cmd = self.gen_gst_command(stream_count, resolution, codec, bitrate, model_name)

            avg_fps, status = self.run_gst_pipeline(
                gst_cmd, expected_stream_count=stream_count, iteration_timeout=iteration_timeout
            )

            self.logger.info(f"[PROGRESS] Pipeline completed. Average fps is {avg_fps}, status={status}")

            if status != 0:
                self.logger.error(f"Failed to run the pipeline with {stream_count} streams")
                return 0.0, status

            # Track best FPS and update telemetry when target is met
            if self._meet_target_fps(avg_fps):
                best_avg_fps = avg_fps
                self.update_telemetry()

            return avg_fps, status

        # Configure adaptive search
        config = AdaptiveSearchConfig(
            safety_factor=safety_factor,
            min_jump=min_jump,
            min_start=min_start,
            confirmation_threshold=15,
            confirmation_offset=3,
        )

        # Calculate initial stream count
        initial_count = get_initial_stream_count(
            max_streams=max_stream,
            min_start=min_start,
            reference_streams=ref_stream,
            previous_model_streams=previous_model_streams,
        )

        # Run adaptive search
        result, metadata = fps_guided_adaptive_search(
            run_pipeline_func=run_pipeline_func,
            max_streams=max_stream,
            target_fps=self.target_fps,
            initial_count=initial_count,
            config=config,
            logger=self.logger,
        )

        self.logger.info(f"[PROGRESS] Adaptive search complete in {metadata.iterations} iterations")
        self.logger.info(f"[PROGRESS] Search path: {metadata.search_path}")
        self.logger.info(f"[PROGRESS] Time elapsed: {metadata.total_time:.1f}s")

        # Handle result formatting (same as run_test_round)
        if result == 0:
            self.telemetry_list = [-1] * 8
        else:
            if self.device == "CPU":
                if float(self.telemetry_list[1]) / os.cpu_count() < 25:
                    result = f"{result}(cpu_usage < 25%)"

        if self.benchmark_name == "LPR Benchmark":
            self.logger.info(f"[PROGRESS] Running final LPR verification with {result} streams...")
            gst_cmd = self.gen_gst_command(result, resolution=resolution, model_name=model_name)
            avg_fps, status = self.run_gst_pipeline(gst_cmd)
            self.logger.info(f"{self.benchmark_name} Average fps is {avg_fps}")
            best_avg_fps = avg_fps
            result = f"{avg_fps}@{result}"

        # Expose best_avg_fps as instance variable for reporting
        self.best_avg_fps = best_avg_fps
        # Expose search metadata for diagnostics
        self.last_search_metadata = metadata

        self.logger.info(f"[PROGRESS] run_test_round_adaptive returning result: {result}")
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
