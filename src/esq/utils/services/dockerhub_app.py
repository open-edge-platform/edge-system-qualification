# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Docker Hub based deployment helpers for timeseries ESQ suites."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from sysagent.utils.core import run_command

logger = logging.getLogger(__name__)


class DockerHubTimeseriesAppManager:
    """Manage lifecycle of a docker-compose stack that uses prebuilt images only."""

    def __init__(self, compose_file: str, project_name: str = "esq-timeseries", timeout: int = 1200):
        self.compose_file = str(Path(compose_file).resolve())
        self.project_name = project_name
        self.timeout = timeout
        self.working_dir = str(Path(self.compose_file).parent)

    def validate_paths(self) -> None:
        """Validate compose path exists."""
        compose_path = Path(self.compose_file)
        if not compose_path.is_file():
            raise FileNotFoundError(f"Compose file was not found: {compose_path}")

    def _compose_cmd(self, sub_cmd: List[str]) -> List[str]:
        return [
            "docker",
            "compose",
            "-f",
            self.compose_file,
            "-p",
            self.project_name,
            *sub_cmd,
        ]

    @staticmethod
    def _sanitize_error_text(text: str, max_chars: int = 600) -> str:
        """Return a compact error summary suitable for reports."""
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        if not lines:
            return "Unknown error"

        priority_markers = (
            "error",
            "failed",
            "timeout",
            "tls",
            "denied",
            "unauthorized",
        )
        selected = None
        for line in reversed(lines):
            lowered = line.lower()
            if any(marker in lowered for marker in priority_markers):
                selected = line
                break
        if selected is None:
            selected = lines[-1]

        return selected[:max_chars]

    @staticmethod
    def _is_transient_pull_error(error_text: str) -> bool:
        """Detect transient network/registry failures that are worth retrying."""
        lowered = str(error_text).lower()
        transient_markers = (
            "tls handshake timeout",
            "i/o timeout",
            "timeout",
            "context deadline exceeded",
            "temporary failure",
            "connection reset",
            "connection refused",
            "unexpected eof",
            "no route to host",
            "service unavailable",
            "too many requests",
        )
        return any(marker in lowered for marker in transient_markers)

    def pull_images(
        self,
        env: Optional[Dict[str, str]] = None,
        retries: int = 3,
        retry_delay_seconds: int = 8,
        retry_backoff: float = 2.0,
        quiet: bool = True,
    ) -> None:
        """Pull all images defined in compose file with retry on transient network failures."""
        attempts = max(1, int(retries))
        delay_seconds = max(1, int(retry_delay_seconds))
        backoff = retry_backoff if retry_backoff and retry_backoff > 1.0 else 1.0

        sub_cmd = ["pull"]
        if quiet:
            sub_cmd.append("--quiet")

        for attempt in range(1, attempts + 1):
            result = run_command(
                self._compose_cmd(sub_cmd),
                cwd=self.working_dir,
                env=env,
                timeout=self.timeout,
            )
            if result.success:
                return

            raw_error = result.stderr or result.stdout or ""
            compact_error = self._sanitize_error_text(raw_error)
            is_transient = self._is_transient_pull_error(raw_error)

            if is_transient and attempt < attempts:
                logger.warning(
                    "Compose pull failed on attempt %s/%s: %s. Retrying in %ss",
                    attempt,
                    attempts,
                    compact_error,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
                delay_seconds = int(delay_seconds * backoff)
                continue

            if is_transient:
                raise RuntimeError(
                    f"Failed to pull compose images after {attempts} attempts: {compact_error}"
                )

            raise RuntimeError(f"Failed to pull compose images: {compact_error}")

    def bring_up(
        self,
        services: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        scale: Optional[Dict[str, int]] = None,
    ) -> None:
        """Bring up compose services in detached mode."""
        sub_cmd = ["up", "-d"]
        if scale:
            for service_name, replicas in scale.items():
                if replicas and replicas > 0:
                    sub_cmd.extend(["--scale", f"{service_name}={int(replicas)}"])
        if services:
            sub_cmd.extend(services)

        result = run_command(
            self._compose_cmd(sub_cmd),
            cwd=self.working_dir,
            env=env,
            timeout=self.timeout,
        )
        if not result.success:
            raise RuntimeError(f"Failed to start compose services: {result.stderr or result.stdout}")

    def bring_down(
        self,
        remove_volumes: bool = True,
        remove_orphans: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """Bring down compose services."""
        sub_cmd = ["down"]
        if remove_volumes:
            sub_cmd.append("-v")
        if remove_orphans:
            sub_cmd.append("--remove-orphans")

        result = run_command(
            self._compose_cmd(sub_cmd),
            cwd=self.working_dir,
            env=env,
            timeout=self.timeout,
        )
        if not result.success:
            logger.warning("Compose down reported issues: %s", result.stderr or result.stdout)

    def status(self, env: Optional[Dict[str, str]] = None) -> str:
        """Return compose ps status output."""
        result = run_command(
            self._compose_cmd(["ps"]),
            cwd=self.working_dir,
            env=env,
            timeout=120,
        )
        if not result.success:
            raise RuntimeError(f"Deployment status check failed: {result.stderr or result.stdout}")
        return result.stdout

    def get_running_services(self, env: Optional[Dict[str, str]] = None) -> List[str]:
        """Return list of currently running service names."""
        result = run_command(
            self._compose_cmd(["ps", "--services", "--status", "running"]),
            cwd=self.working_dir,
            env=env,
            timeout=120,
        )
        if not result.success:
            raise RuntimeError(f"Failed to get running services: {result.stderr or result.stdout}")

        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def wait_for_services_ready(
        self,
        services: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 5,
    ) -> bool:
        """Wait for selected services to be running and healthy when health checks exist."""
        wait_services = services or self.get_running_services(env=env)
        if not wait_services:
            return False

        deadline = time.time() + max(1, int(timeout_seconds))
        poll_interval = max(1, int(poll_interval_seconds))

        while time.time() < deadline:
            all_ready = True
            for service_name in wait_services:
                # Resolve all container IDs for this compose service, including scaled replicas.
                ps_result = run_command(
                    self._compose_cmd(["ps", "-q", service_name]),
                    cwd=self.working_dir,
                    env=env,
                    timeout=60,
                )
                if not ps_result.success:
                    all_ready = False
                    break

                container_ids = [line.strip() for line in ps_result.stdout.splitlines() if line.strip()]
                if not container_ids:
                    all_ready = False
                    break

                for container_id in container_ids:
                    inspect_result = run_command(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                            container_id,
                        ],
                        timeout=60,
                    )
                    if not inspect_result.success:
                        all_ready = False
                        break

                    status_parts = inspect_result.stdout.strip().split("|", 1)
                    state_status = status_parts[0] if status_parts else ""
                    health_status = status_parts[1] if len(status_parts) > 1 else "none"

                    if state_status != "running":
                        all_ready = False
                        break
                    if health_status not in {"healthy", "none"}:
                        all_ready = False
                        break

                if not all_ready:
                    break

            if all_ready:
                return True

            time.sleep(poll_interval)

        return False
