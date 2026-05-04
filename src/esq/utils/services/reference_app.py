# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reference app deployment helpers for timeseries ESQ test suites."""

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sysagent.utils.core import run_command

logger = logging.getLogger(__name__)


_VALID_ENV_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _sanitize_env_overrides(values: Dict[str, str]) -> Dict[str, str]:
    """Validate and sanitize env-file key-value overrides."""
    sanitized: Dict[str, str] = {}
    for key, value in values.items():
        if not _VALID_ENV_KEY.fullmatch(key):
            raise ValueError(f"Invalid env key: {key}")
        sanitized[key] = str(value).strip()
    return sanitized


class ReferenceTimeseriesAppManager:
    """Manage lifecycle of the external timeseries reference app deployment."""

    def __init__(self, app_root: str, timeout: int = 1200):
        self.app_root = Path(app_root).resolve()
        self.timeout = timeout
        self.env_path = self.app_root / ".env"
        self._backup_path: Optional[Path] = None

    def validate_paths(self, get_started_rel_path: str = "docs/user-guide/get-started.md") -> None:
        """Validate required reference application files exist."""
        if not self.app_root.is_dir():
            raise FileNotFoundError(f"Reference app path was not found: {self.app_root}")

        compose_file = self.app_root / "docker-compose.yml"
        makefile = self.app_root / "Makefile"
        get_started_doc = self.app_root / get_started_rel_path

        if not compose_file.is_file():
            raise FileNotFoundError(f"Missing docker-compose file: {compose_file}")
        if not makefile.is_file():
            raise FileNotFoundError(f"Missing Makefile: {makefile}")
        if not get_started_doc.is_file():
            raise FileNotFoundError(f"Missing get-started guide: {get_started_doc}")
        if not self.env_path.is_file():
            raise FileNotFoundError(f"Missing env file: {self.env_path}")

    def pull_images(self, sample_app: str = "wind-turbine-anomaly-detection") -> None:
        """Pull container images used by docker compose deployment."""
        env = {"SAMPLE_APP": sample_app}
        result = run_command(
            ["docker", "compose", "pull", "--ignore-buildable"],
            cwd=str(self.app_root),
            env=env,
            timeout=self.timeout,
        )
        if result.success:
            return

        fallback = run_command(
            ["docker", "compose", "pull"],
            cwd=str(self.app_root),
            env=env,
            timeout=self.timeout,
        )
        if not fallback.success:
            raise RuntimeError(f"Failed to pull compose images: {fallback.stderr or result.stderr}")

    def apply_env_overrides(self, env_overrides: Dict[str, str]) -> None:
        """Patch reference app .env with test-safe runtime values."""
        sanitized = _sanitize_env_overrides(env_overrides)

        original_content = self.env_path.read_text(encoding="utf-8")
        if self._backup_path is None:
            self._backup_path = self.env_path.with_suffix(".env.esq.backup")
            shutil.copy2(self.env_path, self._backup_path)

        updates_applied: Dict[str, bool] = {key: False for key in sanitized}
        updated_lines: List[str] = []
        for line in original_content.splitlines(keepends=True):
            replaced = False
            for key, value in sanitized.items():
                prefix = f"{key}="
                if line.startswith(prefix):
                    updated_lines.append(f"{prefix}{value}\n")
                    updates_applied[key] = True
                    replaced = True
                    break
            if not replaced:
                updated_lines.append(line)

        for key, value in sanitized.items():
            if not updates_applied[key]:
                updated_lines.append(f"{key}={value}\n")

        self.env_path.write_text("".join(updated_lines), encoding="utf-8")

    def restore_env_file(self) -> None:
        """Restore original .env after test execution."""
        if self._backup_path and self._backup_path.is_file():
            shutil.copy2(self._backup_path, self.env_path)
            self._backup_path.unlink(missing_ok=True)
            self._backup_path = None

    def run_make_target(
        self,
        target: str,
        args: Optional[List[str]] = None,
        sample_app: str = "wind-turbine-anomaly-detection",
    ) -> Tuple[bool, str, str]:
        """Execute a make target in the reference app directory."""
        command = ["make", target]
        if args:
            command.extend(args)
        result = run_command(command, cwd=str(self.app_root), env={"SAMPLE_APP": sample_app}, timeout=self.timeout)
        return result.success, result.stdout, result.stderr

    def bring_down(self, sample_app: str = "wind-turbine-anomaly-detection") -> None:
        """Stop existing deployment resources."""
        ok, stdout, stderr = self.run_make_target("down", sample_app=sample_app)
        if not ok:
            logger.warning("Reference app bring-down reported issues: %s", stderr or stdout)

    def bring_up(
        self,
        ingestion_mode: str,
        sample_app: str,
        num_streams: int,
        number_of_data_points_per_stream: Optional[int],
        ingestion_interval: str,
    ) -> None:
        """Deploy reference app using documented make targets."""
        mode = ingestion_mode.strip().lower()
        if mode not in {"opcua", "mqtt"}:
            raise ValueError(f"Unsupported ingestion mode: {ingestion_mode}")

        target = "up_opcua_ingestion" if mode == "opcua" else "up_mqtt_ingestion"
        args = [
            f"app={sample_app}",
            f"num_of_streams={num_streams}",
            f"ingestion_interval={ingestion_interval}",
        ]
        if number_of_data_points_per_stream is not None:
            args.append(f"number_of_data_points_per_stream={number_of_data_points_per_stream}")

        ok, stdout, stderr = self.run_make_target(target, args=args, sample_app=sample_app)
        if not ok:
            raise RuntimeError(f"Failed to deploy reference app with target '{target}': {stderr or stdout}")

    def status(self, sample_app: str = "wind-turbine-anomaly-detection") -> str:
        """Run make status and return output for test metadata."""
        ok, stdout, stderr = self.run_make_target("status", sample_app=sample_app)
        if not ok:
            raise RuntimeError(f"Deployment status check failed: {stderr or stdout}")
        return stdout

    def get_running_services(self, sample_app: str = "wind-turbine-anomaly-detection") -> List[str]:
        """Collect list of running compose services for metrics and reporting."""
        result = run_command(
            ["docker", "compose", "ps", "--services", "--status", "running"],
            cwd=str(self.app_root),
            env={"SAMPLE_APP": sample_app},
            timeout=120,
        )
        if not result.success:
            raise RuntimeError(f"Failed to get running services: {result.stderr}")

        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
