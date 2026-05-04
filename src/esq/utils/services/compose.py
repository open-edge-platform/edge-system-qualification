# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Docker compose helpers for multi-service ESQ test suites."""

import concurrent.futures
import logging
from pathlib import Path
from typing import Dict, List, Optional

from esq.utils.services.models import ComposeProjectConfig, ServiceStartResult
from sysagent.utils.core import check_command_available, run_command

logger = logging.getLogger(__name__)


class ComposeServiceOrchestrator:
    """Thin orchestration layer for docker compose based service stacks."""

    def __init__(self, config: ComposeProjectConfig, timeout: int = 300):
        self.config = config
        self.timeout = timeout

    def validate_runtime(self) -> None:
        """Validate docker and compose availability before test setup."""
        if not check_command_available("docker"):
            raise RuntimeError("docker command is not available")

        result = run_command(["docker", "compose", "version"], timeout=30)
        if not result.success:
            raise RuntimeError("docker compose plugin is not available")

    def validate_compose_file(self) -> None:
        """Ensure configured docker compose file exists."""
        compose_path = Path(self.config.compose_file)
        if not compose_path.is_file():
            raise FileNotFoundError(f"Compose file was not found: {compose_path}")

    def pull_images(self, images: Optional[List[str]] = None) -> None:
        """Pull required container images when explicitly configured."""
        if not images:
            return

        for image in images:
            result = run_command(["docker", "pull", image], timeout=self.timeout)
            if not result.success:
                raise RuntimeError(f"Failed to pull image '{image}': {result.stderr}")

    def up_services(self, services: List[str], env: Optional[Dict[str, str]] = None) -> ServiceStartResult:
        """Start one or more services with docker compose up -d."""
        command = [
            "docker",
            "compose",
            "-f",
            self.config.compose_file,
            "-p",
            self.config.project_name,
            "up",
            "-d",
            *services,
        ]
        result = run_command(command, cwd=self.config.working_dir, env=env, timeout=self.timeout)
        return ServiceStartResult(
            ok=result.success,
            services=services,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def up_services_parallel(
        self,
        service_groups: Dict[str, List[str]],
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, ServiceStartResult]:
        """Start independent service groups in parallel."""
        results: Dict[str, ServiceStartResult] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(service_groups), 1)) as executor:
            futures = {
                executor.submit(self.up_services, services, env): group_name
                for group_name, services in service_groups.items()
            }
            for future in concurrent.futures.as_completed(futures):
                group_name = futures[future]
                try:
                    results[group_name] = future.result()
                except Exception as exc:
                    results[group_name] = ServiceStartResult(
                        ok=False,
                        services=service_groups[group_name],
                        stderr=str(exc),
                    )

        return results

    def down(self, remove_volumes: bool = False) -> ServiceStartResult:
        """Stop and clean up compose resources for the configured project."""
        command = [
            "docker",
            "compose",
            "-f",
            self.config.compose_file,
            "-p",
            self.config.project_name,
            "down",
        ]
        if remove_volumes:
            command.append("-v")

        result = run_command(command, cwd=self.config.working_dir, timeout=self.timeout)
        return ServiceStartResult(
            ok=result.success,
            services=[],
            stdout=result.stdout,
            stderr=result.stderr,
        )
