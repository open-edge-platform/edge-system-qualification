# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Docker client utilities for container management.

This module provides utilities for Docker container operations including
image pulling, container creation, log streaming, and cleanup.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
from datetime import datetime

# Conditional imports for optional dependencies
try:
    import docker
except ImportError:
    docker = None

try:
    import pytest
except ImportError:
    pytest = None

try:
    import allure
except ImportError:
    # Fallback allure for minimal installations
    class allure:
        @staticmethod
        def attach(*args, **kwargs):
            pass


try:
    from tqdm import tqdm
except ImportError:
    # Fallback progress bar for minimal installations
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable or []
            self.total = total or len(self.iterable) if hasattr(self.iterable, "__len__") else None
            self.desc = desc or "Progress"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def set_description(self, desc):
            self.desc = desc

        def close(self):
            pass


import time
from typing import Mapping

logger = logging.getLogger(__name__)


class DockerClient:
    def __init__(self, timeout: int = 300):
        """
        Args:
            timeout: Default timeout for Docker API calls in seconds (default: 300)
        """
        if docker is None:
            raise ImportError("Docker package not available. Install with: pip install docker")

        self.client = docker.from_env(timeout=timeout)
        self._log_threads = {}  # container_name -> (thread, stop_event, logs)
        self._client_timeout = timeout

        try:
            logger.info("Verifying Docker client connection")
            self.client.ping()
            logger.debug(f"Docker client initialized with {timeout}s timeout")
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise RuntimeError(f"Docker is not available: {e}")

    def pull_image(self, image, log_every=10):
        """
        Pull a Docker image only if it does not already exist locally.
        """
        try:
            self.client.images.get(image)
            logger.info(f"Docker image '{image}' already exists locally. Skipping pull.")
            return
        except docker.errors.ImageNotFound:
            logger.info(f"Docker image '{image}' not found locally. Pulling...")
        except Exception as e:
            logger.warning(f"Error checking local image '{image}': {e}. Attempting to pull anyway.")

        logger.info(f"Pulling Docker image: {image}")
        stream = self.client.api.pull(image, stream=True, decode=True)
        download_count = 0
        extract_count = 0
        for line in stream:
            status = line.get("status", "")
            progress = line.get("progress", "")
            if status.lower().startswith("downloading"):
                download_count += 1
                if download_count % log_every == 0:
                    logger.info(f"Downloading... {progress}")
            elif status.lower().startswith("extracting"):
                extract_count += 1
                if extract_count % log_every == 0:
                    logger.info(f"Extracting... {progress}")
            elif status.lower().startswith("pull complete"):
                logger.info("Pull complete for layer")

        logger.info(f"Successfully pulled Docker image: {image}")

    def build_image(
        self,
        path: str = None,
        tag: str = None,
        nocache: bool = False,
        dockerfile: str = None,
        buildargs: dict = None,
        extract_packages: bool = False,
    ):
        """
        Build a Docker image with streaming logs and Allure attachments.

        Args:
            path: Build context path
            tag: Docker image tag
            nocache: Disable Docker build cache
            dockerfile: Dockerfile name (default: Dockerfile)
            buildargs: Build arguments dictionary
            extract_packages: If True, extract and attach package list as Allure attachment

        Returns:
            dict: {"docker_image": tag, "image_id": image.id, "build_log_text": ..., "image_obj": image}
        """
        # If image already built in this CLI session, reuse it
        built_tags_env = os.environ.get("CORE_BUILT_DOCKER_TAGS", "")
        built_tags = set(built_tags_env.split(";")) if built_tags_env else set()
        if tag in built_tags:
            logger.info(f"Reusing cached Docker image for tag: {tag} in CLI session")
            nocache = False

        try:
            logger.info("Verifying Docker client connection")
            self.client.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise RuntimeError(f"Docker is not available: {e}")

        logger.debug(f"Building Docker image from: {path}")
        try:
            logger.info("Starting Docker image build")
            logger.info(f"Building image: {tag}")
            logger.debug(f"Build context: {path}")

            api_client = self.client.api
            step_count = 0
            build_log_text = ""
            logger.info("Initiating Docker build")

            build_stream = api_client.build(
                path=path,
                dockerfile=dockerfile,
                tag=tag,
                nocache=nocache,
                rm=True,
                forcerm=True,
                decode=True,
                buildargs=buildargs,
            )

            image_id = None
            error_keywords = ["E: ", "ERROR:", "returned a non-zero code", "failed", "failure", "fatal"]
            info_keywords = ["Successfully built", "Successfully tagged"]
            ignore_error_patterns = [
                "Current default time zone:",
                "Failed to resolve user 'systemd-network'",
                "Failed to resolve group 'systemd-journal'",
                'Failed to open connection to "system" message bus',
                "Key is stored in legacy",
            ]

            for log_entry in build_stream:
                if "stream" in log_entry:
                    log_line = log_entry["stream"].strip()
                    if log_line:
                        build_log_text += log_entry["stream"]
                        if log_line.startswith("Step "):
                            step_count += 1
                        if "Successfully built" in log_line:
                            image_id = log_line.split()[-1]
                        log_line_lower = log_line.lower()
                        is_ignore_pattern = any(pattern.lower() in log_line_lower for pattern in ignore_error_patterns)
                        is_running_in_container = (
                            log_line.startswith("---> Running in ")
                            and len(log_line.split()) == 4
                            and len(log_line.split()[3]) >= 10
                        )
                        if is_ignore_pattern:
                            logger.debug(f"Docker Build: {log_line}")
                        elif is_running_in_container:
                            logger.info(f"Docker Build: {log_line}")
                        elif any(keyword.lower() in log_line_lower for keyword in error_keywords):
                            logger.warning(f"Docker Build: {log_line}")
                        elif any(keyword in log_line for keyword in info_keywords):
                            logger.info(f"Docker Build: {log_line}")
                        else:
                            logger.debug(f"Docker Build: {log_line}")
                elif "status" in log_entry:
                    status_msg = log_entry.get("status", "")
                    if status_msg:
                        status_lower = status_msg.lower()
                        layer_id = log_entry.get("id", "")
                        is_ignore_pattern = any(pattern.lower() in status_lower for pattern in ignore_error_patterns)
                        if is_ignore_pattern:
                            log_msg = (
                                f"Docker Build Status: {layer_id}: {status_msg}"
                                if layer_id
                                else f"Docker Build Status: {status_msg}"
                            )
                            logger.debug(log_msg)
                        elif any(keyword.lower() in status_lower for keyword in error_keywords):
                            log_msg = (
                                f"Docker Build Status Error: {layer_id}: {status_msg}"
                                if layer_id
                                else f"Docker Build Status Error: {status_msg}"
                            )
                            logger.warning(log_msg)
                        elif any(keyword in status_msg for keyword in ["Pulling", "Downloaded", "Extracting"]):
                            log_msg = (
                                f"Docker Build: {layer_id}: {status_msg}" if layer_id else f"Docker Build: {status_msg}"
                            )
                            logger.info(log_msg)
                        else:
                            log_msg = (
                                f"Docker Build Status: {layer_id}: {status_msg}"
                                if layer_id
                                else f"Docker Build Status: {status_msg}"
                            )
                            logger.debug(log_msg)
                        if layer_id:
                            build_log_text += f"{layer_id}: {status_msg}\n"
                        else:
                            build_log_text += f"{status_msg}\n"
                elif "error" in log_entry:
                    error_msg = log_entry["error"]
                    logger.error(f"Docker Build Error: {error_msg}")
                    build_log_text += f"ERROR: {error_msg}\n"
                    raise docker.errors.BuildError(error_msg, build_log=[log_entry])
                elif "errorDetail" in log_entry:
                    error_detail = log_entry["errorDetail"]
                    error_msg = error_detail.get("message", str(error_detail))
                    logger.error(f"Docker Build Error Detail: {error_msg}")
                    build_log_text += f"ERROR DETAIL: {error_msg}\n"
                    raise docker.errors.BuildError(error_msg, build_log=[log_entry])
                elif log_entry and not any(key in log_entry for key in ["stream", "status", "error", "errorDetail"]):
                    logger.debug(f"Docker Build Unknown Entry: {log_entry}")
                    build_log_text += f"UNKNOWN: {log_entry}\n"

            logger.info(f"Docker image build completed successfully with {step_count} steps")

            # Get the built image object
            if image_id:
                try:
                    image = self.client.images.get(image_id)
                    logger.info(f"Successfully built Docker image: {tag}")
                    logger.debug(f"Image ID: {image.id}")
                except docker.errors.ImageNotFound:
                    image = self.client.images.get(tag)
                    logger.info(f"Successfully built Docker image: {tag}")
                    logger.debug(f"Image ID: {image.id}")
            else:
                image = self.client.images.get(tag)
                logger.info(f"Successfully built Docker image: {tag}")
                logger.debug(f"Image ID: {image.id}")

            logger.info(f"Build completed with {step_count} steps processed")

            allure.attach(
                build_log_text, name=f"Docker Build Logs - {tag}", attachment_type=allure.attachment_type.TEXT
            )

        except docker.errors.BuildError as e:
            logger.error(f"Docker build failed with detailed error: {e}")
            error_logs = build_log_text if "build_log_text" in locals() else str(e)
            if error_logs and error_logs != str(e):
                logger.error(f"Build log context:\n{error_logs[-2000:]}")
            allure.attach(error_logs, name="Docker Build Error Logs", attachment_type=allure.attachment_type.TEXT)
            raise
        except Exception as e:
            logger.error(f"Docker build failed with unexpected error: {e}")
            error_logs = build_log_text if "build_log_text" in locals() else str(e)
            allure.attach(error_logs, name="Docker Build Error Logs", attachment_type=allure.attachment_type.TEXT)
            raise docker.errors.BuildError(str(e), build_log=[])
        finally:
            # Clean up dangling images
            try:
                dangling_images = self.client.images.list(filters={"dangling": True})
                for img in dangling_images:
                    try:
                        self.client.images.remove(img.id, force=True)
                        logger.info(f"Removed dangling image: {img.id}")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not remove dangling image {img.id}: {cleanup_error}")
            except Exception as e:
                logger.warning(f"Failed to list or remove dangling images: {e}")

        # Test if image was created successfully
        try:
            image_info = self.client.images.get(tag)
            logger.debug(f"Image details: Size={image_info.attrs.get('Size', 'Unknown')}")
        except docker.errors.ImageNotFound:
            logger.error(f"Image {tag} was not found after build")
            raise

        # Construct dockerfile path for base image extraction
        dockerfile_path = None
        if path:
            dockerfile_name = dockerfile if dockerfile else "Dockerfile"
            dockerfile_path = os.path.join(path, dockerfile_name)

        # Extract and attach package list if requested, otherwise just extract base image info
        if extract_packages:
            self._extract_package_list(tag, dockerfile_path)
        else:
            # Always extract base image information even when package list is disabled
            logger.info(f"Extracting base image information from Docker image: {tag}")
            base_image_info = self._extract_base_image_info(tag, dockerfile_path)
            self._create_base_image_attachment(tag, base_image_info)

        result = {"docker_image": tag, "image_id": image.id, "build_log_text": build_log_text, "image_obj": image}
        # Cache the result for this CLI session
        built_tags.add(tag)
        os.environ["CORE_BUILT_DOCKER_TAGS"] = ";".join(built_tags)
        return result

    def _extract_package_list(self, tag: str, dockerfile_path: str = None) -> None:
        """
        Extract package list from a Docker image and attach it to Allure report.

        Args:
            tag: Docker image tag
            dockerfile_path: Path to Dockerfile used for build (optional)
        """
        try:
            logger.info(f"Extracting package list from Docker image: {tag}")

            # Use run_container's built-in file extraction to avoid stdout logging
            # Create results directory first, then run package extraction
            package_cmd = (
                "mkdir -p /mnt/results && "
                "dpkg -l | grep '^ii' | awk '{print $2\" \"$3}' | sort > /mnt/results/packages.txt && "
                "echo 'Package extraction completed'"
            )

            container_result = self.run_container(
                image=tag,
                entrypoint=["sh", "-c", package_cmd],
                name=f"package-list-{hash(tag) % 10000}",
                user="root",  # Run as root for dpkg access and directory creation
                remove=True,
                timeout=30,
                mode="batch",
                attach_logs=False,  # Don't attach container logs
                result_file="packages.txt",  # Extract this file from container
                container_result_file_dir="/mnt/results",  # Container directory containing the file
            )

            if container_result.get("container_info", {}).get("exit_code") == 0:
                # Get the package list from the extracted file (not from stdout)
                package_list_raw = container_result.get("result_text", "")
                if package_list_raw:
                    # Process package list and add line numbers
                    package_lines = package_list_raw.strip().split("\n")
                    package_count = len([line for line in package_lines if line.strip()])

                    # Create numbered package list
                    numbered_packages = []
                    line_num = 1
                    for line in package_lines:
                        if line.strip():
                            numbered_packages.append(f"{line_num:4d}. {line}")
                            line_num += 1

                    # Get base image hash information with Dockerfile parsing
                    base_image_info = self._extract_base_image_info(tag, dockerfile_path)

                    # Create comprehensive attachment content
                    attachment_content = "Docker Image Information Report\n"
                    attachment_content += f"Image: {tag}\n"
                    attachment_content += "Generated by: DockerClient.build_image()\n"
                    attachment_content += f"Generated at: {datetime.now().isoformat()}\n"
                    attachment_content += "=" * 60 + "\n\n"

                    # Add base image information using shared formatter
                    if base_image_info:
                        attachment_content += self._format_base_image_section(base_image_info)

                    # Add package list with count and line numbers
                    attachment_content += f"INSTALLED PACKAGES ({package_count} total)\n"
                    attachment_content += "-" * 30 + "\n"
                    attachment_content += "Line  Package Name                     Version\n"
                    attachment_content += "-" * 60 + "\n"
                    attachment_content += "\n".join(numbered_packages)

                    # Attach to Allure report
                    allure.attach(
                        attachment_content,
                        name=f"Docker Image Info & Packages - {tag}",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                    logger.info(f"✓ Package list attached for Docker image: {tag} ({package_count} packages)")
                else:
                    logger.warning(f"No package list output for Docker image: {tag}")

        except Exception as e:
            logger.warning(f"Error extracting package list for {tag}: {str(e)}")

    def _parse_dockerfile_base_image(self, dockerfile_path: str) -> str:
        """
        Parse Dockerfile to extract the base image from FROM instruction.

        Args:
            dockerfile_path: Path to the Dockerfile

        Returns:
            Base image name from FROM instruction, or "Unknown" if not found
        """
        try:
            with open(dockerfile_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Look for FROM instruction (case insensitive)
            # Handle multi-stage builds by getting the first FROM
            import re

            # Pattern to match FROM instruction, handling comments and multi-line
            # This matches: FROM <image> or FROM <image>:<tag> or FROM <image>@<digest>
            pattern = r"^\s*FROM\s+([^\s#]+(?::[^\s#]+|@[^\s#]+)?)"

            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):  # Skip comments and empty lines
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        base_image = match.group(1)
                        logger.debug(f"Extracted base image from Dockerfile: {base_image}")
                        return base_image

            logger.warning(f"No FROM instruction found in Dockerfile: {dockerfile_path}")
            return "Unknown"

        except Exception as e:
            logger.warning(f"Error parsing Dockerfile {dockerfile_path}: {str(e)}")
            return "Unknown"

    def _extract_repository_digest(self, image_name: str) -> str:
        """
        Extract repository digest (RepoDigests) from Docker image.

        Args:
            image_name: Name/tag of the Docker image

        Returns:
            Repository digest string or "Unknown" if not available
        """
        try:
            image = self.client.images.get(image_name)

            # Get RepoDigests from image attributes
            if hasattr(image, "attrs") and image.attrs:
                repo_digests = image.attrs.get("RepoDigests", [])
                if repo_digests:
                    # Return the first repo digest
                    repo_digest = repo_digests[0]
                    logger.debug(f"Repository digest for {image_name}: {repo_digest}")
                    return repo_digest

            logger.debug(f"No repository digest found for {image_name}")
            return "Unknown"

        except Exception as e:
            logger.warning(f"Failed to extract repository digest for {image_name}: {e}")
            return "Unknown"

    def _extract_base_image_info(self, tag: str, dockerfile_path: str = None) -> dict:
        """
        Extract base image information from Dockerfile and repository digests.

        Args:
            tag: Docker image tag
            dockerfile_path: Path to Dockerfile used for build (optional)

        Returns:
            Dictionary containing base image information
        """
        try:
            # Get image object for built image info
            image = self.client.images.get(tag)

            # Extract base image name from Dockerfile
            base_image = "Unknown"
            if dockerfile_path and os.path.exists(dockerfile_path):
                base_image = self._parse_dockerfile_base_image(dockerfile_path)
                logger.debug(f"Base image from Dockerfile: {base_image}")

            # Get repository digests
            built_image_digest = self._extract_repository_digest(tag)
            base_image_digest = "Unknown"

            if base_image != "Unknown":
                base_image_digest = self._extract_repository_digest(base_image)

            return {
                "base_image": base_image,
                "image_id": image.id,
                "digest": built_image_digest,
                "base_image_digest": base_image_digest,
            }

        except Exception as e:
            logger.warning(f"Error extracting base image info for {tag}: {str(e)}")
            return {"base_image": "Unknown", "image_id": "Unknown", "digest": "Unknown", "base_image_digest": "Unknown"}

    def _format_base_image_section(self, base_image_info: dict) -> str:
        """
        Format base image information as a text section.

        Args:
            base_image_info: Dictionary containing base image information

        Returns:
            Formatted string with base image information
        """
        section = "BASE IMAGE INFORMATION\n"
        section += "-" * 30 + "\n"
        section += f"Base Image (FROM): {base_image_info.get('base_image', 'Unknown')}\n"
        section += f"Built Image ID: {base_image_info.get('image_id', 'Unknown')}\n"
        section += f"Built Image Digest: {base_image_info.get('digest', 'Unknown')}\n"

        # Add base image repository digest if available
        base_image_digest = base_image_info.get("base_image_digest")
        if base_image_digest and base_image_digest != "Unknown":
            section += f"FROM Image Digest: {base_image_digest}\n"
        section += "\n"

        return section

    def _create_base_image_attachment(self, tag: str, base_image_info: dict) -> None:
        """
        Create Allure attachment for base image information only.

        Args:
            tag: Docker image tag
            base_image_info: Dictionary containing base image information
        """
        try:
            attachment_content = "Docker Image Base Information\n"
            attachment_content += f"Image: {tag}\n"
            attachment_content += "Generated by: DockerClient.build_image()\n"
            attachment_content += f"Generated at: {datetime.now().isoformat()}\n"
            attachment_content += "=" * 50 + "\n\n"

            # Add base image information using shared formatter
            attachment_content += self._format_base_image_section(base_image_info)

            # Attach to Allure report
            allure.attach(
                attachment_content,
                name=f"Docker Base Image Info - {tag}",
                attachment_type=allure.attachment_type.TEXT,
            )
            logger.info(f"✓ Base image info attached for Docker image: {tag}")

        except Exception as e:
            logger.warning(f"Error creating base image attachment for {tag}: {str(e)}")

    def create_container(self, image, command=None, volumes=None, environment=None, detach=True, name=None, **kwargs):
        """
        Create a Docker container.
        """
        logger.info(f"Creating Docker container from image: {image}")
        if name:
            logger.info(f"Container name: {name}")
        if command:
            logger.info(f"Container command: {command}")
        if volumes:
            logger.info(f"Container volumes: {volumes}")
        if environment:
            logger.info(f"Container environment: {environment}")

        container = self.client.containers.create(
            image, command=command, volumes=volumes, environment=environment, detach=detach, name=name, **kwargs
        )
        logger.info(f"Created container: {container.id}")
        return container

    def run_container(
        self,
        name: str = None,
        image: str = None,
        entrypoint: str | list[str] | None = None,
        environment: str | list[str] | None = None,
        volumes: dict = None,
        devices: list = None,
        labels: dict = None,
        working_dir: str = None,
        network_mode: str = None,
        detach: bool = True,
        remove: bool = True,
        user: str = None,
        group_add: list = None,
        cpuset_cpus: str = None,
        cpuset_mems: str = None,
        shm_size: str = None,
        cap_add: list[str] | None = None,
        command: str | list[str] | None = None,
        timeout: int = None,
        result_file: str = None,
        container_result_file_dir: str = None,
        ports: Mapping[str, int | list[int] | tuple[str, int] | None] | None = None,
        mode: str = "batch",  # "batch" or "server"
        attach_logs: bool = True,  # Whether to attach container logs to Allure report
    ):
        """
        Run a Docker container with specified configuration.

        Args:
            name: Container name
            image: Docker image to use
            entrypoint: Container entrypoint
            environment: Environment variables (dict, list, or string)
            volumes: Volume mounts dictionary
            devices: Device mappings list
            labels: Container labels dictionary
            working_dir: Working directory inside container
            network_mode: Network mode
            detach: Run container in detached mode
            remove: Remove container after completion
            user: User to run container as
            group_add: Additional groups to add
            cpuset_cpus: CPUs in which to allow execution
            cpuset_mems: Memory nodes in which to allow execution
            shm_size: Size of /dev/shm
            cap_add: List of Linux capabilities to add (e.g., ['PERFMON', 'SYS_ADMIN'])
            command: Command to run in container
            timeout: Timeout for container execution in seconds
            result_file: Name of result file to extract
            container_result_file_dir: Directory in container where result file is located
            ports: Port mappings
            mode: Execution mode - "batch" (wait for completion) or "server" (return immediately)
            attach_logs: Whether to attach container logs to Allure report

        Returns:
            For batch mode: dict with container_logs_text, result_text, result_json, container_info
            For server mode: container object
        """
        container = None
        temp_results_dir = None
        container_logs_text = ""
        container_info = {}
        try:
            # Handle result file volume mount
            if result_file and container_result_file_dir:
                temp_results_dir = tempfile.mkdtemp(prefix="pytest_docker_")
                if not volumes:
                    volumes = {}
                volumes[temp_results_dir] = {"bind": container_result_file_dir, "mode": "rw"}

                try:
                    logger.debug(f"Setting secure permissions (0o770) for temp directory: {temp_results_dir}")
                    os.chmod(temp_results_dir, 0o770)
                except OSError as e:
                    raise RuntimeError(f"Cannot set secure permissions on temporary directory: {e}")

            # Check if container with the same name is already created (might not be running)
            if name:
                try:
                    existing = self.client.containers.list(all=True, filters={"name": f"^{name}$"})
                    for ex in existing:
                        logger.info(f"Found existing container '{name}' (status: {ex.status}), removing it first.")
                        try:
                            if ex.status == "running":
                                ex.stop(timeout=5)
                            ex.remove(force=True)
                            logger.info(f"Removed existing container '{name}'.")
                        except Exception as e:
                            logger.warning(f"Failed to remove existing container '{name}': {e}")
                except Exception as e:
                    logger.warning(f"Error checking/removing existing container '{name}': {e}")

            # Ensure LOG_LEVEL is set in environment variables (default to DEBUG if not present)
            if environment is None:
                environment = {"LOG_LEVEL": "DEBUG"}
            elif isinstance(environment, dict):
                # If environment is a dictionary, add LOG_LEVEL only if not already set
                environment = environment.copy()  # Don't modify the original
                if "LOG_LEVEL" not in environment:
                    environment["LOG_LEVEL"] = "DEBUG"
            elif isinstance(environment, list):
                # If environment is a list of KEY=VALUE strings
                environment = environment.copy()  # Don't modify the original
                # Check if LOG_LEVEL already exists, if not add it
                log_level_found = False
                for env_var in environment:
                    if isinstance(env_var, str) and env_var.startswith("LOG_LEVEL="):
                        log_level_found = True
                        break
                if not log_level_found:
                    environment.append("LOG_LEVEL=DEBUG")
            elif isinstance(environment, str):
                # If environment is a single string, check if it's LOG_LEVEL, otherwise add LOG_LEVEL
                if environment.startswith("LOG_LEVEL="):
                    # LOG_LEVEL is already set, keep as single string
                    pass
                else:
                    # Convert to list and add LOG_LEVEL
                    environment = [environment, "LOG_LEVEL=DEBUG"]
            else:
                logger.warning(f"Unexpected environment type: {type(environment)}, adding LOG_LEVEL as dict")
                environment = {"LOG_LEVEL": "DEBUG"}

            logger.debug(f"Environment variables for container (LOG_LEVEL defaults to DEBUG if not set): {environment}")

            logger.info(f"Running Docker container: {name} with image {image}")
            container = self.client.containers.run(
                name=name,
                image=image,
                entrypoint=entrypoint,
                environment=environment,
                volumes=volumes,
                devices=devices,
                labels=labels,
                working_dir=working_dir,
                network_mode=network_mode,
                detach=detach,
                remove=remove,
                user=user,
                group_add=group_add,
                cpuset_cpus=cpuset_cpus,
                cpuset_mems=cpuset_mems,
                shm_size=shm_size,
                cap_add=cap_add,
                command=command,
                ports=ports,
            )
            logger.debug(f"Container {container.name} with mode {mode} started successfully")

            # For batch mode: stream logs and wait for container to finish
            if mode == "batch":
                timeout_triggered = threading.Event()

                def timeout_killer():
                    if timeout:
                        logger.debug(f"Starting container {container.name} timeout killer thread for {timeout} seconds")
                        time.sleep(timeout)
                        try:
                            try:
                                refreshed = self.client.containers.get(container.id)
                            except docker.errors.NotFound:
                                return

                            logger.info(f"Checking container {container.name} status after {timeout} secs")
                            refreshed.reload()
                            if refreshed.status == "running":
                                logger.warning(f"Timeout reached ({timeout}s), stopping container {container.name}")
                                refreshed.stop(timeout=5)
                                timeout_triggered.set()
                        except Exception as e:
                            container_name = container.name if container else "unknown"
                            logger.warning(f"Failed to stop container {container_name} after timeout: {e}")

                killer_thread = threading.Thread(target=timeout_killer, daemon=True)
                killer_thread.start()

                log_stream = container.logs(stream=True, follow=True)
                for log_chunk in log_stream:
                    try:
                        log_line = log_chunk.decode("utf-8").rstrip()
                        if log_line:
                            logger.info(f"Container: {log_line}")
                            container_logs_text += log_chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        container_logs_text += f"[binary data: {len(log_chunk)} bytes]\n"

                result = container.wait(timeout=10)
                killer_thread.join(0)
                if timeout_triggered.is_set():
                    container_name = container.name if container and hasattr(container, "name") else "unknown"
                    logger.error(f"Container {container_name} stopped due to timeout ({timeout}s)")
                    if container_logs_text:
                        allure.attach(
                            container_logs_text,
                            name="Container Execution Logs",
                            attachment_type=allure.attachment_type.TEXT,
                        )
                    pytest.fail(f"Container {container_name} execution stopped due to timeout ({timeout}s)")

                logger.debug(f"Waiting for container {container.name} to finish processing...")
                time.sleep(5)

            elif mode == "server":
                # Return container object immediately, let caller manage timeout/logs
                return container

            exit_code = result.get("StatusCode", -1)
            logger.info(f"Container completed with exit code: {exit_code}")

            container_info = {
                "id": container.id[:12],
                "name": container.name,
                "exit_code": exit_code,
            }

            # attach container log with attachment that show container name and exit code
            # Only attach if enabled and not suppressed by environment flag for consolidation
            if attach_logs and not os.environ.get("CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS"):
                allure.attach(
                    container_logs_text,
                    name=f"Container Logs - {name} (Exit Code: {exit_code})",
                    attachment_type=allure.attachment_type.TEXT,
                )
            elif attach_logs:
                # Add to global collector for later consolidation (only if attach_logs is True)
                try:
                    from sysagent.utils.plugins.pytest_execution import add_container_log_for_consolidation

                    add_container_log_for_consolidation(name, container_logs_text, exit_code)
                except ImportError:
                    pass  # Collector not available, skip

            if exit_code != 0:
                logger.error(f"Container {name} exited with non-zero code: {exit_code}")
                pytest.fail(f"Container {name} failed with exit code: {exit_code}")

            result_text = None
            result_json = None

            if result_file and container_result_file_dir and temp_results_dir:
                results_file_path = os.path.join(temp_results_dir, result_file)
                if os.path.exists(results_file_path):
                    with open(results_file_path, "r") as f:
                        result_text = f.read()
                    try:
                        result_json = json.loads(result_text)
                        logger.debug(f"Successfully extracted results from volume mount: {results_file_path}")
                    except Exception:
                        result_json = None
                else:
                    logger.warning(f"Results file not found in volume mount: {results_file_path}")
                    # Fallback: try to extract results from inside the container
                    try:
                        logger.info(f"Attempting to extract results from container at /app/{result_file}")
                        # Copy the file from container to temporary location
                        temp_file = os.path.join(temp_results_dir, f"fallback_{result_file}")
                        self.copy_from_container(name, f"/app/{result_file}", temp_file)
                        if os.path.exists(temp_file):
                            with open(temp_file, "r") as f:
                                result_text = f.read()
                            try:
                                result_json = json.loads(result_text)
                                logger.info("Successfully extracted results from container fallback location")
                            except Exception as e:
                                logger.warning(f"Failed to parse JSON from fallback results: {e}")
                                result_json = None
                        else:
                            logger.warning(f"Fallback results file not found at /app/{result_file}")
                    except Exception as e:
                        logger.warning(f"Failed to extract results from container fallback: {e}")

            return {
                "container_logs_text": container_logs_text,
                "result_text": result_text,
                "result_json": result_json,
                "container_info": container_info,
            }
        finally:
            if temp_results_dir:
                try:
                    shutil.rmtree(temp_results_dir)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary directory {temp_results_dir}: {cleanup_error}")

    def get_container_logs(self, container_name):
        """
        Get accumulated logs for a container.
        """
        if container_name in self._log_threads:
            return self._log_threads[container_name][2]
        else:
            try:
                container = self.client.containers.get(container_name)
                logs = container.logs().decode("utf-8").split("\n")
                return [log.strip() for log in logs if log.strip()]
            except Exception as e:
                logger.error(f"Failed to get logs for {container_name}: {e}")
                return []

    def wait_for_container(self, container_name, timeout=None):
        """
        Wait for a container to finish execution.
        """
        try:
            container = self.client.containers.get(container_name)
            logger.debug(f"Waiting for container {container_name} to finish...")

            result = container.wait(timeout=timeout)
            exit_code = result["StatusCode"]

            logger.debug(f"Container {container_name} finished with exit code: {exit_code}")
            return exit_code

        except Exception as e:
            logger.error(f"Error waiting for container {container_name}: {e}")
            raise

    def stop_container(self, container_name, timeout=10):
        """
        Stop a running container.
        """
        if not self.container_exists(container_name):
            logger.debug(f"Container {container_name} does not exist, skipping stop operation")
            return

        try:
            container = self.client.containers.get(container_name)
            if container.status == "running":
                logger.info(f"Stopping container: {container_name}")
                container.stop(timeout=timeout)
                logger.info(f"Stopped container: {container_name}")
            else:
                logger.debug(f"Container {container_name} is not running (status: {container.status}), skipping stop")
        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} no longer exists, skipping stop")
        except Exception as e:
            logger.error(f"Failed to stop container {container_name}: {e}")
            raise

    def remove_container(self, container_name, force=False):
        """
        Remove a container.
        """
        if not self.container_exists(container_name):
            logger.debug(f"Container {container_name} does not exist, skipping remove operation")
            return

        try:
            container = self.client.containers.get(container_name)
            logger.info(f"Removing container: {container_name}")
            container.remove(force=force)
            logger.info(f"Removed container: {container_name}")
        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} no longer exists, skipping remove")
        except Exception as e:
            logger.error(f"Failed to remove container {container_name}: {e}")
            raise

    def cleanup_container(self, container_name, timeout=10):
        """
        Stop log streaming, stop and remove a container in any state.
        This method handles containers that may be created, running, stopped, or failed.
        """
        logger.debug(f"Cleaning up container: {container_name}")

        # Always stop log streaming first, regardless of container existence
        self.stop_log_streaming(container_name)

        # Check if container exists before attempting cleanup
        if not self.container_exists(container_name):
            return

        try:
            # Get container to check its status
            container = self.client.containers.get(container_name)
            container_status = container.status
            logger.debug(f"Container {container_name} status: {container_status}")

            # Force remove the container regardless of its state
            # This handles containers in created, running, stopped, exited, or failed states
            logger.debug(f"Forcefully removing container: {container_name} (status: {container_status})")
            container.remove(force=True)
            logger.debug(f"Successfully removed container: {container_name}")

        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} not found during cleanup (already removed)")
        except Exception as e:
            logger.warning(f"Error during container cleanup: {e}")
            # Try alternative cleanup approach
            try:
                logger.debug(f"Attempting alternative cleanup for container: {container_name}")
                # Use low-level API for forced removal
                self.client.api.remove_container(container_name, force=True)
                logger.debug(f"Successfully removed container using API: {container_name}")
            except Exception as api_error:
                logger.warning(f"Alternative cleanup also failed for {container_name}: {api_error}")

        logger.info(f"Cleanup completed for container: {container_name}")

    def cleanup_all_containers(self, timeout=10):
        """
        Cleanup all containers managed by this client.
        """
        container_names = list(self._log_threads.keys())
        logger.debug(f"Cleaning up {len(container_names)} containers")

        for container_name in container_names:
            try:
                self.cleanup_container(container_name, timeout=timeout)
            except Exception as e:
                logger.error(f"Error cleaning up container {container_name}: {e}")

    def cleanup_containers_by_name_pattern(self, name_pattern: str, timeout=10):
        """
        Cleanup containers that match a specific name pattern.
        Handles containers in any state (created, running, stopped, exited, failed).

        Args:
            name_pattern (str): Pattern to match in container names
            timeout (int): Timeout for stopping containers (unused, kept for compatibility)
        """
        try:
            logger.info(f"Cleaning up containers matching pattern: '{name_pattern}'")
            all_containers = self.client.containers.list(all=True)
            matching_containers = []

            for container in all_containers:
                if container.name and name_pattern in container.name:
                    matching_containers.append(container)

            if not matching_containers:
                logger.debug(f"No containers found matching pattern: '{name_pattern}'")
                return

            logger.info(f"Found {len(matching_containers)} containers matching pattern '{name_pattern}'")

            for container in matching_containers:
                try:
                    container_status = container.status
                    logger.debug(
                        f"Removing container: {container.name} ({container.id[:12]}, status: {container_status})"
                    )

                    # Force remove regardless of state - this handles created, running, stopped, exited states
                    container.remove(force=True)
                    logger.debug(f"Successfully removed container: {container.name}")

                except docker.errors.NotFound:
                    logger.debug(f"Container {container.name} not found (already removed)")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove container {container.name}: {cleanup_error}")
                    # Try alternative approach with API
                    try:
                        logger.debug(f"Attempting API removal for container: {container.name}")
                        self.client.api.remove_container(container.id, force=True)
                        logger.debug(f"Successfully removed container via API: {container.name}")
                    except Exception as api_error:
                        logger.warning(f"API removal also failed for {container.name}: {api_error}")

        except Exception as e:
            logger.warning(f"Failed to cleanup containers by pattern '{name_pattern}': {e}")

    def copy_to_container(self, container_name, src_path, dst_path):
        """
        Copy a file or directory from host to container.
        """
        try:
            container = self.client.containers.get(container_name)

            # Create a temporary tar file
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_file:
                tar_path = tmp_file.name

            # Create tar archive
            shutil.make_archive(
                tar_path.replace(".tar", ""), "tar", os.path.dirname(src_path), os.path.basename(src_path)
            )

            # Copy to container
            with open(tar_path, "rb") as tar_file:
                container.put_archive(dst_path, tar_file.read())

            # Cleanup
            os.unlink(tar_path)

            logger.info(f"Copied {src_path} to {container_name}:{dst_path}")

        except Exception as e:
            logger.error(f"Failed to copy {src_path} to {container_name}:{dst_path}: {e}")
            raise

    def copy_from_container(self, container_name, src_path, dst_path):
        """
        Copy a file or directory from container to host.
        """
        try:
            container = self.client.containers.get(container_name)

            # Get archive from container
            archive_stream, _ = container.get_archive(src_path)

            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_file:
                for chunk in archive_stream:
                    tmp_file.write(chunk)
                tar_path = tmp_file.name

            # Extract archive
            shutil.unpack_archive(tar_path, dst_path)

            # Cleanup
            os.unlink(tar_path)

            logger.info(f"Copied {container_name}:{src_path} to {dst_path}")

        except Exception as e:
            logger.error(f"Failed to copy {container_name}:{src_path} to {dst_path}: {e}")
            raise

    def execute_command(self, container_name, command, workdir=None):
        """
        Execute a command in a running container.
        """
        try:
            container = self.client.containers.get(container_name)
            logger.info(f"Executing command in {container_name}: {command}")

            exec_result = container.exec_run(command, workdir=workdir)
            exit_code = exec_result.exit_code
            output = exec_result.output.decode("utf-8")

            logger.info(f"Command exit code: {exit_code}")
            if output:
                logger.info(f"Command output:\n{output}")

            return exit_code, output

        except Exception as e:
            logger.error(f"Failed to execute command in {container_name}: {e}")
            raise

    def get_container_status(self, container_name):
        """
        Get the status of a container.
        """
        try:
            container = self.client.containers.get(container_name)
            return container.status
        except Exception as e:
            logger.error(f"Failed to get status for container {container_name}: {e}")
            return "unknown"

    def list_containers(self, all_containers=True):
        """
        List all containers.
        """
        try:
            containers = self.client.containers.list(all=all_containers)
            return [(c.name, c.status, c.id) for c in containers]
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")
            return []

    def prune_containers(self):
        """
        Remove all stopped containers.
        """
        try:
            logger.info("Pruning stopped containers...")
            result = self.client.containers.prune()
            logger.info(f"Pruned containers: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to prune containers: {e}")
            raise

    def container_exists(self, name_or_id):
        """
        Check if a container with the given name or id exists.
        """
        try:
            self.client.containers.get(name_or_id)
            return True
        except docker.errors.NotFound:
            return False
        except Exception as e:
            logger.warning(f"Error checking if container {name_or_id} exists: {e}")
            return False

    def is_container_running(self, name_or_id):
        """
        Check if a container with the given name or id exists and is running.
        """
        try:
            container = self.client.containers.get(name_or_id)
            return container.status == "running"
        except Exception:
            return False

    def stream_container_logs(self, container, stop_event=None):
        """
        Stream logs from a running container and return as a string.
        Can be run in a background thread or directly.
        """
        container_logs_text = ""
        for log_chunk in container.logs(stream=True, follow=True):
            if stop_event and stop_event.is_set():
                break
            try:
                log_line = log_chunk.decode("utf-8").rstrip()
                logger.info(f"[{container.name}] {log_line}")
                container_logs_text += log_line + "\n"
            except Exception as e:
                logger.warning(f"Log decode error: {e}")
        return container_logs_text

    def start_log_streaming(self, container, container_name):
        """
        Start a background thread to stream logs for a container.
        Also attaches logs to Allure when the container exits, if not already attached.
        """
        stop_event = threading.Event()
        logs = {"text": "", "attached": False}

        def log_collector():
            for log_chunk in container.logs(stream=True, follow=True):
                if stop_event.is_set():
                    break
                try:
                    log_line = log_chunk.decode("utf-8").rstrip()
                    logger.info(f"[{container.name}] {log_line}")
                    logs["text"] += log_line + "\n"
                except Exception as e:
                    logger.warning(f"Log decode error: {e}")
            if logs["text"] and not logs["attached"]:
                # Get container exit code if container has exited
                exit_code = "Running"
                try:
                    container.reload()  # Refresh status from Docker daemon
                    if container.status != "running":
                        exit_code = container.attrs.get("State", {}).get("ExitCode", "Unknown")
                        logger.debug(f"Container {container_name} exit code: {exit_code}")
                except Exception as e:
                    logger.warning(f"Failed to get exit code for container {container_name}: {e}")

                # Only attach if not suppressed by environment flag
                if not os.environ.get("CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS"):
                    allure.attach(
                        logs["text"],
                        name=f"Container Logs - {container_name} (Exit Code: {exit_code})",
                        attachment_type=allure.attachment_type.TEXT,
                    )
                    logs["attached"] = True
                else:
                    # Add to global collector for later consolidation
                    try:
                        from sysagent.utils.plugins.pytest_execution import add_container_log_for_consolidation

                        add_container_log_for_consolidation(container_name, logs["text"], exit_code)
                        logs["attached"] = True  # Mark as handled to avoid duplicates
                    except ImportError:
                        pass  # Collector not available, skip

        thread = threading.Thread(target=log_collector, daemon=True)
        thread.start()
        self._log_threads[container_name] = (thread, stop_event, logs)

    def stop_log_streaming(self, container_name, attach_to_allure=True):
        """
        Stop the log streaming thread and optionally attach logs to Allure.
        """
        logger.debug(f"Stopping log streaming for container: {container_name}")
        if container_name not in self._log_threads:
            logger.warning(f"No active log streaming for container: {container_name}")
            return ""

        thread, stop_event, logs = self._log_threads[container_name]
        stop_event.set()
        thread.join(timeout=5)
        logger.debug(f"Saving logs for container: {container_name}")

        # Only attach if not already attached and not suppressed by environment flag
        if (
            attach_to_allure
            and logs["text"]
            and not logs.get("attached", False)
            and not os.environ.get("CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS")
        ):
            # Get container exit code if possible
            exit_code = "Unknown"
            try:
                # Try to find the container by name
                containers = self.client.containers.list(all=True, filters={"name": container_name})
                if containers:
                    container = containers[0]
                    container.reload()
                    if container.status != "running":
                        exit_code = container.attrs.get("State", {}).get("ExitCode", "Unknown")
                        logger.debug(f"Container {container_name} exit code: {exit_code}")
            except Exception as e:
                logger.warning(f"Failed to get exit code for container {container_name}: {e}")

            allure.attach(
                logs["text"],
                name=f"Container Logs - {container_name} (Exit Code: {exit_code})",
                attachment_type=allure.attachment_type.TEXT,
            )
            logs["attached"] = True
        elif (
            attach_to_allure
            and logs["text"]
            and not logs.get("attached", False)
            and os.environ.get("CORE_SUPPRESS_CONTAINER_LOG_ATTACHMENTS")
        ):
            # Get container exit code if possible
            exit_code = "Unknown"
            try:
                # Try to find the container by name
                containers = self.client.containers.list(all=True, filters={"name": container_name})
                if containers:
                    container = containers[0]
                    container.reload()
                    if container.status != "running":
                        exit_code = container.attrs.get("State", {}).get("ExitCode", "Unknown")
                        logger.debug(f"Container {container_name} exit code: {exit_code}")
            except Exception as e:
                logger.warning(f"Failed to get exit code for container {container_name}: {e}")

            # Add to global collector for later consolidation
            try:
                from sysagent.utils.plugins.pytest_execution import add_container_log_for_consolidation

                add_container_log_for_consolidation(container_name, logs["text"], exit_code)
                logs["attached"] = True  # Mark as handled to avoid duplicates
            except ImportError:
                pass  # Collector not available, skip

        del self._log_threads[container_name]
        return logs["text"]

    def get_container_logs_text(self, container_name):
        """
        Get the collected logs for a container.
        """
        if container_name in self._log_threads:
            return self._log_threads[container_name][2]["text"]
        return ""

    def extract_docker_image_digest(self, image_name: str) -> str:
        """
        Extract SHA256 digest from Docker image for supply chain security tracking.

        Args:
            image_name: Name/tag of the Docker image

        Returns:
            SHA256 digest string, empty string if extraction fails
        """
        try:
            # Get image object
            image = self.client.images.get(image_name)

            # Extract SHA256 digest from image ID (remove 'sha256:' prefix if present)
            image_id = image.id
            if image_id.startswith("sha256:"):
                digest = image_id[7:]  # Remove 'sha256:' prefix
            else:
                digest = image_id

            logger.info(f"Extracted digest for image {image_name}: {digest[:12]}...")
            return digest

        except Exception as e:
            logger.warning(f"Failed to extract digest for image {image_name}: {e}")
            return ""

    def __del__(self):
        """
        Cleanup on object destruction.
        """
        try:
            self.cleanup_all_containers()
        except Exception:
            pass
