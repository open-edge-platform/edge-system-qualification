# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from sysagent.utils.core import check_command_available, run_command

logger = logging.getLogger(__name__)


_INSECURE_HOST_IPS = ("0.0.0.0", "::", "")
_LOOPBACK_HOST_IPS = ("127.0.0.1", "localhost", "::1")
_LOOPBACK_BIND = "127.0.0.1"


def _normalize_port_entry(entry: object) -> Tuple[object, bool, str]:
    """Normalize a single compose ``ports:`` entry to bind to loopback.

    Returns ``(new_entry, modified, reason)``. Container-only entries (no published
    host port) and entries already bound to a non-wildcard, non-loopback IP are
    returned unchanged. Short-form (``"PORT:PORT"``) and wildcard entries
    (``0.0.0.0`` / ``::`` / empty host_ip) are rewritten to ``127.0.0.1``.
    """
    # Long-syntax dict form: {target, published, host_ip, protocol, mode}
    if isinstance(entry, dict):
        if "published" not in entry:
            return entry, False, ""
        host_ip = str(entry.get("host_ip", "")).strip()
        if host_ip in _LOOPBACK_HOST_IPS:
            return entry, False, ""
        if host_ip in _INSECURE_HOST_IPS:
            new = dict(entry)
            new["host_ip"] = _LOOPBACK_BIND
            return new, True, f"host_ip '{host_ip or '<unset>'}' -> {_LOOPBACK_BIND}"
        return entry, False, ""

    s = str(entry).strip()
    if not s:
        return entry, False, ""

    # Preserve trailing protocol suffix (e.g. "/tcp", "/udp").
    proto = ""
    if "/" in s:
        s, proto_part = s.split("/", 1)
        proto = "/" + proto_part

    # IPv6 bracket form: "[IP]:host:container"
    if s.startswith("["):
        end = s.find("]")
        if end == -1:
            return entry, False, ""  # malformed; let docker surface the error
        ip = s[1:end]
        rest = s[end + 1 :]  # noqa: E203
        if ip in _LOOPBACK_HOST_IPS:
            return entry, False, ""
        if ip in _INSECURE_HOST_IPS:
            new_str = f"{_LOOPBACK_BIND}{rest}{proto}"
            return new_str, True, f"'[{ip}]{rest}' -> '{new_str}'"
        return entry, False, ""

    parts = s.split(":")
    if len(parts) == 1:
        # Container-only port, not published to host -> safe.
        return entry, False, ""
    if len(parts) == 2:
        # Short form "host:container" -> Docker defaults to 0.0.0.0.
        new_str = f"{_LOOPBACK_BIND}:{s}{proto}"
        return new_str, True, f"short-form '{entry}' -> '{new_str}'"
    # Three or more parts: explicit "IP:host:container" form.
    ip = parts[0]
    if ip in _LOOPBACK_HOST_IPS:
        return entry, False, ""
    if ip in _INSECURE_HOST_IPS:
        rest = ":".join(parts[1:])
        new_str = f"{_LOOPBACK_BIND}:{rest}{proto}"
        return new_str, True, f"'{ip}:{rest}' -> '{new_str}'"
    return entry, False, ""


def _has_published_target_port(ports: object, target_port: int) -> bool:
    """Return True if a compose ``ports`` block publishes ``target_port``."""
    if not isinstance(ports, list):
        return False

    target_str = str(target_port)
    for entry in ports:
        if isinstance(entry, dict):
            if str(entry.get("target", "")).strip() == target_str and "published" in entry:
                return True
            continue

        s = str(entry).strip()
        if not s:
            continue
        if "/" in s:
            s = s.split("/", 1)[0]

        if s.startswith("["):
            end = s.find("]")
            if end == -1:
                continue
            s = s[end + 1 :].lstrip(":")

        parts = s.split(":")
        if len(parts) >= 2 and parts[-1] == target_str:
            return True

    return False


def _has_exposed_target_port(expose: object, target_port: int) -> bool:
    """Return True if a compose ``expose`` block contains ``target_port``."""
    if not isinstance(expose, list):
        return False

    target_str = str(target_port)
    for entry in expose:
        raw = str(entry).strip()
        if not raw:
            continue
        value = raw.split("/", 1)[0].split(":", 1)[0].strip()
        if value == target_str:
            return True
    return False


def harden_compose_port_bindings(compose_file: str) -> None:
    """Rewrite the local compose file so every published port binds to ``127.0.0.1``.

    The upstream ``edge-ai-libraries`` compose uses Docker short-form port entries
    (e.g. ``"8888:8888"``), which default to ``0.0.0.0`` (all interfaces). We are a
    downstream consumer and must not edit the upstream artifact, but the local
    cached copy we pass to ``docker compose`` is ours to harden. This function
    normalizes the file in place; it is idempotent and a no-op once hardened.
    """
    compose_path = Path(compose_file).expanduser().resolve()
    if not compose_path.is_file():
        return  # Path existence is enforced later in ``validate_paths``.

    try:
        with compose_path.open("r", encoding="utf-8") as f:
            compose_config = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse compose file for port hardening: {compose_file}. Error: {e}")

    services = compose_config.get("services", {})
    if not isinstance(services, dict):
        return

    changed = False
    for service_name, service_config in services.items():
        if not isinstance(service_config, dict):
            continue

        # Upstream v2026.1.0 compose changed backend 8888 from host-published
        # ``ports`` to internal-only ``expose``. Our tests intentionally poll
        # localhost:8888, so preserve backward compatibility by publishing 8888
        # to loopback for backend services when it is only exposed.
        if service_name.startswith("chatqna-core-") and service_name != "chatqna-core-ui":
            ports_obj = service_config.get("ports", [])
            expose_obj = service_config.get("expose", [])
            if _has_exposed_target_port(expose_obj, 8888) and not _has_published_target_port(ports_obj, 8888):
                ports_list = list(ports_obj) if isinstance(ports_obj, list) else []
                ports_list.append(f"{_LOOPBACK_BIND}:8888:8888")
                service_config["ports"] = ports_list
                changed = True
                logger.info(
                    "Added loopback port publish for service '%s' in %s: %s",
                    service_name,
                    compose_path.name,
                    f"{_LOOPBACK_BIND}:8888:8888",
                )

        ports = service_config.get("ports", [])
        if not isinstance(ports, list) or not ports:
            continue
        new_ports: List[object] = []
        for entry in ports:
            new_entry, modified, reason = _normalize_port_entry(entry)
            if modified:
                changed = True
                logger.info(
                    "Hardened port binding for service '%s' in %s: %s",
                    service_name,
                    compose_path.name,
                    reason,
                )
            new_ports.append(new_entry)
        service_config["ports"] = new_ports

    if changed:
        try:
            with compose_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(compose_config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise RuntimeError(f"Failed to write hardened compose file: {compose_file}. Error: {e}")
        logger.info("Compose port bindings hardened to %s in %s", _LOOPBACK_BIND, compose_path)
    else:
        logger.debug("Compose port bindings already restricted to loopback: %s", compose_path)


def validate_compose_port_bindings(compose_file: str) -> None:
    """Strict post-hardening check: fail if any published port is not bound to loopback.

    Run after :func:`harden_compose_port_bindings`. Catches both explicit wildcard
    binds (``0.0.0.0:host:container``) and Docker short-form entries
    (``"host:container"``) that implicitly resolve to ``0.0.0.0``.
    """
    compose_path = Path(compose_file).expanduser().resolve()
    if not compose_path.is_file():
        return  # File validation happens later in ``validate_paths``.

    try:
        with compose_path.open("r", encoding="utf-8") as f:
            compose_config = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse compose file for port validation: {compose_file}. Error: {e}")

    services = compose_config.get("services", {})
    if not isinstance(services, dict):
        return

    for service_name, service_config in services.items():
        if not isinstance(service_config, dict):
            continue
        ports = service_config.get("ports", [])
        if not isinstance(ports, list):
            continue

        for entry in ports:
            # Long-syntax dict form
            if isinstance(entry, dict):
                if "published" not in entry:
                    continue
                host_ip = str(entry.get("host_ip", "")).strip()
                if host_ip in _LOOPBACK_HOST_IPS:
                    continue
                if host_ip in _INSECURE_HOST_IPS:
                    raise RuntimeError(
                        f"Service '{service_name}' publishes port {entry.get('published')} without "
                        f"a loopback host_ip in compose file {compose_path}. Expected host_ip={_LOOPBACK_BIND}."
                    )
                logger.warning(
                    "Service '%s' binds to non-loopback host_ip '%s' (entry: %r). "
                    "Restrict to %s if exposure outside the host is not intended.",
                    service_name,
                    host_ip,
                    entry,
                    _LOOPBACK_BIND,
                )
                continue

            s = str(entry).strip()
            if not s:
                continue
            if "/" in s:
                s = s.split("/", 1)[0]

            # IPv6 bracket form
            if s.startswith("["):
                end = s.find("]")
                if end == -1:
                    continue
                ip = s[1:end]
                if ip in _LOOPBACK_HOST_IPS:
                    continue
                raise RuntimeError(
                    f"Service '{service_name}' publishes port via non-loopback IPv6 bind '{entry}' "
                    f"in {compose_path}. Expected loopback (::1)."
                )

            parts = s.split(":")
            if len(parts) == 1:
                # Container-only, not published -> safe.
                continue
            if len(parts) == 2:
                raise RuntimeError(
                    f"Service '{service_name}' uses Docker short-form port '{entry}' in {compose_path}, "
                    f"which binds to all interfaces (0.0.0.0). Expected '{_LOOPBACK_BIND}:{s}'."
                )
            ip = parts[0]
            if ip in _LOOPBACK_HOST_IPS:
                continue
            if ip in _INSECURE_HOST_IPS:
                raise RuntimeError(
                    f"Service '{service_name}' binds port '{entry}' to all interfaces in {compose_path}. "
                    f"Expected host IP '{_LOOPBACK_BIND}'."
                )
            logger.warning(
                "Service '%s' binds to non-loopback address '%s' in port '%s'. "
                "Restrict to %s if exposure outside the host is not intended.",
                service_name,
                ip,
                entry,
                _LOOPBACK_BIND,
            )

    logger.debug("Port binding security validation passed for %s", compose_file)


def _ensure_render_gid() -> str:
    render_dir = Path("/dev/dri")
    if not render_dir.exists():
        return ""

    render_devices = sorted(render_dir.glob("render*"))
    if not render_devices:
        return ""

    try:
        return str(render_devices[0].stat().st_gid)
    except OSError:
        return ""


def build_runtime_env(configs: Dict[str, object], resolved_paths: Dict[str, str]) -> Dict[str, str]:
    runtime = str(configs.get("backend_runtime", "openvino")).strip().lower()
    compose_profile = str(configs.get("compose_profile", "")).strip()
    if not compose_profile:
        compose_profile = "OLLAMA" if runtime == "ollama" else "OPENVINO"

    env = dict(os.environ)
    env.update(
        {
            "COMPOSE_PROFILES": compose_profile,
            "MODEL_CACHE_PATH": resolved_paths["model_cache_path"],
            "MODEL_CONFIG_PATH": resolved_paths["model_config_path"],
            "APP_BACKEND_URL": str(configs.get("app_backend_url", "/v1/chatqna")),
            "REGISTRY": str(configs.get("registry", "intel/")),
            "BACKEND_TAG": str(configs.get("backend_tag", "core_2026.1.0")),
            "UI_TAG": str(configs.get("ui_tag", "2026.1.0")),
            "USER_GROUP_ID": str(os.getgid()),
            "BACKEND_HOST": str(
                configs.get(
                    "backend_host",
                    configs.get(
                        "backend_service",
                        "chatqna-core-ollama" if runtime == "ollama" else "chatqna-core-ov-cpu",
                    ),
                )
            ),
            "UI_HOST": str(configs.get("ui_host", "chatqna-core-ui")),
        }
    )

    # Forward HuggingFace token if configured by environment.
    token_env_var = str(configs.get("hf_access_token_env_var", "HUGGINGFACEHUB_API_TOKEN")).strip()
    token_value = str(os.environ.get(token_env_var, "")).strip()
    if token_value:
        env["HF_ACCESS_TOKEN"] = token_value

    no_proxy_entries = [item.strip() for item in str(env.get("no_proxy", "")).split(",") if item.strip()]
    for entry in ["127.0.0.1", "127.0.0.1/8", "localhost", "::1", "chatqna-core-ui"]:
        if entry not in no_proxy_entries:
            no_proxy_entries.append(entry)
    backend_host = env.get("BACKEND_HOST", "")
    if backend_host and backend_host not in no_proxy_entries:
        no_proxy_entries.append(backend_host)
    env["no_proxy"] = ",".join(no_proxy_entries)
    env["NO_PROXY"] = env["no_proxy"]

    render_gid = _ensure_render_gid()
    if render_gid:
        env["RENDER_DEVICE_GID"] = render_gid

    return env


def render_nginx_config(resolved_paths: Dict[str, str], env: Dict[str, str]) -> Optional[str]:
    template_path = Path(resolved_paths["nginx_template_path"]).expanduser().resolve()
    if not template_path.is_file():
        logger.warning("Skipping nginx config generation because the template is missing: %s", template_path)
        return None

    output_path = Path(resolved_paths["nginx_output_path"]).expanduser().resolve()
    template_text = template_path.read_text(encoding="utf-8")
    rendered_text = template_text.replace("${BACKEND_HOST}", env.get("BACKEND_HOST", "chatqna-core-ov-cpu"))
    rendered_text = rendered_text.replace("${UI_HOST}", env.get("UI_HOST", "chatqna-core-ui"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_text, encoding="utf-8")
    return str(output_path)


def get_selected_services(configs: Dict[str, object]) -> List[str]:
    backend_service = str(configs.get("backend_service", "chatqna-core-ov-cpu")).strip()
    include_ui = bool(configs.get("include_ui", False))
    services = [backend_service]
    if include_ui:
        compose_profile = str(configs.get("compose_profile", "OPENVINO")).strip().upper()
        if compose_profile == "OLLAMA":
            nginx_service = "nginx-ollama"
        elif compose_profile == "OPENVINO-GPU":
            nginx_service = "nginx-gpu"
        else:
            nginx_service = "nginx-default"
        services.extend(["chatqna-core-ui", nginx_service])
    return services


class ChatQnAComposeManager:
    """Manage the Chat Q&A Core docker compose lifecycle."""

    def __init__(self, compose_file: str, project_name: str, timeout: int = 1800):
        self.compose_file = str(Path(compose_file).expanduser().resolve())
        self.project_name = str(project_name)
        self.timeout = int(timeout)
        self.working_dir = str(Path(self.compose_file).parent)

    def validate_runtime(self) -> None:
        if not check_command_available("docker"):
            raise RuntimeError("docker command is not available")

        result = run_command(["docker", "compose", "version"], timeout=60)
        if not result.success:
            raise RuntimeError("docker compose plugin is not available")

    def validate_paths(self) -> None:
        compose_path = Path(self.compose_file)
        if not compose_path.is_file():
            raise FileNotFoundError(f"Compose file was not found: {compose_path}")
        # Security: We are a downstream consumer of the upstream edge-ai-libraries
        # compose file, which uses Docker short-form port entries that default to
        # 0.0.0.0 (all interfaces). Harden the local cached copy in place by
        # rewriting every published port to bind to 127.0.0.1, then run the strict
        # validator as a post-rewrite sanity check.
        harden_compose_port_bindings(self.compose_file)
        validate_compose_port_bindings(self.compose_file)

    def _compose_cmd(self, sub_cmd: List[str]) -> List[str]:
        return ["docker", "compose", "-f", self.compose_file, "-p", self.project_name, *sub_cmd]

    def pull_images(
        self,
        env: Optional[Dict[str, str]] = None,
        retries: int = 3,
        retry_delay_seconds: int = 30,
    ) -> None:
        """Pull compose images with retry for transient network errors (e.g. Docker Hub TLS timeouts)."""
        last_error: Optional[str] = None
        for attempt in range(1, retries + 1):
            result = run_command(self._compose_cmd(["pull"]), cwd=self.working_dir, env=env, timeout=self.timeout)
            if result.success:
                return
            last_error = result.stderr or result.stdout
            # Transient network conditions worth retrying: TLS timeouts, connection resets, temporary DNS failures
            transient_markers = ("tls handshake timeout", "connection reset", "eof", "i/o timeout", "temporary failure")
            is_transient = any(m in last_error.lower() for m in transient_markers)
            if is_transient and attempt < retries:
                logger.warning(
                    "docker compose pull failed (attempt %d/%d, transient network error); retrying in %ds.\n%s",
                    attempt,
                    retries,
                    retry_delay_seconds,
                    last_error,
                )
                time.sleep(retry_delay_seconds)
            else:
                break
        raise RuntimeError(f"Failed to pull compose images: {last_error}")

    def bring_up(self, services: List[str], env: Optional[Dict[str, str]] = None) -> None:
        sub_cmd = ["up", "-d", *services]
        result = run_command(self._compose_cmd(sub_cmd), cwd=self.working_dir, env=env, timeout=self.timeout)
        if not result.success:
            raise RuntimeError(f"Failed to start compose services: {result.stderr or result.stdout}")

    def bring_down(self, env: Optional[Dict[str, str]] = None) -> None:
        result = run_command(
            self._compose_cmd(["down", "-v"]),
            cwd=self.working_dir,
            env=env,
            timeout=self.timeout,
        )
        if not result.success:
            logger.warning("Compose down reported issues: %s", result.stderr or result.stdout)

    def get_running_services(self, env: Optional[Dict[str, str]] = None) -> List[str]:
        result = run_command(
            self._compose_cmd(["ps", "--services", "--status", "running"]),
            cwd=self.working_dir,
            env=env,
            timeout=120,
        )
        if not result.success:
            raise RuntimeError(f"Failed to get running services: {result.stderr or result.stdout}")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def get_container_id(self, service_name: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Return the first container ID for a compose service, including stopped containers."""
        ps_result = run_command(
            self._compose_cmd(["ps", "-q", "--all", service_name]),
            cwd=self.working_dir,
            env=env,
            timeout=60,
        )
        if not ps_result.success:
            return None
        ids = [line.strip() for line in ps_result.stdout.splitlines() if line.strip()]
        return ids[0] if ids else None

    def get_container_state(self, container_id: str) -> Optional[Tuple[str, str, str]]:
        """Return container state as (status, exit_code, error), or None on failure."""
        inspect_result = run_command(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}}|{{.State.ExitCode}}|{{.State.Error}}",
                container_id,
            ],
            timeout=30,
        )
        if not inspect_result.success:
            return None
        parts = inspect_result.stdout.strip().split("|", 2)
        if len(parts) < 3:
            return None
        return parts[0], parts[1], parts[2]

    @staticmethod
    def _detect_fatal_bootstrap_error(log_text: str) -> Optional[str]:
        # Only match unambiguous startup failures. Generic tracebacks are
        # excluded since benign ones are common during startup; crashed
        # containers are caught via exit-status inspection.
        fatal_markers = (
            "LocalEntryNotFoundError",
            "We couldn't connect to 'https://huggingface.co'",
            "does not appear to have a file named pytorch_model.bin",
            "OSError: We couldn't connect to 'https://huggingface.co'",
            "requests.exceptions.ReadTimeout",
            "RuntimeError: Infer Request is busy",
        )
        for marker in fatal_markers:
            if marker in log_text:
                return marker
        return None

    def wait_for_container_log_ready(
        self,
        service_name: str,
        ready_log_marker: str,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 2700,
        poll_interval_seconds: int = 15,
    ) -> bool:
        """
        Poll docker logs until ready_log_marker appears or timeout expires.

        This is more reliable than HTTP health polling for services where the
        HTTP server only starts AFTER model loading completes (e.g. OpenVINO CPU).

        Args:
            service_name: Compose service name to watch.
            ready_log_marker: Log substring that signals service is ready.
            env: Optional environment for compose commands.
            timeout_seconds: Maximum seconds to wait.
            poll_interval_seconds: Seconds between log checks.

        Returns:
            True if marker found within timeout, False otherwise.
        """
        deadline = time.time() + max(1, int(timeout_seconds))
        poll_interval = max(5, int(poll_interval_seconds))
        elapsed_reported = 0

        while time.time() < deadline:
            container_id = self.get_container_id(service_name, env=env)
            if container_id:
                container_state = self.get_container_state(container_id)
                logs_result = run_command(
                    ["docker", "logs", container_id],
                    timeout=30,
                )
                combined = (logs_result.stdout or "") + (logs_result.stderr or "")

                fatal_marker = self._detect_fatal_bootstrap_error(combined)
                if fatal_marker:
                    raise RuntimeError(f"Service '{service_name}' reported fatal startup error marker: {fatal_marker}")

                if container_state:
                    state_status, exit_code, state_error = container_state
                    if state_status in {"exited", "dead"}:
                        snippet = "\n".join(combined.splitlines()[-30:])
                        raise RuntimeError(
                            f"Service '{service_name}' exited during startup "
                            f"(status={state_status}, exit_code={exit_code}, error={state_error}). "
                            f"Recent logs:\n{snippet}"
                        )

                if ready_log_marker in combined:
                    logger.info(
                        "Service '%s' is ready (log marker found: '%s')",
                        service_name,
                        ready_log_marker,
                    )
                    return True

            elapsed = int(time.time() - (deadline - timeout_seconds))
            if elapsed - elapsed_reported >= 60:
                elapsed_reported = elapsed
                logger.debug(
                    "Waiting for '%s' to be ready... %d/%d seconds elapsed",
                    service_name,
                    elapsed,
                    timeout_seconds,
                )

            time.sleep(poll_interval)

        logger.warning(
            "Service '%s' did not emit ready log marker '%s' within %d seconds.",
            service_name,
            ready_log_marker,
            timeout_seconds,
        )
        return False

    def wait_for_services_ready(
        self,
        services: List[str],
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 5,
    ) -> bool:
        deadline = time.time() + max(1, int(timeout_seconds))
        poll_interval = max(1, int(poll_interval_seconds))

        while time.time() < deadline:
            all_ready = True
            for service_name in services:
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
