# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Functional (runtime) sanity checks for the prerun check system.

Unlike presence-only dependency checks ("is X installed?"), these checks
verify that an already-detected component is *actually* functional. The
canonical motivating case: OpenVINO reports a GPU/NPU as "available", but
compiling/running a model on it hangs or fails due to an incompatible BIOS
version, missing firmware, or a bad driver for the detected silicon. Running
a full test profile against such a device just burns time until it times
out. These checks catch that in seconds, before any profile/test executes.

Each check is intentionally quick and runs in an isolated subprocess where a
hang is possible, so a misbehaving device/daemon can be killed via timeout
instead of hanging the whole CLI.
"""

import logging
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request

from sysagent.utils.core.process import run_command
from sysagent.utils.spinner import set_spinner_suffix

logger = logging.getLogger(__name__)

DEFAULT_OV_SMOKE_TIMEOUT = 30.0
DEFAULT_NETWORK_TIMEOUT = 10.0
# Environment override for the per-host network check timeout (seconds), for testing slow or
# lossy networks without editing config.
NETWORK_TIMEOUT_ENV = "CORE_NETWORK_TIMEOUT_SECONDS"
DEFAULT_NETWORK_PORT = 443
DEFAULT_NETWORK_RETRY_DELAY = 1.0
DEFAULT_DOCKER_SMOKE_TIMEOUT = 60.0
DEFAULT_DOCKER_SMOKE_IMAGE = "curlimages/curl:latest"

_HOSTNAME_PATTERN = re.compile(r"[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?")


_OK_MARK = "✓"
_FAIL_MARK = "✗"


_warned_invalid_timeouts: set[str] = set()


def get_network_timeout(default: float = DEFAULT_NETWORK_TIMEOUT) -> float:
    """Resolve the per-host network check timeout: env override, else ``default``.

    ``CORE_NETWORK_TIMEOUT_SECONDS`` wins when set to a positive number; an invalid value
    (non-numeric, <= 0) is logged once as a warning and ignored.
    """
    raw = os.environ.get(NETWORK_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            timeout = float(raw)
            if 0 < timeout < float("inf"):
                return timeout
        except ValueError:
            pass
        if raw not in _warned_invalid_timeouts:
            _warned_invalid_timeouts.add(raw)
            logger.warning("Ignoring invalid %s=%r (expected a positive number of seconds)", NETWORK_TIMEOUT_ENV, raw)
    return default


def _mark(ok: bool, text: str) -> str:
    """Prefix a check-result line with a pass/fail glyph for quick visual scanning."""
    return f"{_OK_MARK if ok else _FAIL_MARK} {text}"


def split_mark(line: str) -> tuple[bool | None, str]:
    """Inverse of ``_mark``: ``(ok, text)``, with ``ok`` None if the line has no glyph."""
    for mark, ok in ((_OK_MARK, True), (_FAIL_MARK, False)):
        if line.startswith(f"{mark} "):
            return ok, line[len(mark) + 1 :]
    return None, line


def check_openvino_device(device: str, model_path: str, timeout: float = DEFAULT_OV_SMOKE_TIMEOUT) -> tuple[bool, str]:
    """Run ``benchmark_app`` on a single OpenVINO device in an isolated subprocess.

    Returns:
        (ok, message)
    """
    from sysagent.utils.system.ov_benchmark import build_benchmark_command, parse_throughput_fps

    result = run_command(build_benchmark_command(model_path, device), timeout=timeout)

    if result.timed_out:
        return False, _mark(
            False, f"{device}: timed out after {timeout:.0f}s (possible BIOS/firmware/driver incompatibility)"
        )
    if not result.success:
        output = (result.stderr or result.stdout or "unknown error").strip()
        last_line = output.splitlines()[-1] if output else "unknown error"
        return False, _mark(False, f"{device}: {last_line}")

    fps = parse_throughput_fps(result.stdout)
    if fps is None:
        return True, _mark(True, f"{device}: OK")
    return True, _mark(True, f"{device}: OK ({fps:.0f} FPS)")


def check_openvino_devices(
    devices: list[str] | None = None,
    timeout: float = DEFAULT_OV_SMOKE_TIMEOUT,
    progress_cb=None,
) -> tuple[bool, list[str]]:
    """Run smoke tests across OpenVINO devices (auto-detected if ``devices`` is empty).

    Returns:
        (all_ok, messages) - all_ok is True when every checked device passed
        (or when there was nothing to check).
    """
    from sysagent.utils.system.ov_benchmark import create_benchmark_model
    from sysagent.utils.system.ov_helper import get_available_devices, is_openvino_available

    if not is_openvino_available():
        return True, [_mark(True, "OpenVINO not installed - device smoke test skipped")]

    target_devices = devices or get_available_devices()
    if not target_devices:
        return True, [_mark(True, "No OpenVINO devices detected - device smoke test skipped")]

    all_ok = True
    messages = []
    with tempfile.TemporaryDirectory(prefix="ov_benchmark_") as model_dir:
        model_path = create_benchmark_model(model_dir)
        for device in target_devices:
            ok, message = check_openvino_device(device, model_path, timeout=timeout)
            messages.append(message)
            if not ok:
                all_ok = False
            if progress_cb:
                progress_cb(device, ok, message)

    return all_ok, messages


def _connect_host(host: str, port: int, timeout: float, results: dict) -> None:
    """Attempt a single reachability check, storing the outcome in ``results[host]``."""
    if not _HOSTNAME_PATTERN.fullmatch(host):
        results[host] = (False, _mark(False, f"{host}:{port}: invalid hostname"))
        return

    url = f"https://{host}" if port == 443 else f"https://{host}:{port}"

    # An HTTPS request through the default opener (not a raw socket connect) honors
    # HTTP_PROXY/HTTPS_PROXY/NO_PROXY env vars the same way curl does. A raw socket
    # connect bypasses any configured proxy entirely and reports hosts as
    # unreachable on networks where only proxied traffic is allowed out.
    request = urllib.request.Request(url, method="HEAD")
    opener = urllib.request.build_opener()
    try:
        with opener.open(request, timeout=timeout):
            pass
        results[host] = (True, _mark(True, f"{host}:{port}: reachable"))
    except urllib.error.HTTPError:
        # Any HTTP response (even an error status) means the connection -
        # including through a proxy, if configured - succeeded.
        results[host] = (True, _mark(True, f"{host}:{port}: reachable"))
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        results[host] = (False, _mark(False, f"{host}:{port}: unreachable ({reason})"))


def resolve_network_hosts(
    hosts: list[str],
    host_alternatives: dict,
    flagged_hosts: dict | None = None,
    active_flags: set[str] | None = None,
) -> list[str]:
    """Expand ``hosts`` with the host(s) for whichever alternative source is preferred.

    ``host_alternatives`` maps a group name (e.g. "model_download") to a dict
    of source name -> hosts (e.g. {"huggingface": [...], "modelscope": [...]}).
    Only the preferred source's host(s) are added - resolved the same way
    (``PREFER_<SOURCE>`` env vars) a CLI's downloaders pick a source - so a
    source the user isn't configured to use (e.g. huggingface.co when
    PREFER_MODELSCOPE=1 is set) is never reported as a false failure.
    """
    resolved = list(hosts)
    # Hosts gated behind a profile requirement flag (e.g. Docker registry only when
    # docker_required) are skipped when the flags are known and don't include theirs.
    for flag, flag_hosts in (flagged_hosts or {}).items():
        if active_flags is None or flag in active_flags:
            resolved.extend(h for h in flag_hosts if h not in resolved)

    if not host_alternatives:
        return resolved

    from sysagent.utils.infrastructure.network import get_preferred_download_source

    for group in host_alternatives.values():
        if not group:
            continue
        sources = list(group.keys())
        preferred = get_preferred_download_source(default=sources[0], valid_sources=sources)
        resolved.extend(group.get(preferred, []))
    return resolved


def check_network_connectivity(
    hosts: list[str],
    timeout: float = DEFAULT_NETWORK_TIMEOUT,
    port: int = DEFAULT_NETWORK_PORT,
    attempts: int | None = None,
    retry_delay: float = DEFAULT_NETWORK_RETRY_DELAY,
) -> tuple[bool, list[str]]:
    """Check outbound network connectivity via an HTTP(S) request to each host.

    Networks with intermittent connectivity (proxies/firewalls that sporadically
    drop or reset connections) can fail a host on one attempt and pass on the next,
    so each failing host is retried up to ``attempts`` times in total, waiting a fixed
    ``retry_delay`` between rounds. Hosts that already passed are not probed again.
    The current attempt is shown on the status spinner when more than one is allowed.

    Returns:
        (all_reachable, messages) - passes only if every host is reachable. The
        host list is already narrowed to what the run needs (core hosts, plus
        flag-gated and preferred-source hosts), so an unreachable host would
        otherwise fail later during its download. A host that only passed after a
        retry is reported with the attempt count, to make intermittency visible.
    """
    if not hosts:
        return True, []

    # Same attempt count as the downloaders (env override, else the shared default).
    from sysagent.utils.infrastructure.retry import get_retry_attempts

    attempts = get_retry_attempts(attempts)
    hosts = list(dict.fromkeys(hosts))
    outcomes: dict[str, tuple[bool, str]] = {}
    passed_on: dict[str, int] = {}
    pending = hosts
    for attempt in range(1, max(attempts, 1) + 1):
        if attempts > 1:
            set_spinner_suffix(f"attempt {attempt}/{attempts}")
        for host, (ok, message) in _probe_hosts(pending, timeout, port).items():
            outcomes[host] = (ok, message)
            if ok:
                passed_on[host] = attempt
        pending = [h for h in hosts if not outcomes[h][0]]
        if not pending:
            break
        # "attempt N/M failed" refers to the attempt that just ran, not the next one
        # One host per indented line so a long list never overflows a narrow terminal.
        host_lines = "\n".join(f"    {h}" for h in pending)
        logger.info(
            "Network check attempt %d/%d failed: %d host(s) unreachable; %s\n%s",
            attempt,
            attempts,
            len(pending),
            f"retrying in {retry_delay:.1f}s" if attempt < attempts else "no attempts left",
            host_lines,
        )
        if attempt == attempts:
            break
        time.sleep(retry_delay)

    set_spinner_suffix("")
    messages = []
    for host in hosts:
        ok, message = outcomes[host]
        if ok and passed_on.get(host, 1) > 1:
            message = f"{message} (after {passed_on[host]} attempts)"
        elif not ok and attempts > 1:
            message = f"{message} [{attempts} attempts]"
        messages.append(message)
    return not pending, messages


def _probe_hosts(hosts: list[str], timeout: float, port: int) -> dict[str, tuple[bool, str]]:
    """Probe ``hosts`` in parallel, returning ``{host: (reachable, message)}``."""
    # urlopen()'s own `timeout` only bounds each individual connect() call -
    # it does NOT bound DNS resolution (getaddrinfo) or iterating over
    # multiple addresses resolved for a host, either of which can block far
    # longer than `timeout` (observed: 60s+ for a single host whose DNS
    # lookup alone took that long). Each host is checked in its own
    # daemon thread, run in parallel against a single shared wall-clock
    # deadline, so a slow/blocked host can never make this check take longer
    # than ~`timeout` seconds overall, and a still-hung thread never blocks
    # interpreter shutdown.
    results: dict = {}
    threads = []
    for host in hosts:
        thread = threading.Thread(target=_connect_host, args=(host, port, timeout, results), daemon=True)
        thread.start()
        threads.append((host, thread))

    deadline = time.monotonic() + timeout
    outcomes: dict[str, tuple[bool, str]] = {}
    for host, thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if host in results:
            outcomes[host] = results[host]
        else:
            outcomes[host] = (False, _mark(False, f"{host}:{port}: unreachable (timed out after {timeout:g}s)"))
    return outcomes


def _last_line(result) -> str:
    """Last non-empty line of stderr/stdout, for a compact one-line failure message."""
    output = (result.stderr or result.stdout or "unknown error").strip()
    return output.splitlines()[-1] if output else "unknown error"


def check_docker_runtime(
    image: str = DEFAULT_DOCKER_SMOKE_IMAGE,
    hosts: list[str] | None = None,
    timeout: float = DEFAULT_DOCKER_SMOKE_TIMEOUT,
) -> tuple[bool, list[str]]:
    """Verify docker is usable end-to-end for the current user, stopping at the first failure:

    1. ``docker version`` as the current user - fails if docker isn't installed/running
       or requires sudo (e.g. user not in the ``docker`` group).
    2. ``docker pull <image>`` - exercises the daemon's own outbound network path.
    3. ``docker run --rm <image> curl ...`` against a host - exercises the
       *container's* outbound network path, which is configured independently of (2).

    ``image`` is only a canary used internally to run the checks and is deliberately
    omitted from the returned messages, which describe each check's outcome, not
    how it was carried out.
    """
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False, [_mark(False, "docker access: not found on PATH")]

    try:
        result = run_command([docker_bin, "version", "--format", "{{.Server.Version}}"], timeout=timeout)
    except Exception as e:
        return False, [_mark(False, f"docker access: check failed ({e})")]
    if not result.success:
        return False, [_mark(False, f"docker access: not accessible without sudo ({_last_line(result)})")]
    messages = [_mark(True, "docker access: OK (no sudo required)")]

    try:
        from sysagent.utils.infrastructure.retry import run_command_with_retry

        # Transient registry/network errors are retried; goes through the daemon, so it also
        # follows any configured registry mirror or proxy.
        pull_result = run_command_with_retry(
            [docker_bin, "pull", image], description="docker pull (readiness)", timeout=timeout
        )
    except Exception as e:
        messages.append(_mark(False, f"image pull: failed ({e})"))
        return False, messages
    if not pull_result.success:
        messages.append(_mark(False, f"image pull: failed ({_last_line(pull_result)})"))
        return False, messages
    messages.append(_mark(True, "image pull: OK"))

    host = (hosts or ["github.com"])[0]
    try:
        run_result = run_command(
            [
                docker_bin,
                "run",
                "--rm",
                image,
                "curl",
                "-sS",
                "-o",
                "/dev/null",
                "-m",
                str(int(timeout)),
                f"https://{host}",
            ],
            timeout=timeout + 10,
        )
    except Exception as e:
        messages.append(_mark(False, f"container network ({host}): check failed ({e})"))
        return False, messages
    if not run_result.success:
        messages.append(_mark(False, f"container network ({host}): unreachable ({_last_line(run_result)})"))
        return False, messages
    messages.append(_mark(True, f"container network ({host}): OK"))

    return True, messages
