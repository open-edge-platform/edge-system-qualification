# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Pytest telemetry fixture for the sysagent framework.

Provides an ``autouse`` function-scoped fixture that automatically starts
background telemetry collection when a test's ``configs`` dict contains a
``telemetry`` block with ``enabled: true``.

Telemetry results are stored on the pytest ``request.node`` and are
automatically applied to the ``Result`` object inside ``summarize_test_results``.

The fixture is intentionally designed so that:
- Existing tests require NO code changes to gain telemetry support.
- Telemetry overhead is minimal (daemon background thread, configurable interval).
- Disabling telemetry in the profile YAML makes the fixture a complete no-op.
"""

import logging

import pytest

logger = logging.getLogger(__name__)

# Attribute name used to stash the collector on the pytest node
_TELEMETRY_ATTR = "_sysagent_telemetry_collector"


def _resolve_scope(telemetry_cfg) -> str:
    """
    Resolve the telemetry collection scope.

    Supported values:
    - ``execution`` (default): collector is constructed during setup but the
      sampling thread is only started around the actual test workload via
      ``execute_test_with_cache``. Excludes preparation phases such as
      Docker image build, asset/model download, container launch, and
      result post-processing.
    - ``test`` (legacy): sampling runs for the entire pytest function
      lifecycle (setup -> body -> teardown), matching the historical
      behavior prior to scoping support.

    Resolution order (highest precedence first):
    1. ``telemetry.scope`` in profile YAML.
    2. ``CORE_TELEMETRY_SCOPE`` environment variable.
    3. Default: ``execution``.
    """
    import os as _os

    raw = telemetry_cfg.get("scope") if isinstance(telemetry_cfg, dict) else None
    if not raw:
        raw = _os.environ.get("CORE_TELEMETRY_SCOPE")
    scope = str(raw or "execution").strip().lower()
    if scope not in {"execution", "test"}:
        logger.warning(
            "Unsupported telemetry.scope '%s'; falling back to 'execution'.", scope
        )
        scope = "execution"
    return scope


# Attribute name used to stash the resolved scope on the pytest node so that
# execute_test_with_cache can decide whether to start/stop the collector.
_TELEMETRY_SCOPE_ATTR = "_sysagent_telemetry_scope"


@pytest.fixture(scope="function", autouse=True)
def _auto_telemetry(request, configs):
    """
    Construct (and optionally run) the telemetry collector for a test.

    Reads ``configs["telemetry"]`` to determine whether collection is enabled
    and which modules to activate. The collector is attached to ``request.node``
    so that ``summarize_test_results`` can apply the collected data to the
    ``Result`` without any changes to test code.

    Scope (``telemetry.scope`` in profile YAML):
    - ``execution`` (default): the collector is created here but its sampling
      thread is started by ``execute_test_with_cache`` immediately before the
      workload function runs and stopped immediately after. This excludes
      docker build / asset preparation / teardown from the collected window.
    - ``test`` (legacy): the collector starts here (during pytest setup) and
      stops in teardown, matching the previous behavior.

    The fixture is a no-op when:
    - ``configs`` does not contain a ``telemetry`` key.
    - ``configs["telemetry"]["enabled"]`` is False (or absent).
    - No telemetry modules could be loaded (e.g., missing dependencies).
    """
    # Lazy import to avoid hard dependency at plugin load time
    try:
        from sysagent.utils.telemetry.collector import TelemetryCollector
    except ImportError as exc:
        logger.debug("Telemetry framework not available: %s", exc)
        yield
        return

    telemetry_cfg = configs.get("telemetry") or {}
    if not telemetry_cfg.get("enabled", False):
        # Telemetry disabled in this profile/test — skip entirely
        yield
        return

    collector = TelemetryCollector(configs)
    if not collector._modules:
        logger.debug("No telemetry modules loaded; skipping telemetry collection.")
        yield
        return

    # Stash collector + scope on the node so other fixtures can find them
    setattr(request.node, _TELEMETRY_ATTR, collector)
    scope = _resolve_scope(telemetry_cfg)
    setattr(request.node, _TELEMETRY_SCOPE_ATTR, scope)

    if scope == "test":
        collector.start()
    else:
        logger.debug(
            "Telemetry scope='execution': collector created but not started; "
            "execute_test_with_cache will start/stop it around the workload."
        )

    try:
        yield
    finally:
        # Always stop in teardown as a safety net (no-op if never started, or
        # already stopped by execute_test_with_cache).
        collector.stop()
        logger.debug(
            "Telemetry collection complete for test: %s (scope=%s)",
            request.node.nodeid,
            scope,
        )


def get_telemetry_collector(request_node):
    """
    Retrieve the telemetry collector attached to a pytest node, if any.

    Args:
        request_node: The ``request.node`` object from a pytest fixture.

    Returns:
        The TelemetryCollector instance, or None if telemetry was not active.
    """
    return getattr(request_node, _TELEMETRY_ATTR, None)


def get_telemetry_scope(request_node) -> str:
    """
    Retrieve the resolved telemetry scope for a pytest node.

    Returns:
        ``execution``, ``test``, or ``""`` if telemetry was not active.
    """
    return getattr(request_node, _TELEMETRY_SCOPE_ATTR, "") or ""
