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


@pytest.fixture(scope="function", autouse=True)
def _auto_telemetry(request, configs):
    """
    Automatically start/stop telemetry collection for every test function.

    Reads ``configs["telemetry"]`` to determine whether collection is enabled
    and which modules to activate.  The collector is attached to
    ``request.node`` so that ``summarize_test_results`` can apply the
    collected data to the ``Result`` without any changes to test code.

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

    # Stash collector on the node so summarize_test_results can find it
    setattr(request.node, _TELEMETRY_ATTR, collector)

    collector.start()
    try:
        yield
    finally:
        collector.stop()
        logger.debug("Telemetry collection complete for test: %s", request.node.nodeid)


def get_telemetry_collector(request_node):
    """
    Retrieve the telemetry collector attached to a pytest node, if any.

    Args:
        request_node: The ``request.node`` object from a pytest fixture.

    Returns:
        The TelemetryCollector instance, or None if telemetry was not active.
    """
    return getattr(request_node, _TELEMETRY_ATTR, None)
