# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Network utilities for download source selection.

Reachability checks live in the pre-run readiness checks
(``sysagent.utils.core.dependencies.prerun``), not here.
"""

import logging
import os

logger = logging.getLogger(__name__)


def get_preferred_download_source(
    source_type: str | None = None,
    default: str = "huggingface",
    valid_sources: list | None = None,
) -> str:
    """
    Determine which download source to use based on preference flags in environment variables.

    Uses flag-based environment variables for flexible, discoverable source configuration:
    - PREFER_<SOURCE>_<TYPE>=1  (e.g., PREFER_HUGGINGFACE_MODEL=1)
    - PREFER_<SOURCE>=1         (e.g., PREFER_MODELSCOPE=1)

    Priority order:
    1. Type-specific preference flags (PREFER_<SOURCE>_<TYPE>=1)
    2. Global preference flags (PREFER_<SOURCE>=1)
    3. Function parameter default (caller-specified, defaults to "huggingface")

    This approach allows:
    - Easy discovery of new sources without modifying shared code
    - Test suites can add new sources by setting PREFER_<NEWSOURCE>=1
    - No hardcoded source names in the function
    - Simple boolean flags instead of string values

    Examples:
        # Prefer ModelScope globally
        export PREFER_MODELSCOPE=1

        # Prefer HuggingFace for models, ModelScope for datasets
        export PREFER_HUGGINGFACE_MODEL=1
        export PREFER_MODELSCOPE_DATASET=1

        # Prefer Ultralytics for video downloads
        export PREFER_ULTRALYTICS_VIDEO=1

    Common source types:
    - "model": Model downloads
    - "dataset": Dataset downloads
    - "video": Video/media downloads

    Args:
        source_type: Optional resource type (e.g., "model", "dataset", "video")
                    Used to check type-specific flags (PREFER_<SOURCE>_<TYPE>)
        default: Default source if no preference flag is set (default: "huggingface")
        valid_sources: Optional list of valid sources for this context
                      If provided, validates the source and warns on invalid values

    Returns:
        str: Download source identifier (lowercase)
    """
    source = None
    source_origin = None

    # Priority 1: Check type-specific preference flags (PREFER_<SOURCE>_<TYPE>)
    if source_type:
        type_suffix = f"_{source_type.upper()}"
        for env_var, env_value in os.environ.items():
            if env_var.startswith("PREFER_") and env_var.endswith(type_suffix):
                # Extract source name: PREFER_HUGGINGFACE_MODEL -> huggingface
                source_name = env_var[len("PREFER_") : -len(type_suffix)].lower()
                # Check if flag is set (any truthy value: 1, true, yes, etc.)
                if env_value.lower() in ("1", "true", "yes", "on"):
                    source = source_name
                    source_origin = env_var
                    break

    # Priority 2: Check global preference flags (PREFER_<SOURCE>)
    if source is None:
        for env_var, env_value in os.environ.items():
            if env_var.startswith("PREFER_") and "_" not in env_var[len("PREFER_") :]:
                # Extract source name: PREFER_MODELSCOPE -> modelscope
                source_name = env_var[len("PREFER_") :].lower()
                # Check if flag is set
                if env_value.lower() in ("1", "true", "yes", "on"):
                    source = source_name
                    source_origin = env_var
                    break

    # Priority 3: Use default
    if source is None:
        source = default.lower()
        source_origin = "default parameter"

    # Validate against valid_sources if provided
    if valid_sources:
        valid_sources_lower = [s.lower() for s in valid_sources]
        if source not in valid_sources_lower:
            logger.warning(
                f"Invalid download source '{source}' from {source_origin}. "
                f"Valid sources: {', '.join(valid_sources)}. "
                f"Defaulting to '{default}'."
            )
            source = default.lower()
            source_origin = "default (after validation)"

    # Log the selection
    type_label = f" for {source_type}" if source_type else ""
    logger.debug(f"Using '{source}' as download source{type_label} (from {source_origin})")

    return source
