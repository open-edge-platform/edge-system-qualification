#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Advanced System Setup
# Applies session-scoped system settings that change runtime behaviour but do
# NOT persist across reboots. Changes here go beyond the read-permission and
# package-installation scope of system-setup.sh and require explicit user
# acknowledgment because they affect resource limits or other kernel tunables
# that could impact system behaviour.
#
# Each module prompts for confirmation before applying.  Use --force to accept
# all modules automatically (CI pipelines and automated provisioning).
#
# Usage:
#   sudo bash system-setup-advanced.sh           Interactive (prompts per module)
#   sudo bash system-setup-advanced.sh --force   Non-interactive (auto-accept all)
#   sudo bash system-setup-advanced.sh -f        Same as --force
#
# Adding new modules:
#   1. Write a _setup_<name>() function following the pattern below.
#   2. Add a row to the module table above.
#   3. Call it from main() with: run_module _setup_<name> "Module Display Name"

set -euo pipefail

CURRENT_USER="${SUDO_USER:-$USER}"
_FORCE=false
_MODULE_PASS=()
_MODULE_SKIP=()
_MODULE_FAIL=()

_is_root() { [ "$EUID" -eq 0 ]; }

_require_root() {
    if _is_root; then
        return 0
    fi
    echo "  [SKIP] Root required. Re-run with sudo."
    _MODULE_SKIP+=("$1")
    return 1
}

# Prompt the user for confirmation unless --force is active.
# Returns 0 (proceed) or 1 (skip).
_confirm() {
    local prompt="$1"
    if [ "$_FORCE" = true ]; then
        echo "  [INFO] --force active: auto-confirming '$prompt'"
        return 0
    fi
    local answer
    read -r -p "  $prompt [y/N] " answer
    case "$answer" in
        [Yy]*) return 0 ;;
        *)     return 1 ;;
    esac
}

# Run a module function with a confirmation prompt.
# Usage: run_module <function_name> "<Display Name>"
run_module() {
    local func="$1" display="$2"
    echo ""
    echo "--- $display ---"
    if _confirm "Apply '$display'?"; then
        "$func" "$display"
    else
        echo "  [SKIP] Skipped by user."
        _MODULE_SKIP+=("$display")
    fi
}

# ---------------------------------------------------------------------------
# Module 1: Locked Memory Limit — current session only
# ---------------------------------------------------------------------------
# Sets RLIMIT_MEMLOCK to unlimited for the current shell and all ancestor
# processes so memtester can mlock() its full dynamic allocation.
# The change applies to the CURRENT SESSION ONLY — it resets automatically
# after the next reboot with no cleanup required.
_setup_memlock_session() {
    local MODULE="$1"

    _require_root "$MODULE" || return 0

    # Raise the limit in the current shell.
    ulimit -l unlimited 2>/dev/null \
        && echo "  [OK] ulimit -l unlimited applied to current shell" \
        || echo "  [WARN] ulimit -l unlimited failed in this shell (may already be set)"

    # Walk the ancestor process tree so the calling terminal also gets the
    # raised limit; child processes inherit limits from their parent at fork time.
    if command -v prlimit > /dev/null 2>&1; then
        local pid="$$" applied=0
        while true; do
            pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
            if ! [ "${pid:-0}" -gt 1 ] 2>/dev/null; then break; fi
            if prlimit --memlock=unlimited:unlimited -p "$pid" 2>/dev/null; then
                applied=$((applied + 1))
                echo "  [OK] prlimit unlimited applied to PID $pid"
            fi
        done
        [ "$applied" -eq 0 ] \
            && echo "  [INFO] Run in your terminal to apply: ulimit -l unlimited"
    else
        echo "  [INFO] prlimit unavailable; run in your terminal: ulimit -l unlimited"
    fi

    echo "  [INFO] Limit resets to system default after reboot."
    _MODULE_PASS+=("$MODULE")
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    # Parse flags
    for arg in "$@"; do
        case "$arg" in
            --force|-f) _FORCE=true ;;
            --help|-h)
                sed -n '/^#/!q; s/^# \{0,1\}//p' "$0"
                exit 0
                ;;
            *)
                echo "[ERROR] Unknown argument: $arg"
                echo "Usage: sudo bash $0 [--force|-f]"
                exit 1
                ;;
        esac
    done

    # All modules in this script require root. Exit early so the user is not
    # prompted for each module only to see every one silently skipped.
    if ! _is_root; then
        echo "[ERROR] This script must be run as root."
        echo "Usage: sudo bash $0 [--force|-f]"
        exit 1
    fi

    echo "=== Advanced System Setup (session-only) ==="
    echo "Running as: $CURRENT_USER"
    if [ "$_FORCE" = true ]; then
        echo "Mode: --force (non-interactive, all modules accepted)"
    else
        echo "Mode: interactive (prompts per module)"
        echo "Note: all changes apply to the current session only and reset after reboot."
    fi

    run_module _setup_memlock_session "Locked Memory Limit (session)"

    echo ""
    echo "=== Summary ==="
    local m
    if [ ${#_MODULE_PASS[@]} -gt 0 ]; then
        for m in "${_MODULE_PASS[@]}"; do echo "  [OK]   $m"; done
    fi
    if [ ${#_MODULE_SKIP[@]} -gt 0 ]; then
        for m in "${_MODULE_SKIP[@]}"; do echo "  [SKIP] $m"; done
    fi
    if [ ${#_MODULE_FAIL[@]} -gt 0 ]; then
        for m in "${_MODULE_FAIL[@]}"; do echo "  [FAIL] $m"; done
    fi
    echo ""
    if [ ${#_MODULE_FAIL[@]} -eq 0 ]; then
        echo "Advanced setup complete."
    else
        echo "Advanced setup complete with errors. Review the output above."
        return 1
    fi
}

main "$@"

