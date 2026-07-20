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
# Modules
# -------
# 1. Locked Memory Limit (session) -- RLIMIT_MEMLOCK unlimited; resets on reboot.
# 2. MSR CAP_SYS_RAWIO (session)   -- read/write access on /dev/cpu/N/msr via
#                                     group ownership (0660); resets on reboot.
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
# Module 2: MSR CAP_SYS_RAWIO — TCC MSR access (session-only)
# ---------------------------------------------------------------------------
# Enables non-root MSR read/write access (CAP_SYS_RAWIO) for the current
# session. Required for Intel TCC detection (tcc_capable): the Linux MSR
# kernel driver checks CAP_SYS_RAWIO in msr_open() regardless of file
# permissions — chmod/chown alone cannot grant access.
#
# Mechanism:
#   1. Ensures the 'msr' kernel module is loaded (modprobe msr).
#   2. Locates the system Python3 ELF binary (resolves symlinks, since file
#      capabilities are stored on the real inode, not on symlinks).
#   3. Mounts a fresh tmpfs at /dev/shm/esq/ (NOT nosuid — file capabilities
#      are honoured). Shared with the TC module if already mounted.
#   4. Copies python3 into /dev/shm/esq/python3-msr.
#   5. Applies setcap cap_sys_rawio+ep to the COPY only — the system Python3
#      binary is never modified.
#
# Session-only guarantee:
#   On reboot, /dev/shm is recreated as a fresh empty tmpfs. The esq/
#   sub-mount is not in /etc/fstab so it is never re-established. Both the
#   mount and the copied binary are gone without any explicit cleanup.
#
# To restore after reboot: re-run this script.
# To revoke manually:  sudo umount /dev/shm/esq && sudo rm -rf /dev/shm/esq
# ---------------------------------------------------------------------------
_setup_msr_rawio() {
    local MODULE="$1"
    local SESSION_DIR="/dev/shm/esq"
    local SESSION_PYTHON="$SESSION_DIR/python3-msr"

    _require_root "$MODULE" || return 0

    # Verify setcap/getcap are available (provided by libcap2-bin).
    if ! command -v setcap > /dev/null 2>&1 || ! command -v getcap > /dev/null 2>&1; then
        echo "  [ERROR] setcap/getcap not found. Install libcap2-bin:"
        echo "          sudo apt-get install libcap2-bin"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi

    # Ensure the msr kernel module is loaded.
    if lsmod | grep -q "^msr "; then
        echo "  [INFO] msr kernel module already loaded"
    elif modprobe msr 2>/dev/null; then
        echo "  [OK] msr kernel module loaded"
    else
        echo "  [ERROR] Failed to load msr module."
        echo "          Verify it is available: modinfo msr"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi

    # Check if the session python3-msr is already configured and valid.
    if [ -f "$SESSION_PYTHON" ]; then
        local EXISTING_CAPS
        EXISTING_CAPS=$(getcap "$SESSION_PYTHON" 2>/dev/null || true)
        if printf '%s' "$EXISTING_CAPS" | grep -q "cap_sys_rawio"; then
            echo "  [OK] Session python3-msr already configured: $EXISTING_CAPS"
            echo "  [INFO] Session python3-msr: $SESSION_PYTHON (session-only, cleared on reboot)"
            _MODULE_PASS+=("$MODULE")
            return 0
        fi
    fi

    # Locate the real Python3 ELF binary. File capabilities are stored on the
    # inode, not on symlinks — readlink -f resolves to the actual binary file.
    local PYTHON_SYS="" candidate
    for candidate in /usr/bin/python3 /usr/local/bin/python3; do
        if [ -f "$candidate" ]; then
            PYTHON_SYS=$(readlink -f "$candidate")
            break
        fi
    done
    if [ -z "$PYTHON_SYS" ]; then
        local PY_CMD
        PY_CMD=$(command -v python3 2>/dev/null || true)
        [ -n "$PY_CMD" ] && PYTHON_SYS=$(readlink -f "$PY_CMD" 2>/dev/null || true)
    fi
    if [ -z "$PYTHON_SYS" ] || [ ! -f "$PYTHON_SYS" ]; then
        echo "  [ERROR] python3 not found. Install python3:"
        echo "          sudo apt-get install python3"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    echo "  [INFO] System python3: $PYTHON_SYS (unchanged)"

    # Mount a fresh tmpfs at SESSION_DIR without nosuid.
    # /dev/shm is mounted with nosuid, which disables file capabilities on
    # files inside it. A separate inner tmpfs mount (no nosuid by default)
    # works around this. Shared with the TC module if already mounted.
    mkdir -p "$SESSION_DIR"
    if ! mountpoint -q "$SESSION_DIR"; then
        if ! mount -t tmpfs tmpfs "$SESSION_DIR"; then
            echo "  [ERROR] Failed to mount tmpfs at $SESSION_DIR"
            _MODULE_FAIL+=("$MODULE")
            return 1
        fi
        echo "  [OK] Mounted tmpfs at $SESSION_DIR (no nosuid; session-only)"
    else
        echo "  [INFO] $SESSION_DIR already mounted"
    fi

    # Copy python3 into the session tmpfs and lock down permissions.
    if ! cp "$PYTHON_SYS" "$SESSION_PYTHON"; then
        echo "  [ERROR] Failed to copy $PYTHON_SYS to $SESSION_PYTHON"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    chmod 0755 "$SESSION_PYTHON"
    chown root:root "$SESSION_PYTHON"
    echo "  [OK] Copied python3 to $SESSION_PYTHON"

    # Apply cap_sys_rawio+ep to the session copy only.
    if ! setcap cap_sys_rawio+ep "$SESSION_PYTHON"; then
        echo "  [ERROR] setcap failed on $SESSION_PYTHON"
        echo "  [INFO] Verify the tmpfs is not nosuid: cat /proc/mounts | grep esq"
        rm -f "$SESSION_PYTHON"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    local NEW_CAPS
    NEW_CAPS=$(getcap "$SESSION_PYTHON" 2>/dev/null || true)
    echo "  [OK] Capability applied: $NEW_CAPS"
    echo "  [INFO] Session python3-msr: $SESSION_PYTHON"
    echo "  [INFO] Automatically cleared on reboot. Re-run this script to restore."

    # Verify: the calling user can now read /dev/cpu/0/msr via session python.
    local CALLING_USER="${SUDO_USER:-}"
    if [ -f "/dev/cpu/0/msr" ] && [ -n "$CALLING_USER" ] && [ "$CALLING_USER" != "root" ]; then
        local TEST_OUT
        TEST_OUT=$(sudo -u "$CALLING_USER" "$SESSION_PYTHON" -c "
import os, struct
try:
    with open('/dev/cpu/0/msr', 'rb') as f:
        raw = os.pread(f.fileno(), 8, 0x3B)
    print('ok')
except Exception as e:
    print('error: ' + str(e))
" 2>&1 || true)
        if printf '%s' "$TEST_OUT" | grep -q "^ok$"; then
            echo "  [OK] Verified: '$CALLING_USER' can read MSR via session python3-msr"
        else
            echo "  [WARN] MSR access verification failed: $TEST_OUT"
        fi
    fi

    _MODULE_PASS+=("$MODULE")
}


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
    run_module _setup_msr_rawio "MSR CAP_SYS_RAWIO (TCC MSR access)"

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

