#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Real-Time System Setup
# Applies session-scoped settings required for RT latency tests.
# Changes reset on reboot — re-run after each reboot before executing RT tests.
#
# Usage:
#   sudo bash system-setup-rt.sh           Interactive (prompts per module)
#   sudo bash system-setup-rt.sh --force   Non-interactive (auto-accept all)
#   sudo bash system-setup-rt.sh -f        Same as --force
#
# Modules
# -------
# 1. Real-Time Latency Tools   -- cyclictest/chrt session copies with
#                                 cap_sys_nice+cap_ipc_lock in /run/user/<UID>/esq/;
#                                 accessible only by the calling user; resets on reboot.
# 2. MSR Tools (rdmsr/wrmsr)   -- rdmsr+wrmsr session copies with
#                                 cap_sys_rawio+cap_dac_read_search/cap_dac_override;
#                                 enables non-root MSR read/write for RDT/CAT reporting
#                                 and L3 CAT write verification; resets on reboot.
# 3. Kernel Tuning             -- cpuidle sysfs world-write (chmod o+w) for C-state
#                                 control; timer_migration=0 written directly as root;
#                                 session tee with cap_sys_admin+ep for non-root cpuidle
#                                 writes at test runtime; resets on reboot.
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

# Detect whether the running kernel was built with CONFIG_PREEMPT_RT=y.
# Returns 0 (true) when PREEMPT_RT is active, 1 otherwise.
_is_preempt_rt_kernel() {
    local config_file
    config_file="/boot/config-$(uname -r)"
    if [[ -f "$config_file" ]] && grep -q '^CONFIG_PREEMPT_RT=y' "$config_file"; then
        return 0
    fi
    return 1
}

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
# Module 1: Real-Time Latency Tools — cyclictest/chrt file capabilities (session-only)
# ---------------------------------------------------------------------------
# Copies /usr/bin/cyclictest and /usr/bin/chrt into a user-specific tmpfs at
# /run/user/<UID>/esq/ and applies file capabilities only to those copies.
# On reboot the tmpfs is gone — no cleanup required.
#
# Security advantages over permanent setcap on system binaries:
#   1. Session-scoped: capability lifetime is limited to a single boot.
#   2. User-scoped: the directory /run/user/<UID>/esq/ is owned by the calling
#      user with mode 0700, so no other unprivileged user can execute the
#      capability-enhanced copies.
#
# Capabilities applied:
#   cyclictest: cap_sys_nice,cap_ipc_lock+ep  (SCHED_FIFO priority + mlockall)
#   chrt:       cap_sys_nice+ep               (SCHED_FIFO wrapper; no mlockall)
# ---------------------------------------------------------------------------
_setup_realtime_latency() {
    local MODULE="$1"

    _require_root "$MODULE" || return 0

    local CALLING_USER CALLING_UID CALLING_GID
    CALLING_USER="${SUDO_USER:-$USER}"
    CALLING_UID=$(id -u "$CALLING_USER" 2>/dev/null || echo "0")
    CALLING_GID=$(id -g "$CALLING_USER" 2>/dev/null || echo "0")

    local SESSION_DIR="/run/user/${CALLING_UID}/esq"
    local SESSION_CYCLIC="$SESSION_DIR/cyclictest"
    local SESSION_CHRT="$SESSION_DIR/chrt"

    echo "  [INFO] Target session dir: $SESSION_DIR (uid=$CALLING_UID)"

    if ! command -v setcap > /dev/null 2>&1 || ! command -v getcap > /dev/null 2>&1; then
        echo "  [ERROR] setcap/getcap not found — run system-setup.sh to install required dependencies."
        echo "  [ERROR] Refer to the installation guide for setup instructions."
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi

    local PARENT_DIR="/run/user/${CALLING_UID}"
    if [ ! -d "$PARENT_DIR" ]; then
        mkdir -p "$PARENT_DIR"
        chown "${CALLING_UID}:${CALLING_GID}" "$PARENT_DIR"
        chmod 0700 "$PARENT_DIR"
        echo "  [INFO] Created $PARENT_DIR (systemd normally creates this on login)"
    fi

    mkdir -p "$SESSION_DIR"
    if ! mountpoint -q "$SESSION_DIR"; then
        if mount -t tmpfs -o "size=64m,uid=${CALLING_UID},gid=${CALLING_GID},mode=0700" tmpfs "$SESSION_DIR"; then
            echo "  [OK] Mounted tmpfs at $SESSION_DIR (uid=$CALLING_UID, mode=0700, no nosuid)"
        else
            echo "  [ERROR] Failed to mount tmpfs at $SESSION_DIR"
            _MODULE_FAIL+=("$MODULE")
            return 1
        fi
    else
        echo "  [INFO] $SESSION_DIR already mounted"
    fi

    # --- cyclictest: cap_sys_nice + cap_ipc_lock ---
    local CYCLIC_BIN CYCLIC_REAL
    CYCLIC_BIN=$(command -v cyclictest 2>/dev/null || true)
    if [ -z "$CYCLIC_BIN" ]; then
        echo "  [ERROR] cyclictest not found — run system-setup.sh to install required dependencies."
        echo "  [ERROR] Refer to the installation guide for setup instructions."
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    CYCLIC_REAL=$(readlink -f "$CYCLIC_BIN" 2>/dev/null || echo "$CYCLIC_BIN")
    echo "  [INFO] System cyclictest: $CYCLIC_REAL (unchanged)"

    if ! cp "$CYCLIC_REAL" "$SESSION_CYCLIC"; then
        echo "  [ERROR] Failed to copy cyclictest to $SESSION_CYCLIC"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    chmod 0500 "$SESSION_CYCLIC"
    chown "${CALLING_UID}:${CALLING_GID}" "$SESSION_CYCLIC"

    if ! setcap cap_sys_nice,cap_ipc_lock+ep "$SESSION_CYCLIC"; then
        echo "  [ERROR] setcap failed for $SESSION_CYCLIC"
        rm -f "$SESSION_CYCLIC"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    local CYCLIC_CAPS
    CYCLIC_CAPS=$(getcap "$SESSION_CYCLIC" 2>/dev/null || true)
    echo "  [OK] Session cyclictest: $SESSION_CYCLIC [$CYCLIC_CAPS]"

    # --- chrt: cap_sys_nice only (chrt does not call mlockall) ---
    local CHRT_BIN CHRT_REAL
    CHRT_BIN=$(command -v chrt 2>/dev/null || true)
    if [ -z "$CHRT_BIN" ]; then
        echo "  [WARN] chrt not found (util-linux); cyclictest will run via its own capabilities"
    else
        CHRT_REAL=$(readlink -f "$CHRT_BIN" 2>/dev/null || echo "$CHRT_BIN")
        echo "  [INFO] System chrt: $CHRT_REAL (unchanged)"
        if cp "$CHRT_REAL" "$SESSION_CHRT" && chmod 0500 "$SESSION_CHRT" && chown "${CALLING_UID}:${CALLING_GID}" "$SESSION_CHRT"; then
            if setcap cap_sys_nice+ep "$SESSION_CHRT"; then
                local CHRT_CAPS
                CHRT_CAPS=$(getcap "$SESSION_CHRT" 2>/dev/null || true)
                echo "  [OK] Session chrt: $SESSION_CHRT [$CHRT_CAPS]"
            else
                echo "  [WARN] setcap failed for session chrt; cyclictest will run via its own capabilities"
                rm -f "$SESSION_CHRT"
            fi
        else
            echo "  [WARN] Failed to copy chrt; cyclictest will run via its own capabilities"
        fi
    fi

    if [ -n "$CALLING_USER" ] && [ "$CALLING_USER" != "root" ]; then
        local TEST_OUT
        TEST_OUT=$(sudo -u "$CALLING_USER" "$SESSION_CYCLIC" --help 2>&1 | head -1 || true)
        if [ -n "$TEST_OUT" ]; then
            echo "  [OK] Verified: '$CALLING_USER' can invoke session cyclictest without sudo"
        else
            echo "  [WARN] Could not verify session cyclictest invocation for '$CALLING_USER'"
        fi
    fi

    echo "  [INFO] Automatically cleared on reboot. Re-run this script to restore."
    _MODULE_PASS+=("$MODULE")
}


# ---------------------------------------------------------------------------
# Module 2: MSR Tools — rdmsr/wrmsr file capabilities (session-only)
# ---------------------------------------------------------------------------
# Copies /usr/sbin/rdmsr and /usr/sbin/wrmsr (msr-tools) into the user-
# specific tmpfs at /run/user/<UID>/esq/ with separate capabilities:
#
#   rdmsr: cap_sys_rawio,cap_dac_read_search+ep
#     — read-only MSR access for Intel RDT/CAT partition reporting
#       (IA32_PQR_ASSOC, L3/L2 bitmasks).
#
#   wrmsr: cap_sys_rawio,cap_dac_override+ep
#     — read+write MSR access for L3 CAT write verification
#       (round-trip write IA32_L3_QOS_MASK_0).
#
# Both require CAP_SYS_RAWIO because msr_open() always checks it regardless
# of file permissions on the device node.
# ---------------------------------------------------------------------------
_setup_rdmsr_cap() {
    local MODULE="$1"

    _require_root "$MODULE" || return 0

    local CALLING_USER CALLING_UID CALLING_GID
    CALLING_USER="${SUDO_USER:-$USER}"
    CALLING_UID=$(id -u "$CALLING_USER" 2>/dev/null || echo "0")
    CALLING_GID=$(id -g "$CALLING_USER" 2>/dev/null || echo "0")

    local SESSION_DIR="/run/user/${CALLING_UID}/esq"
    local SESSION_RDMSR="$SESSION_DIR/rdmsr"

    echo "  [INFO] Target session dir: $SESSION_DIR (uid=$CALLING_UID)"

    if ! command -v setcap > /dev/null 2>&1 || ! command -v getcap > /dev/null 2>&1; then
        echo "  [ERROR] setcap/getcap not found — run system-setup.sh to install required dependencies."
        echo "  [ERROR] Refer to the installation guide for setup instructions."
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi

    local RDMSR_BIN RDMSR_REAL
    RDMSR_BIN=$(command -v rdmsr 2>/dev/null || true)
    if [ -z "$RDMSR_BIN" ]; then
        echo "  [ERROR] rdmsr not found — run system-setup.sh to install required dependencies."
        echo "  [ERROR] Refer to the installation guide for setup instructions."
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    RDMSR_REAL=$(readlink -f "$RDMSR_BIN" 2>/dev/null || echo "$RDMSR_BIN")
    echo "  [INFO] System rdmsr: $RDMSR_REAL (unchanged)"

    if lsmod | grep -q "^msr "; then
        echo "  [INFO] msr kernel module already loaded"
    elif modprobe msr 2>/dev/null; then
        echo "  [OK] msr kernel module loaded"
    else
        echo "  [ERROR] Failed to load msr kernel module. Check kernel config."
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi

    [ -f "$SESSION_RDMSR" ] && rm -f "$SESSION_RDMSR"

    local PARENT_DIR="/run/user/${CALLING_UID}"
    if [ ! -d "$PARENT_DIR" ]; then
        mkdir -p "$PARENT_DIR"
        chown "${CALLING_UID}:${CALLING_GID}" "$PARENT_DIR"
        chmod 0700 "$PARENT_DIR"
        echo "  [INFO] Created $PARENT_DIR (systemd normally creates this on login)"
    fi

    mkdir -p "$SESSION_DIR"
    if ! mountpoint -q "$SESSION_DIR"; then
        if mount -t tmpfs -o "size=64m,uid=${CALLING_UID},gid=${CALLING_GID},mode=0700" tmpfs "$SESSION_DIR"; then
            echo "  [OK] Mounted tmpfs at $SESSION_DIR (uid=$CALLING_UID, mode=0700, no nosuid)"
        else
            echo "  [ERROR] Failed to mount tmpfs at $SESSION_DIR"
            _MODULE_FAIL+=("$MODULE")
            return 1
        fi
    else
        echo "  [INFO] $SESSION_DIR already mounted"
    fi

    if ! cp "$RDMSR_REAL" "$SESSION_RDMSR"; then
        echo "  [ERROR] Failed to copy rdmsr to $SESSION_RDMSR"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    chmod 0500 "$SESSION_RDMSR"
    chown "${CALLING_UID}:${CALLING_GID}" "$SESSION_RDMSR"
    echo "  [OK] Copied rdmsr to $SESSION_RDMSR"

    # cap_sys_rawio         : satisfies msr_open() kernel check.
    # cap_dac_read_search   : bypasses VFS DAC read check on /dev/cpu/N/msr.
    if ! setcap "cap_sys_rawio,cap_dac_read_search+ep" "$SESSION_RDMSR"; then
        echo "  [ERROR] setcap cap_sys_rawio,cap_dac_read_search+ep failed for $SESSION_RDMSR"
        rm -f "$SESSION_RDMSR"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    local NEW_CAPS
    NEW_CAPS=$(getcap "$SESSION_RDMSR" 2>/dev/null || true)
    echo "  [OK] Capability applied: $NEW_CAPS"

    if [ -n "$CALLING_USER" ] && [ "$CALLING_USER" != "root" ]; then
        local TEST_OUT
        TEST_OUT=$(sudo -u "$CALLING_USER" "$SESSION_RDMSR" -p 0 0xC8F 2>&1 || true)
        if [ -n "$TEST_OUT" ]; then
            echo "  [OK] Verified: '$CALLING_USER' can read MSR 0xC8F (IA32_PQR_ASSOC) without sudo"
            echo "  [INFO] cpu0 IA32_PQR_ASSOC = $TEST_OUT"
        else
            echo "  [WARN] Could not verify MSR read for '$CALLING_USER' — check msr module and permissions"
        fi
    fi

    echo "  [INFO] Session rdmsr at: $SESSION_RDMSR"

    # ── wrmsr (MSR write, cap_dac_override) ──────────────────────────────────
    # wrmsr opens /dev/cpu/N/msr with O_RDWR — needs cap_dac_override (not
    # just cap_dac_read_search) to bypass the VFS write-permission check.
    local SESSION_WRMSR="$SESSION_DIR/wrmsr"
    local WRMSR_BIN WRMSR_REAL
    WRMSR_BIN=$(command -v wrmsr 2>/dev/null || true)
    if [ -z "$WRMSR_BIN" ]; then
        echo "  [WARN] wrmsr not found — run system-setup.sh to install required dependencies."
        echo "  [WARN] L3 CAT write verification will be skipped."
    else
        WRMSR_REAL=$(readlink -f "$WRMSR_BIN" 2>/dev/null || echo "$WRMSR_BIN")
        echo "  [INFO] System wrmsr: $WRMSR_REAL (unchanged)"

        [ -f "$SESSION_WRMSR" ] && rm -f "$SESSION_WRMSR"

        if ! cp "$WRMSR_REAL" "$SESSION_WRMSR"; then
            echo "  [ERROR] Failed to copy wrmsr to $SESSION_WRMSR"
            _MODULE_FAIL+=("$MODULE")
            return 1
        fi
        chmod 0500 "$SESSION_WRMSR"
        chown "${CALLING_UID}:${CALLING_GID}" "$SESSION_WRMSR"
        echo "  [OK] Copied wrmsr to $SESSION_WRMSR"

        # cap_sys_rawio    : satisfies msr_open() kernel check.
        # cap_dac_override : bypasses VFS DAC check for both read and write.
        if ! setcap "cap_sys_rawio,cap_dac_override+ep" "$SESSION_WRMSR"; then
            echo "  [ERROR] setcap cap_sys_rawio,cap_dac_override+ep failed for $SESSION_WRMSR"
            rm -f "$SESSION_WRMSR"
            echo "  [WARN] L3 CAT write verification will be unavailable."
        else
            local WRMSR_CAPS
            WRMSR_CAPS=$(getcap "$SESSION_WRMSR" 2>/dev/null || true)
            echo "  [OK] Capability applied: $WRMSR_CAPS"
            echo "  [INFO] Session wrmsr at: $SESSION_WRMSR"
        fi
    fi

    echo "  [INFO] Automatically cleared on reboot. Re-run this script to restore."
    _MODULE_PASS+=("$MODULE")
}


# ---------------------------------------------------------------------------
# Module 3: Kernel Tuning — RT kernel tunables (session-only)
# ---------------------------------------------------------------------------
# Configures session-scoped kernel tunables for RT workloads.
#
# Step 1a — chmod o+w on cpuidle sysfs disable files:
#   Makes files world-writable at the DAC level.  Volatile: resets on reboot.
#   Always runs: grants permission only; no C-state is actually changed until
#   a workload explicitly writes to these files at runtime.
#
# Step 1b — write timer_migration=0 directly as root:
#   /proc/sys uses sysctl_perm (not inode_permission) for access control.
#   sysctl_perm only checks euid == 0 and root group membership — file
#   capabilities like cap_dac_override have no effect.  Writing directly
#   as root during setup is the only session-scoped solution that works.
#   /proc/sys resets to kernel defaults on reboot, so this is inherently
#   session-scoped without any cleanup required.
#   PREEMPT_RT guard: timer_migration=0 takes effect immediately and shifts
#   timer scheduling for all running workloads.  On a standard kernel this
#   degrades power efficiency without any RT benefit, so this step is
#   skipped automatically in --force mode when PREEMPT_RT is not detected.
#   In interactive mode a warning is printed and the user prompt still fires,
#   allowing deliberate override.
#
# Step 2 — session tee with cap_sys_admin+ep:
#   cap_sys_admin : satisfies cpuidle disable_store() capable() check.
#   Python pipes values through the session tee when direct write fails.
#   Always runs: the session tee binary sitting in tmpfs has no effect on
#   the system until a workload actually invokes it at runtime.
#
# Automatically cleared on reboot.  Re-run this script to restore.
# ---------------------------------------------------------------------------
_setup_kernel_tuning() {
    local MODULE="$1"

    _require_root "$MODULE" || return 0

    local CALLING_USER CALLING_UID CALLING_GID
    CALLING_USER="${SUDO_USER:-$USER}"
    CALLING_UID=$(id -u "$CALLING_USER" 2>/dev/null || echo "0")
    CALLING_GID=$(id -g "$CALLING_USER" 2>/dev/null || echo "0")

    local SESSION_DIR="/run/user/${CALLING_UID}/esq"
    local SESSION_TEE="$SESSION_DIR/tee"

    echo "  [INFO] Target session dir: $SESSION_DIR (uid=$CALLING_UID)"

    if ! command -v setcap > /dev/null 2>&1 || ! command -v getcap > /dev/null 2>&1; then
        echo "  [ERROR] setcap/getcap not found — run system-setup.sh to install required dependencies."
        echo "  [ERROR] Refer to the installation guide for setup instructions."
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi

    # ── Step 1a: chmod o+w — DAC write permission on cpuidle sysfs disable files ──
    echo "  [INFO] Step 1a: Granting world-write on cpuidle state disable sysfs files..."
    local count=0 failed=0
    while IFS= read -r -d '' f; do
        if chmod o+w "$f" 2>/dev/null; then
            count=$((count + 1))
        else
            failed=$((failed + 1))
        fi
    done < <(find /sys/devices/system/cpu -name "disable" -path "*/cpuidle/*" -print0 2>/dev/null)

    if [ "$count" -gt 0 ]; then
        echo "  [OK] World-write granted on $count cpuidle disable file(s)"
    else
        echo "  [WARN] No cpuidle disable files found (cpuidle may not be active on this kernel)"
    fi
    if [ "$failed" -gt 0 ]; then
        echo "  [WARN] $failed file(s) could not be chmod'd"
    fi

    # ── Step 1b: write timer_migration=0 directly as root ────────────────────
    # /proc/sys uses sysctl_perm which only checks euid == 0; file capabilities
    # (cap_dac_override etc.) have no effect on this path.  We're running as
    # root here, so write directly.  The value resets to default on reboot.
    # Guard: only meaningful on a PREEMPT_RT kernel — writing 0 on a standard
    # kernel shifts timer scheduling for all running workloads without any RT
    # benefit.  Skipped automatically in --force mode; interactive mode warns
    # but allows deliberate override.
    echo "  [INFO] Step 1b: Disabling timer migration (/proc/sys/kernel/timer_migration=0)..."
    if [ -f /proc/sys/kernel/timer_migration ]; then
        local _apply_timer_migration=true
        if ! _is_preempt_rt_kernel; then
            if [ "$_FORCE" = true ]; then
                echo "  [SKIP] Step 1b: PREEMPT_RT kernel not detected — timer_migration skipped in --force mode."
                echo "  [INFO] Setting timer_migration=0 on a standard kernel shifts timer scheduling for all"
                echo "  [INFO] running workloads without RT benefit. Re-run interactively to override."
                _apply_timer_migration=false
            else
                echo "  [WARN] PREEMPT_RT kernel not detected."
                echo "  [WARN] Setting timer_migration=0 on a standard kernel shifts timer scheduling for all"
                echo "  [WARN] running workloads without any RT benefit. Proceed only if intentional."
            fi
        fi
        if [ "$_apply_timer_migration" = true ]; then
            if echo 0 > /proc/sys/kernel/timer_migration 2>/dev/null; then
                local TMR_ACTUAL
                TMR_ACTUAL=$(cat /proc/sys/kernel/timer_migration 2>/dev/null || echo "?")
                echo "  [OK] Timer migration set to $TMR_ACTUAL (resets to default on reboot)"
            else
                echo "  [WARN] Failed to write /proc/sys/kernel/timer_migration"
            fi
        fi
    else
        echo "  [WARN] /proc/sys/kernel/timer_migration not found (kernel <3.14 or non-standard config)"
    fi

    # ── Step 2: session tee with cap_sys_admin+ep ────────────────────────────
    echo "  [INFO] Step 2: Setting up session tee with cap_sys_admin+ep..."

    local PARENT_DIR="/run/user/${CALLING_UID}"
    if [ ! -d "$PARENT_DIR" ]; then
        mkdir -p "$PARENT_DIR"
        chown "${CALLING_UID}:${CALLING_GID}" "$PARENT_DIR"
        chmod 0700 "$PARENT_DIR"
        echo "  [INFO] Created $PARENT_DIR (systemd normally creates this on login)"
    fi

    mkdir -p "$SESSION_DIR"
    if ! mountpoint -q "$SESSION_DIR"; then
        if mount -t tmpfs -o "size=64m,uid=${CALLING_UID},gid=${CALLING_GID},mode=0700" tmpfs "$SESSION_DIR"; then
            echo "  [OK] Mounted tmpfs at $SESSION_DIR (uid=$CALLING_UID, mode=0700, no nosuid)"
        else
            echo "  [ERROR] Failed to mount tmpfs at $SESSION_DIR"
            _MODULE_FAIL+=("$MODULE")
            return 1
        fi
    else
        echo "  [INFO] $SESSION_DIR already mounted"
    fi

    [ -f "$SESSION_TEE" ] && rm -f "$SESSION_TEE"

    local TEE_BIN TEE_REAL
    TEE_BIN=$(command -v tee 2>/dev/null || true)
    if [ -z "$TEE_BIN" ]; then
        echo "  [ERROR] tee not found — run system-setup.sh to install required dependencies."
        echo "  [ERROR] Refer to the installation guide for setup instructions."
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    TEE_REAL=$(readlink -f "$TEE_BIN" 2>/dev/null || echo "$TEE_BIN")
    echo "  [INFO] System tee: $TEE_REAL (unchanged)"

    if ! cp "$TEE_REAL" "$SESSION_TEE"; then
        echo "  [ERROR] Failed to copy tee to $SESSION_TEE"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    chmod 0500 "$SESSION_TEE"
    chown "${CALLING_UID}:${CALLING_GID}" "$SESSION_TEE"

    # cap_sys_admin : satisfies cpuidle disable_store() capable() check.
    if ! setcap "cap_sys_admin+ep" "$SESSION_TEE"; then
        echo "  [ERROR] setcap cap_sys_admin+ep failed for $SESSION_TEE"
        rm -f "$SESSION_TEE"
        _MODULE_FAIL+=("$MODULE")
        return 1
    fi
    local TEE_CAPS
    TEE_CAPS=$(getcap "$SESSION_TEE" 2>/dev/null || true)
    echo "  [OK] Session tee: $SESSION_TEE [$TEE_CAPS]"

    # Verify the calling user can write a cpuidle disable file via session tee.
    local SAMPLE
    SAMPLE=$(find /sys/devices/system/cpu -name "disable" -path "*/cpuidle/state[1-9]*" -print -quit 2>/dev/null || true)
    if [ -n "$SAMPLE" ] && [ -n "$CALLING_USER" ] && [ "$CALLING_USER" != "root" ]; then
        local ORIG_VAL
        ORIG_VAL=$(cat "$SAMPLE" 2>/dev/null || echo "0")
        local WRITE_OK=false
        if echo "$ORIG_VAL" | sudo -u "$CALLING_USER" "$SESSION_TEE" "$SAMPLE" > /dev/null 2>&1; then
            WRITE_OK=true
        fi
        if [ "$WRITE_OK" = true ]; then
            echo "  [OK] Verified: '$CALLING_USER' can write cpuidle disable via session tee (kernel accepted)"
        else
            echo "  [WARN] Write verification failed for '$CALLING_USER' on $SAMPLE — check kernel support"
        fi
    fi

    echo "  [INFO] Applies to current session only; resets on reboot."
    echo "  [INFO] Re-run this script after reboot to restore."
    _MODULE_PASS+=("$MODULE")
}


main() {
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

    if ! _is_root; then
        echo "[ERROR] This script must be run as root."
        echo "Usage: sudo bash $0 [--force|-f]"
        exit 1
    fi

    echo "=== Real-Time System Setup (session-only) ==="
    echo "Running as: $CURRENT_USER"
    if [ "$_FORCE" = true ]; then
        echo "Mode: --force (non-interactive, all modules accepted)"
    else
        echo "Mode: interactive (prompts per module)"
        echo "Note: all changes apply to the current session only and reset after reboot."
    fi

    run_module _setup_realtime_latency "Real-Time Latency Tools"
    run_module _setup_rdmsr_cap "MSR Tools (rdmsr/wrmsr)"
    run_module _setup_kernel_tuning "Kernel Tuning"

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
        echo "RT setup complete."
    else
        echo "RT setup complete with errors. Review the output above."
        return 1
    fi
}

main "$@"
