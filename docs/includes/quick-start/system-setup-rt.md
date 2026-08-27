For RT latency tests, also run the RT setup script:

```bash
sudo bash -c "$(wget -qLO - https://raw.githubusercontent.com/open-edge-platform/edge-system-qualification/refs/heads/main/scripts/system-setup-rt.sh)"
```

!!! warning "Re-run after reboot"
    All changes applied by `system-setup-rt.sh` are **current session only** and reset automatically after reboot. Re-run after each reboot before executing RT tests.

**RT setup modules:**

| Module | What it changes |
|--------|----------------|
| **Real-Time Latency Tools** | Copies `cyclictest` and `chrt` with `cap_sys_nice` and `cap_ipc_lock` capabilities to a session tmpfs so RT latency tests run at `SCHED_FIFO` priority without `sudo`. Cleared on reboot |
| **MSR Tools** | Copies `rdmsr` and `wrmsr` with `cap_sys_rawio` to the same session tmpfs for Intel® RDT/CAT partition reporting and L3 CAT write verification. Cleared on reboot |
| **Kernel Tuning** | Grants world-write on cpuidle sysfs disable files (C-state control); disables timer migration (`/proc/sys/kernel/timer_migration=0`); sets up a session `tee` with `cap_sys_admin` for non-root cpuidle writes. Cleared on reboot |
