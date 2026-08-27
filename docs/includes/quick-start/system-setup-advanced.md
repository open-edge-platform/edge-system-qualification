Run the advanced setup script to enable full test coverage:

```bash
sudo bash -c "$(wget -qLO - https://raw.githubusercontent.com/open-edge-platform/edge-system-qualification/refs/heads/main/scripts/system-setup-advanced.sh)"
```

!!! warning "Re-run after reboot"
    All changes applied by `system-setup-advanced.sh` are **current session only** and reset automatically after reboot. Re-run the script after each reboot before executing tests that depend on it.

**Advanced setup modules:**

| Module | What it changes |
|--------|----------------|
| **Locked Memory Limit** | Sets `RLIMIT_MEMLOCK` to unlimited for the current session so memtester can lock all available RAM pages. Without this, memory-related tests will be limited or fail |
