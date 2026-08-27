Run `system-setup.sh` once after OS installation to install required system packages and configure file read permissions:

```bash
sudo bash -c "$(wget -qLO - https://raw.githubusercontent.com/open-edge-platform/edge-system-qualification/refs/heads/main/scripts/system-setup.sh)"
```

!!! note "Feature Availability"
    Some features configured by this script are optional. Intel® ESQ will still run if skipped, but certain metrics and capabilities may be unavailable.
