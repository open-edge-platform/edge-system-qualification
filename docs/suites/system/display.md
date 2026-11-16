# Display Connectivity

Counts the display ports exposed by the DRM subsystem and how many of them have a monitor connected. It collects data only and does not fail on missing displays.

| | |
|---|---|
| **Test suite** | `test_display_connectivity` (`src/esq/suites/system/display/test_display_connectivity.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ checks the system directly, and the profile selects the checks. |
| **Key metric** | `total_connected` (all ports), `hdmi_ports_connected` (HDMI), or `dp_ports_connected` (DisplayPort), in ports |
| **Devices** | DRM display connectors (`/sys/class/drm`) |

## How It Works

1. List the DRM cards and their connectors, optionally only those of `drm_device`.
2. Keep the connectors selected by the `check_*` flags. HDMI matches connector types containing `HDMI`, DisplayPort matches types containing `DP` (including eDP).
3. Count total and connected ports per type. A port is connected when its DRM status is `connected`.
4. Save the device and port list, with the modes of connected ports, to `esq_data/data/suites/system/display/results/<test_id>/`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `check_all`, `check_hdmi`, `check_displayport` | `false` | Ports to check: all connectors, HDMI only, or DisplayPort only. Set one per test case. |
| `drm_device` | unset | DRM card to check, for example `card0`. Unset checks all cards. |

## Profiles

<!-- esq:suite-profiles test_display_connectivity -->

## Run

```bash
esq run --profile profile.horizontal.system.display --filter test_id=SYS-DISP-001
```
