# Peripheral Detection

Enumerates the peripheral devices of the system, such as USB controllers and devices, PS/2 input, network and graphics controllers, and connected monitors, and counts the devices of the checked type.

| | |
|---|---|
| **Test suite** | `test_peripheral_detection` (`src/esq/suites/system/peripheral/test_peripheral_detection.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ checks the system directly, and the profile selects the checks. |
| **Key metric** | The count of the checked type, for example `usb_controllers_count` or `monitor_count` (count) |
| **Devices** | Peripherals |

## How It Works

1. Enumerate USB, PS/2, PCI network and graphics devices, and connected monitors.
2. Count the devices of the type selected by the profile entry.
3. Report the count as the key metric.

## Parameters

Each profile entry enables one check:

| Parameter | Key metric | Counts |
|-----------|------------|--------|
| `check_usb_controllers` | `usb_controllers_count` | USB host controllers |
| `check_usb_devices` | `usb_devices_count` | USB devices |
| `check_usb_keyboard`, `check_usb_mouse` | `usb_keyboard_count`, `usb_mouse_count` | USB keyboards and mice |
| `check_usb_storage`, `check_usb_audio`, `check_usb_camera`, `check_usb_hub` | `usb_storage_count`, `usb_audio_count`, `usb_camera_count`, `usb_hub_count` | USB storage, audio, cameras, and hubs |
| `check_ps2_keyboard`, `check_ps2_mouse` | `ps2_keyboard_count`, `ps2_mouse_count` | Wired PS/2 keyboards and mice |
| `check_total_keyboard`, `check_total_mouse` | `total_keyboard_count`, `total_mouse_count` | USB and PS/2 together |
| `check_network` | `network_controllers_count` | Network controllers |
| `check_graphics` | `graphics_controllers_count` | Graphics controllers |
| `check_monitors` | `monitor_count` | Connected monitors |

All checks default to `false`. `check_usb` enumerates USB controllers and devices without selecting one count. `timeout` sets the seconds for the test case.

## Profiles

<!-- esq:suite-profiles test_peripheral_detection -->

## Run

```bash
esq run --profile profile.horizontal.system.peripheral --filter test_id=PER-USB-001
```
