# Network Connectivity

Detects the network interfaces and counts how many are connected (up with a valid IP address), for all interfaces or only wired or wireless ones.

| | |
|---|---|
| **Test suite** | `test_network_connectivity` (`src/esq/suites/system/network/test_network_connectivity.py`) |
| **Integration type** | [Type 1: generic suite](../../guides/developer/integration-types.md#type-1-generic-suite-with-configurable-ingredients). ESQ checks the system directly, and the profile selects the checks. |
| **Key metric** | `total_connected`, `wired_interfaces_connected`, or `wireless_interfaces_connected` (count), depending on the check |
| **Devices** | Network interfaces |

## How It Works

1. List the network interfaces and their type (wired, wireless, other).
2. Check which ones are up and have a valid IP address.
3. Report the totals and connected counts of the checked type.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `check_all` | `false` | Count all connected interfaces. Key metric `total_connected`. |
| `check_wired` | `false` | Count wired interfaces. Key metric `wired_interfaces_connected`. |
| `check_wireless` | `false` | Count wireless interfaces. Key metric `wireless_interfaces_connected`. |
| `timeout` | per profile | Seconds for the test case |

## Profiles

<!-- esq:suite-profiles test_network_connectivity -->

## Run

```bash
esq run --profile profile.horizontal.system.network --filter test_id=NET-CON-001
```
