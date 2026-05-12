# Troubleshooting Guide

This guide covers common issues you may encounter when using Intel® ESQ and provides solutions for each.

---

## 1. Idle memory retention after test execution

After test execution, idle memory usage may increase due to accumulated slab and filesystem cache. This can leave insufficient memory for subsequent workloads.

**Solution:** Force a filesystem cache cleanup by dropping dentries, inodes, and page cache:

```bash

echo 2 | sudo tee /proc/sys/vm/drop_caches && sleep 2 && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

---

## 2. Docker* permission denied

When running containerized tests, you may see an error such as `permission denied while trying to connect to the Docker daemon socket`.

**Solution:** Add your user to the `docker` group and refresh your session:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Verify access by running:

```bash
docker ps
```

If the issue persists, log out and log back in or reboot the system.

---

## 3. Test shows as skipped

Tests may appear as skipped for one of the following reasons:

- A cached result already exists for the same parameters.
- The system does not meet the test's hardware or software requirements.
- Required devices (GPU, NPU) are not available.

**Solution:**

- Run without cache to force a fresh execution:

    ```bash
    esq -d run -nc --profile <profile_name>
    ```

- Check system requirements:

    ```bash
    esq info
    ```

- Review debug output for device availability messages:

    ```bash
    esq -d run --profile <profile_name>
    ```

---

## 4. GPU not detected or inaccessible

Tests that require GPU access may fail with errors like `No available devices found for configured device category` or `Required GPU hardware not available`.

**Possible causes:**

- Intel GPU drivers are not installed.
- The GPU hardware is not present on the system.
- The user lacks permissions to access `/dev/dri` or `/dev/accel`.

**Solution:**

1. Verify GPU hardware is present:

    ```bash
    lspci | grep -Ei "DISPLAY|VGA"
    ```

2. Install or update Intel GPU drivers for your platform.

3. Confirm device access:

    ```bash
    ls -la /dev/dri/
    ```

4. Run `esq info` to verify device detection.

---

## 5. NPU not detected

NPU-dependent tests may fail if the NPU driver is missing or not loaded.

**Solution:**

1. Install or update Intel NPU drivers.

2. Verify the NPU is detected:

    ```bash
    ls /sys/class/accel/accel*
    ```

3. Confirm the driver in use is `intel_vpu`.

---

## 6. Profile not found

Using a shortened or incorrect profile name (for example, `ai_vision`) causes a profile-not-found error.

**Solution:** Always use the full dotted profile name. Run `esq list` to see available profiles:

```bash
esq list
```

Profile names follow the format `profile.suite.ai.vision`, `profile.suite.ai.gen`, and so on.

---

## 7. Test fails with no clear error

When a test fails but the CLI output does not show a clear root cause:

1. Run the test with debug-level logging:

    ```bash
    esq -d run --profile <profile_name>
    ```

2. Check the log files:

    ```bash
    cat esq_data/logs/esq_run.log
    ```

3. Review the Allure* report for detailed attachments and execution timeline:

    ```
    esq_data/reports/allure/index.html
    ```

4. Verify that asset preparation completed without errors.

5. For containerized tests, check Docker* container logs.

---

## 8. Stale cache causing unexpected results

Cached results from previous runs may cause tests to return outdated data, especially after code or configuration changes.

**Solution:**

- Clean the cache:

    ```bash
    esq clean --cache
    ```

- Clean all data, including results and reports:

    ```bash
    esq clean --all
    ```

- Run a specific test without cache:

    ```bash
    esq -d run -nc --profile <profile_name>
    ```

---

## 9. RAPL power monitoring not working

Platform power monitoring through RAPL (Running Average Power Limit) may fail if device permissions are not configured.

**Solution:** Run the system setup script:

```bash
sudo bash -c "$(wget -qLO - https://raw.githubusercontent.com/open-edge-platform/edge-system-qualification/refs/heads/main/scripts/system-setup.sh)"
```

!!! note
    RAPL is not supported on all platforms. Check your hardware documentation for compatibility.

---

## 10. Memory DIMM information unavailable

The `esq info` command may fail to retrieve memory DIMM details if `dmidecode` lacks the required capabilities.

**Solution:** Install the necessary packages and set the required capability with system setup script:

```bash
sudo bash -c "$(wget -qLO - https://raw.githubusercontent.com/open-edge-platform/edge-system-qualification/refs/heads/main/scripts/system-setup.sh)"
```

---



