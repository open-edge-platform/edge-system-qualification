!!! tip "Upgrading Intel® ESQ"
    Before running a new version of Intel® ESQ, clean up any previously created `esq_data` folder to prevent leftover data from interfering with the new installation:

    ```bash
    esq clean --all
    ```

!!! tip "Verbose Output"
    Use `--verbose` to show detailed execution logs:

    ```bash
    esq --verbose run
    ```

!!! info "Driver Requirements and Hardware Changes"
    Intel® GPU and NPU tests require specific drivers. Ensure you have the latest Intel® drivers installed for your hardware configuration. If you run tests on different hardware or swap system components, clean up the cache before rerunning tests:

    ```bash
    esq --verbose clean --cache-only
    ```

!!! info "Download Source"
    Intel® ESQ uses HuggingFace* as the default download source. For ModelScope*, configure it with:

    ```bash
    export PREFER_MODELSCOPE=1
    ```
