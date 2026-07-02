# Getting Started

Intel® ESQ runs on a range of supported platforms with discrete GPU and NPU accelerators. The architecture adapts to the available hardware, exercising compute resources through a structured stack of runtimes, test suites, and application-level workloads.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontSize': '14px', 'primaryColor': '#F5F5F5', 'primaryBorderColor': '#BDBDBD', 'lineColor': '#757575', 'edgeLabelBackground': '#FFFFFF'}}}%%
block
    columns 6

    A_OV["OpenVINO*"] A_DLS["DL Streamer*"] A_OVMS["OVMS*"] A_VLLM["vLLM*"] A_APP1["STREAM*"] S_OTH_APP["App-N"]

    block:T_BLK:6
        columns 7
        space:3 T_HDR["Test Cases"] space:3
    end

    S_AI["AI"] S_VSN["Vision"] S_SYS["System"] S_RT["Real-Time"] S_VRT["Vertical"] S_OTH["Test Suite-N"]

    block:S_BLK:6
        columns 7
        space:3 S_HDR["Test Suites"] space:3
    end

    block:CORE_BLK:6
        columns 5
        space:2 CORE_HDR["CLI"] space:2
    end

    block:RUN_BLK:6
        columns 5
        space:2 RUN_HDR["Runtimes"] space:2
        space RUN_DK["Docker*"] RUN_PY["Python*"] RUN_ND["Node.js*"] space
    end

    block:OS_BLK:6
        columns 5
        space:2 OS_HDR["Operating system"] space:2
        space:2 OS_UBUNTU["Ubuntu*"] space:2
    end

    classDef whiteNodeDash fill:#f0f0f0,stroke:#A0A0A0,stroke-dasharray: 4;
    classDef greenNode fill:#C9EED5,stroke:#A0A0A0,stroke-width:0.2px;
    classDef blueNode fill:#C7DAE9,stroke:#A0A0A0,stroke-width:0.2px;
    classDef blueHdrNode  fill:#C7DAE9,stroke:#C7DAE9;
    classDef blueBlkNode fill:#C7DAE9,stroke:#A0A0A0,stroke-width:0.2px;
    classDef greyHdrNode fill:#f0f0f0,stroke:#f0f0f0;
    classDef greyBlkNode fill:#f0f0f0,stroke:#A0A0A0,stroke-width:0.2px;

    class A_OV,A_DLS,A_OVMS,A_VLLM,A_APP1,A_APP2 greenNode
    class S_AI,S_VSN,S_SYS,S_RT,S_VRT blueNode
    class OS_UBUNTU,RUN_PY,RUN_DK,RUN_ND greenNode
    class CORE_HDR,S_HDR,T_HDR blueHdrNode
    class S_OTH,S_OTH_APP whiteNodeDash
    class CORE_BLK,S_BLK,T_BLK blueBlkNode
    class OS_BLK,RUN_BLK greyBlkNode
    class OS_HDR,RUN_HDR greyHdrNode
```

The following table describes each component in the stack, from the application layer down to the operating system foundation.

| Layer | Description |
|-------|-------------|
| **Applications and Components** | Workloads and services exercised by test cases — OpenVINO™ toolkit, Intel® DL Streamer*, OVMS*, vLLM*, and custom applications |
| **Test Cases** | Individual parameterized test cases within each suite, each targeting a specific workload or use case |
| **Test Suites** | Domain-specific test suites — AI, Vision, System, Real-Time, Vertical, and extensible custom suite types |
| **CLI** | Intel® ESQ command-line interface — the primary entry point for running tests, managing profiles, and viewing results |
| **Runtimes** | Docker*, Python*, and Node.js* runtime environments required for test execution and containerized workloads |
| **Operating System** | Ubuntu* as the supported base Linux* distribution |

## Next Steps

1. **[Quick Start](quick-start.md)** – Install dependencies and run all tests.


## Need Help?

If you encounter issues during setup:

1. Refer to the [Optimization](../guides/optimization.md) guide.
2. Check the [Troubleshooting](../guides/troubleshooting.md) guide.
3. Visit [GitHub* Issues](https://github.com/open-edge-platform/edge-system-qualification/issues) for community support.

---
