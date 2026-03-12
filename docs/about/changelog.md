# Changelog

Intel® ESQ changelog and version history.

## Version 2026.1.0 - March 2026

### Changes
- New Platform Support:
  - Intel® Core™ Ultra Processors (Series 3), Products formerly Panther Lake
  - Intel® Core™ Processors (Series 2), Products formerly Bartlett Lake
  - Intel® Xeon® 6 processors, Products formerly Granite Rapids
- Enabled Verified Reference Blueprints (VRB) data on Intel® ESQ Report
- Enhanced OpenVINO™ Model Server (OVMS) with version 2025.4.1 for Gen AI workloads
- OpenVINO toolkit 2025.3.0 support for Vision AI related Test Suite
- Platform power configuration in report generation
- Configurable DL Streamer pipeline enhancements:
  - Configurable sink element support
  - Configurable consecutive timeout threshold
- New supported qualification platform detection logic
- New retail-focused pipeline - Loss Prevention VLM
- Enabled ModelScope download source support
- New Metro vertical profile with proxy workload coverage for LPR, Smart NVR, headed Visual AI, and VSaaS pipelines
- New Horizontal suites: Vision OpenVINO, System GPU OpenVINO, Media Performance, and Video Analytics
- Enabled Metro execution with automatic dependent suite execution
- AI Frequency long run test suite integration
- Generic GPU required flag support in system validator
- Enabled display mode by default for media & proxy test cases
- Updated iGPU/dGPU ID lookup table with latest hardware IDs
- Enhanced container security with cap_add parameter exposure
- Enabled new horizontal test cases for Edge Workloads and Benchmarks Pipelines
- Enhanced CPU streams handling to prevent drops due to CPU scaling logic
- Updated qualification model for scalable performance tier with DeepSeek-R1-Distill-Qwen-14B
- Increased default test execution timeout value to 640 minutes
- Fixed vision AI dGPU pipeline timeout when iGPU is disabled
- Improved cache reuse for multi-device DL Streamer analysis
- Enhanced multi-device max streams metric population for multi-socket CPUs
- Fixed dGPU performance optimization to exceed iGPU performance
- Addressed protobuf JSON recursion depth bypass vulnerability
- Fixed Pillow version security vulnerability
- Enhanced resource download failure handling and error management
- Fixed PDD model download failure and performance optimization

### Known Issues
- DL Streamer dGPU test cases use Intel® DL Streamer 2025.1.2 with OpenVINO™ 2025.2.0, while CPU and NPU tests use the newer OpenVINO™ 2025.3.0 version
- GPU utilization and package power monitoring may show inconsistent readings on Ubuntu* 24.04.3 systems with Linux* kernel 6.14
- Metro proxy pipeline test case generates excessive download progress log output
- Metro proxy pipeline intermittently report zero stream counts across different display environments
- Metro proxy pipeline and System GPU OpenVINO test cases use reference data when runtime telemetry is unavailable
- LPR proxy pipeline fails to execute on Panther Lake NPU devices
- Media Performance encode and decode test cases produce duplicate results in CSV table
- Vision AI test cases run concurrently on all devices (dGPU/iGPU/CPU/NPU), while Verified Reference data test cases run on individual components. This will be fixed in the next release.
- Gen AI test cases running on NPU with Intel® Core™ Ultra Processors (Series 2) may fail if throughput drops below the 10 token/s threshold. This will be fixed in the next release.

## Version 2025.2.0 - December 2025

### Features
- New qualification profile with Gen AI and Vision AI test cases
- New manufacturing-focused detection & classification pipelines
- New retail-focused detection pipeline
- New system memory test suites
- CSV attachment visualization and improved vertical benchmarking sections
- Support for versioned docs
- Automatically execute prerequisite profiles
- Extractor utility for Allure test results
- Update CLI run command with opt-out prompt for vertical profiles
- Enable multiple video format conversion including h265
- Enable YOLO model export with dynamic shape and kagglehub resnet support
- Enable DL Streamer pytest with configurable sync element for benchmarking analysis

## Version 2025.1.0 - October 2025

### New Features

- **Framework Revamp**: Rebuilt Intel® ESQ with new, more robust architecture
- **GenAI Testing**: New generative AI test suite for AI workload evaluation
- **VisionAI Testing**: Comprehensive computer vision testing pipeline
- **System Detection**: Automatic software configuration detection and reporting
- **Enhanced Logging**: Detailed execution logs for each test module
- **Timeout Management**: Automatic test termination when time limits are exceeded

### Notes

This major release establishes the foundation for future Intel® ESQ development with breaking changes from previous versions.