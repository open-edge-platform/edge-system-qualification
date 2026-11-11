# Horizontals Profile Configurations

This directory contains profiles that are automatically executed by the default CLI `run` command. Tests run from these profiles may pass or fail based on their results, but are not subject to KPI validation.

Example:
```yml
name: "profile.horizontal.ai.vision"
description: "Vision AI horizontal profile"
params:
  labels:
    profile_group_name: "Vision AI"
    profile_display_name: "Vision AI - OpenVINO Benchmark"
    type: "horizontal"
```

For profiles with KPI-based validation, see the `qualifications` folder. 