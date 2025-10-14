# Suites Profile Configurations

This directory contains profiles that are automatically executed by the default CLI `run` command. Tests run from these profiles may pass or fail based on their results, but are not subject to KPI validation.

Example:
```yml
name: "profile.suite.ai.vision"
description: "Vision AI suite profile"
params:
  labels:
    profile_display_name: "Vision AI"
    type: "suite"
```

For profiles with KPI-based validation, see the `qualifications` folder.
