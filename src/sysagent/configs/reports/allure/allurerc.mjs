// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from "allure";

export default defineConfig({
  name: "Report",
  plugins: {
    awesome: {
      options: {
        allureVersion: "0.0.0",
        singleFile: true,
        reportLanguage: "en",
        defaultSection: "report",
      },
    },
  },
});
