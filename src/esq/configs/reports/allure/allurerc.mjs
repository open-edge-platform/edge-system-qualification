// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from "allure";

export default defineConfig({
  name: "Intel® Edge System Qualification Report",
  plugins: {
    awesome: {
      options: {
        allureVersion: "0.0.0",
        singleFile: true,
        reportLanguage: "en",
        sections: ["summary"],
        defaultSection: "summary",
      },
    },
  },
});