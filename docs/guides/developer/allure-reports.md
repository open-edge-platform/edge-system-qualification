# Allure Report Customization

The project bundles a customized build of Allure3 for report generation. This page explains the customization architecture and the developer workflow for iterating on the report UI.

---

## Two-Tier Architecture

Customizations are split into two tiers to keep small core changes and large new UI components managed separately.

| Tier | Location | Contents | Format |
|------|----------|----------|--------|
| **Tier 1 — Core patch** | `src/sysagent/configs/core/patches/allure3/` | Modifications to vanilla allure3 files: `allurerc.mjs`, `packages/core*/src/`, `packages/plugin*/src/` | Unified diff (`.patch`) |
| **Tier 2 — Component overlay** | `src/sysagent/configs/core/overlay/allure3/` | Full source files for custom UI components that are new or substantially rewritten | Plain source files |

During CLI setup, the core patch is applied first, then the overlay files are copied on top. Both tiers are committed to the repository.

### Tier 2 Overlay Structure

```
src/sysagent/configs/core/overlay/allure3/
└── packages/
    └── web-awesome/
        └── src/
            ├── components/
            │   ├── Footer/          # FooterLogo.tsx, FooterVersion.tsx
            │   ├── SectionPicker/   # index.tsx
            │   ├── SectionSwitcher/ # index.tsx
            │   └── Summary/         # Custom summary section (charts, telemetry, KPIs)
            ├── locales/
            │   └── en.json          # Localisation strings
            └── stores/
                └── sections.ts      # Section registry
```

Overlay files mirror the allure3 directory tree and can be edited directly without downloading or building allure3.

---

## How the Patch Workflow Works

When the CLI runs, it downloads vanilla allure3 at a fixed tag and applies every `*.patch` file in the patches directory using `patch --backup`. The `--backup` flag creates a `*.orig` file alongside each patched file, recording the pre-patch (vanilla) content.

The `patch` mode of `scripts/allure3-dev.sh` uses those `*.orig` files to reconstruct the vanilla state inside a throwaway Git\* repository, overlays the current modified files (excluding Tier 2 overlay paths from the diff), and captures a clean unified diff with `git diff --cached`.

---

## Prerequisites

Set up automatically the first time you run `<cli> run`:

- `<data-dir>/thirdparty/allure3` — allure3 working copy
- `<data-dir>/thirdparty/node` — Node.js\* installation
- `rsync`, `patch`, `git` — system packages (`apt install rsync patch git`)

---

## Editing Source Files

**Tier 2 changes (component overlay)** — Edit directly in the overlay source directory; no allure3 build required:

```
src/sysagent/configs/core/overlay/allure3/packages/web-awesome/src/
```

**Tier 1 changes (core patch)** — Edit directly in the allure3 working copy:

```
<data-dir>/thirdparty/allure3/allurerc.mjs
<data-dir>/thirdparty/allure3/packages/core/src/
<data-dir>/thirdparty/allure3/packages/plugin-awesome/src/
<data-dir>/thirdparty/allure3/packages/plugin-log/src/
<data-dir>/thirdparty/allure3/packages/core-api/src/model.ts
```

---

## Developer Workflow

### Step 1 — Set up a one-time project

```bash
<cli> run
```

This downloads allure3, applies the core patch, copies overlay files, and builds everything into `<data-dir>/thirdparty/allure3/`.

### Step 2 — Edit and preview

Edit the relevant source files (see [Editing Source Files](#editing-source-files) above), then preview:

```bash
bash scripts/allure3-dev.sh test
```

This sets `singleFile: false` in `allurerc.mjs`, applies overlay files, rebuilds the `web-awesome` package, generates a multi-file report from the latest test results, and opens a browser. Press `Ctrl+C` to stop the web server.

**Why multi-file mode?** The custom UI components (`TelemetrySection`, `BulletChart`, `AttachmentImage`, etc.) call `fetchAttachment()` from `@allurereport/web-commons` to retrieve full attachment content at runtime — the Core Metrics JSON, System Info JSON, and `test_summary.json` that carry telemetry, KPI results, and hardware context. In multi-file mode, each attachment is a discrete file served by the dev web server, so every `fetchAttachment()` call resolves to a real HTTP request. Production builds use `singleFile: true` where allure3 handles embedding differently; test mode keeps `singleFile: false` to make attachment content inspectable and to avoid rebuilding the entire report on each UI iteration.

Repeat until the UI looks correct.

!!! note
    For changes in packages other than `web-awesome` (e.g., `core`, `plugin-awesome`, `plugin-log`), run a full rebuild first:
    ```bash
    bash scripts/allure3-dev.sh build
    bash scripts/allure3-dev.sh test
    ```

### Step 3 — Export changes

```bash
bash scripts/allure3-dev.sh patch
```

This does two things automatically:

1. Syncs any Tier 2 files modified in the allure3 working copy back to the overlay source directory.
2. Regenerates the Tier 1 core patch from the remaining differences (Tier 2 files are excluded from the diff).

### Step 4 — Remove the working copy

Remove the allure3 working copy so the next CLI run re-applies the updated patch and overlay from scratch:

```bash
# Option A: automatic
bash scripts/allure3-dev.sh patch --clean-allure

# Option B: manual
rm -rf <data-dir>/thirdparty/allure3
```

### Step 5 — Verify and commit

```bash
# Verify the updated patch applies and the report renders
<cli> run

# Run a specific profile to verify more quickly
<cli> run --profile <profile-name>
```

Commit both the updated patch file and any changed overlay files.

---

## `allure3-dev.sh` Script Reference

The script lives at `scripts/allure3-dev.sh`. All path defaults are derived from `--app-name` (defaults to `esq`).

| Option | Default | Description |
|--------|---------|-------------|
| `--app-name NAME` | `esq` | CLI data directory prefix; all paths default to `<NAME>_data/…` |
| `--allure-dir DIR` | `<app-name>_data/thirdparty/allure3` | Allure3 working copy |
| `--node-dir DIR` | `<app-name>_data/thirdparty/node` | Node.js\* installation |
| `--results-dir DIR` | `<app-name>_data/results/allure` | Allure results for the test preview |
| `--patches-dir DIR` | `src/sysagent/configs/core/patches/allure3` | Destination for the generated patch |
| `--overlay-dir DIR` | `src/sysagent/configs/core/overlay/allure3` | Component overlay source directory |
| `--patch-name NAME` | `allure3-v<version>.patch` | Filename for the generated patch |
| `--clean-allure` | — | Remove the allure3 working copy after `patch` mode completes |
| `--no-open` | — | Skip opening the browser (`test` mode only) |
| `--dry-run` | — | Print actions without executing them |

---

## Related Pages

- [Advanced Topics](advanced.md) — Custom fixtures, multi-device testing, Docker\* integration
- [Writing Tests](writing-tests.md) — `summarize_test_results` and `enable_visualizations`
