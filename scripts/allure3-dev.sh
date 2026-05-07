#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Allure3 Report Customization Development Helper
#
# Three modes:
#   test   - Apply overlay files, rebuild web-awesome, generate a preview
#            report from current results, and open it in the browser.
#   patch  - Sync Tier 2 overlay files from the allure3 working copy back
#            to the source directory, then regenerate the Tier 1 core patch.
#   build  - Full rebuild of all allure3 packages (needed after changes to
#            core or plugin sources).
#
# For the full developer workflow, architecture overview, and editing guide see:
#   docs/guides/developer.md  →  "Allure3 Report Customization" section
#
# Usage:
#   bash scripts/allure3-dev.sh test   [OPTIONS]
#   bash scripts/allure3-dev.sh patch  [OPTIONS]
#   bash scripts/allure3-dev.sh build  [OPTIONS]
#
# Options:
#   --app-name    NAME    CLI data directory prefix (default: esq)
#   --allure-dir  DIR     Path to the allure3 working copy
#                         (default: <app-name>_data/thirdparty/allure3)
#   --node-dir    DIR     Path to the Node.js installation
#                         (default: <app-name>_data/thirdparty/node)
#   --results-dir DIR     Path to Allure results for the test preview
#                         (default: <app-name>_data/results/allure)
#   --patch-name  NAME    Filename for the generated patch
#                         (default: allure3-v<version>.patch)
#   --patches-dir DIR     Destination directory for the generated patch
#                         (default: src/sysagent/configs/core/patches/allure3)
#   --overlay-dir DIR     Path to the component overlay source directory
#                         (default: src/sysagent/configs/core/overlay/allure3)
#   --clean-allure        Remove the allure3 working copy after patch generation
#   --no-open             Skip opening the browser (test mode only)
#   --dry-run             Print actions without executing them

set -euo pipefail

# ---------------------------------------------------------------------------
# Script and project root
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_info()    { echo "[INFO]  $*"; }
_ok()      { echo "[OK]    $*"; }
_warn()    { echo "[WARN]  $*"; }
_error()   { echo "[ERROR] $*" >&2; }
_step()    { echo ""; echo "--- $* ---"; }
_dry_run() { echo "[DRY-RUN] Would run: $*"; }

# ---------------------------------------------------------------------------
# Default app name and path derivation
# ---------------------------------------------------------------------------
#
# Runtime data paths are derived from APP_NAME (overridable via --app-name):
#   <project-root>/<APP_NAME>_data/thirdparty/allure3  (allure3 working copy)
#   <project-root>/<APP_NAME>_data/thirdparty/node     (Node.js)
#   <project-root>/<APP_NAME>_data/results/allure      (test results for preview)
#
# Source paths are fixed relative to the project root:
#   src/sysagent/configs/core/patches/allure3   Tier 1 core patch files
#   src/sysagent/configs/core/overlay/allure3   Tier 2 component overlay files
#
# Override APP_NAME with --app-name if your CLI uses a different prefix.

APP_NAME="esq"

set_default_paths() {
    local data_dir="${PROJECT_ROOT}/${APP_NAME}_data"
    DEFAULT_ALLURE_DIR="${data_dir}/thirdparty/allure3"
    DEFAULT_NODE_DIR="${data_dir}/thirdparty/node"
    DEFAULT_RESULTS_DIR="${data_dir}/results/allure"
    DEFAULT_PATCHES_DIR="${PROJECT_ROOT}/src/sysagent/configs/core/patches/allure3"
    DEFAULT_OVERLAY_DIR="${PROJECT_ROOT}/src/sysagent/configs/core/overlay/allure3"
}

# Set initial defaults (APP_NAME=esq).  parse_args will call this again if
# --app-name overrides APP_NAME so that all DEFAULT_* vars are recomputed.
set_default_paths

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

MODE=""
ALLURE_DIR=""
NODE_DIR=""
RESULTS_DIR=""
PATCH_NAME=""
PATCHES_DIR=""
OVERLAY_DIR=""
NO_OPEN=false
DRY_RUN=false
CLEAN_ALLURE=false

usage() {
    awk '/^set -euo pipefail/{exit} /^#!/{next} {sub(/^# ?/,"",$0); print}' "${BASH_SOURCE[0]}"
    exit 0
}

parse_args() {
    if [ $# -eq 0 ]; then
        usage
    fi

    MODE="$1"
    shift

    case "${MODE}" in
        test|patch|build|help|-h|--help) ;;
        *) _error "Unknown mode: '${MODE}'. Use: test, patch, or build"; exit 1 ;;
    esac

    if [ "${MODE}" = "help" ] || [ "${MODE}" = "-h" ] || [ "${MODE}" = "--help" ]; then
        usage
    fi

    while [ $# -gt 0 ]; do
        case "$1" in
            --app-name)     APP_NAME="$2";     shift 2 ;;
            --allure-dir)   ALLURE_DIR="$2";   shift 2 ;;
            --node-dir)     NODE_DIR="$2";     shift 2 ;;
            --results-dir)  RESULTS_DIR="$2";  shift 2 ;;
            --patch-name)   PATCH_NAME="$2";   shift 2 ;;
            --patches-dir)  PATCHES_DIR="$2";  shift 2 ;;
            --overlay-dir)  OVERLAY_DIR="$2";  shift 2 ;;
            --clean-allure) CLEAN_ALLURE=true;  shift   ;;
            --no-open)      NO_OPEN=true;       shift   ;;
            --dry-run)      DRY_RUN=true;       shift   ;;
            -h|--help)      usage ;;
            *) _error "Unknown option: $1"; exit 1 ;;
        esac
    done

    # Recompute defaults now that APP_NAME may have been overridden by --app-name
    set_default_paths

    # Apply defaults for any path not explicitly set
    ALLURE_DIR="${ALLURE_DIR:-${DEFAULT_ALLURE_DIR}}"
    NODE_DIR="${NODE_DIR:-${DEFAULT_NODE_DIR}}"
    RESULTS_DIR="${RESULTS_DIR:-${DEFAULT_RESULTS_DIR}}"
    PATCHES_DIR="${PATCHES_DIR:-${DEFAULT_PATCHES_DIR}}"
    OVERLAY_DIR="${OVERLAY_DIR:-${DEFAULT_OVERLAY_DIR}}"
}

# ---------------------------------------------------------------------------
# Path / dependency verification
# ---------------------------------------------------------------------------

verify_paths() {
    _step "Verifying paths"

    local ok=true

    if [ ! -d "${ALLURE_DIR}" ]; then
        _error "Allure3 working copy not found: ${ALLURE_DIR}"
        _error "Run the CLI (e.g. '<app-name> run') once to download and set up Allure3."
        ok=false
    else
        _ok "Allure3 dir:   ${ALLURE_DIR}"
    fi

    if [ ! -d "${NODE_DIR}" ]; then
        _error "Node.js installation not found: ${NODE_DIR}"
        _error "Run the CLI (e.g. '<app-name> run') once to download Node.js."
        ok=false
    else
        _ok "Node dir:      ${NODE_DIR}"
    fi

    if [ "${ok}" = false ]; then
        exit 1
    fi
}

get_yarn_bin() {
    local node_bin_dir="${NODE_DIR}/bin"
    if [ -x "${node_bin_dir}/yarn" ]; then
        echo "${node_bin_dir}/yarn"
    elif command -v yarn &>/dev/null; then
        command -v yarn
    else
        _error "yarn not found in ${node_bin_dir} and not on PATH."
        exit 1
    fi
}

setup_node_path() {
    export PATH="${NODE_DIR}/bin:${PATH}"
}

# ---------------------------------------------------------------------------
# allurerc.mjs helpers
# ---------------------------------------------------------------------------

get_allurerc_path() {
    echo "${ALLURE_DIR}/allurerc.mjs"
}

get_single_file_value() {
    local allurerc
    allurerc="$(get_allurerc_path)"
    grep -oP "(?<=singleFile:\s)(true|false)" "${allurerc}" 2>/dev/null || echo "unknown"
}

set_single_file() {
    local value="$1"
    local allurerc
    allurerc="$(get_allurerc_path)"

    if [ ! -f "${allurerc}" ]; then
        _error "allurerc.mjs not found: ${allurerc}"
        exit 1
    fi

    local current
    current="$(get_single_file_value)"

    if [ "${current}" = "${value}" ]; then
        _info "singleFile is already set to ${value} in allurerc.mjs"
        return 0
    fi

    if [ "${DRY_RUN}" = true ]; then
        _dry_run "sed -i 's/singleFile: ${current}/singleFile: ${value}/' ${allurerc}"
        return 0
    fi

    sed -i "s/singleFile: ${current}/singleFile: ${value}/" "${allurerc}"
    _ok "Set singleFile: ${value} in allurerc.mjs"
}

# ---------------------------------------------------------------------------
# Allure3 version detection
# ---------------------------------------------------------------------------

get_allure3_version() {
    local pkg_json="${ALLURE_DIR}/package.json"
    if [ -f "${pkg_json}" ]; then
        python3 -c "import json; print(json.load(open('${pkg_json}')).get('version','unknown'))" 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

resolve_patch_name() {
    if [ -n "${PATCH_NAME}" ]; then
        echo "${PATCH_NAME}"
        return
    fi
    local version
    version="$(get_allure3_version)"
    echo "allure3-v${version}.patch"
}

# ---------------------------------------------------------------------------
# Copy allure results into the allure3 data directory for standalone preview
# ---------------------------------------------------------------------------

copy_results_to_allure3() {
    _step "Copying Allure results for preview"

    local dest_results="${ALLURE_DIR}/data/results/allure"

    if [ ! -d "${RESULTS_DIR}" ]; then
        _warn "Results directory not found: ${RESULTS_DIR}"
        _warn "No results will be available in the preview report."
        _warn "Run the CLI to generate test results first."
        return 0
    fi

    if [ "${DRY_RUN}" = true ]; then
        _dry_run "mkdir -p ${dest_results}"
        _dry_run "cp -r ${RESULTS_DIR}/. ${dest_results}/"
        return 0
    fi

    # Clean destination first so stale files from previous test runs do not
    # confuse allure3's historyId deduplication (it can pick an older result
    # over a newer one if both are present).
    rm -rf "${dest_results}"
    mkdir -p "${dest_results}"
    cp -r "${RESULTS_DIR}/." "${dest_results}/"
    _ok "Copied results from ${RESULTS_DIR}"
    _info "           to ${dest_results}"
}

# ---------------------------------------------------------------------------
# Tier 2 component overlay helpers
# ---------------------------------------------------------------------------

# Copy overlay source files into the allure3 working copy.
# Called before every build so edits to overlay files are immediately reflected.
apply_component_files() {
    if [ ! -d "${OVERLAY_DIR}" ]; then
        return 0
    fi

    local count
    count="$(find "${OVERLAY_DIR}" -type f 2>/dev/null | wc -l)"
    if [ "${count}" -eq 0 ]; then
        return 0
    fi

    _step "Applying component overlay files (Tier 2)"
    _info "Source: ${OVERLAY_DIR}"
    _info "Files:  ${count} file(s)"

    if [ "${DRY_RUN}" = true ]; then
        _dry_run "rsync -a ${OVERLAY_DIR}/ ${ALLURE_DIR}/"
        return 0
    fi

    rsync -a "${OVERLAY_DIR}/" "${ALLURE_DIR}/"
    _ok "Applied ${count} overlay file(s) to allure3 working copy"
}

# Sync component files from allure3 working copy back to the overlay source dir.
# Called during patch mode so that any edits made directly in the allure3 working
# copy are preserved in the project source tree.
sync_component_files() {
    if [ ! -d "${OVERLAY_DIR}" ]; then
        _info "Overlay directory not found; skipping component sync: ${OVERLAY_DIR}"
        return 0
    fi

    _step "Syncing component files to overlay directory (Tier 2)"
    _info "Overlay: ${OVERLAY_DIR}"

    local count=0
    local missing=0

    while IFS= read -r -d '' overlay_file; do
        local rel_path="${overlay_file#"${OVERLAY_DIR}/"}"
        local allure_file="${ALLURE_DIR}/${rel_path}"

        if [ -f "${allure_file}" ]; then
            if [ "${DRY_RUN}" = true ]; then
                _dry_run "cp ${allure_file} ${overlay_file}  # ${rel_path}"
            else
                cp "${allure_file}" "${overlay_file}"
                count=$((count + 1))
            fi
        else
            _warn "Not found in allure3 working copy: ${rel_path}"
            missing=$((missing + 1))
        fi
    done < <(find "${OVERLAY_DIR}" -type f -print0)

    if [ "${DRY_RUN}" = false ]; then
        _ok "Synced ${count} overlay file(s)"
        if [ "${missing}" -gt 0 ]; then
            _warn "${missing} file(s) not found in allure3 working copy"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Build web-awesome package (fast rebuild for UI iteration)
# ---------------------------------------------------------------------------

build_web_awesome() {
    _step "Building web-awesome package"

    local yarn_bin
    yarn_bin="$(get_yarn_bin)"
    local web_awesome_dir="${ALLURE_DIR}/packages/web-awesome"

    if [ ! -d "${web_awesome_dir}" ]; then
        _error "web-awesome package not found: ${web_awesome_dir}"
        exit 1
    fi

    if [ "${DRY_RUN}" = true ]; then
        _dry_run "cd ${web_awesome_dir} && ${yarn_bin} build"
        return 0
    fi

    _info "Running: yarn build in packages/web-awesome/"
    (cd "${web_awesome_dir}" && "${yarn_bin}" build)
    _ok "web-awesome built successfully"
}

# ---------------------------------------------------------------------------
# Full Allure3 project build (all packages)
# ---------------------------------------------------------------------------

build_allure3() {
    _step "Building full Allure3 project"

    local yarn_bin
    yarn_bin="$(get_yarn_bin)"

    if [ "${DRY_RUN}" = true ]; then
        _dry_run "cd ${ALLURE_DIR} && ${yarn_bin} build"
        return 0
    fi

    _info "Running: yarn build in $(basename "${ALLURE_DIR}")/"
    (cd "${ALLURE_DIR}" && "${yarn_bin}" build)
    _ok "Allure3 built successfully"
}

# ---------------------------------------------------------------------------
# Generate report (standalone / multi-file mode)
# ---------------------------------------------------------------------------

generate_report() {
    _step "Generating Allure report"

    local yarn_bin
    yarn_bin="$(get_yarn_bin)"
    local out_dir="${ALLURE_DIR}/out"
    local data_results="${ALLURE_DIR}/data/results/allure"

    if [ "${DRY_RUN}" = true ]; then
        _dry_run "rm -rf ${out_dir}"
        _dry_run "cd ${ALLURE_DIR} && ${yarn_bin} allure generate ${data_results}"
        return 0
    fi

    # Remove both the expected output directory and the allure3 default
    # (./allure-report) in case a previous run generated there.
    _info "Removing old output: ${out_dir}"
    rm -rf "${out_dir}"
    rm -rf "${ALLURE_DIR}/allure-report"

    _info "Running: yarn allure generate"
    (cd "${ALLURE_DIR}" && "${yarn_bin}" allure generate "${data_results}")
    _ok "Report generated in ${out_dir}"
}

# ---------------------------------------------------------------------------
# Open report in browser (web server mode)
# ---------------------------------------------------------------------------

open_report() {
    if [ "${NO_OPEN}" = true ]; then
        _info "Skipping browser open (--no-open)"
        return 0
    fi

    _step "Opening report"

    local yarn_bin
    yarn_bin="$(get_yarn_bin)"

    if [ "${DRY_RUN}" = true ]; then
        _dry_run "cd ${ALLURE_DIR} && ${yarn_bin} allure open"
        return 0
    fi

    _info "Running: yarn allure open"
    _info "Press Ctrl+C to stop the web server when done."
    (cd "${ALLURE_DIR}" && "${yarn_bin}" allure open) || true
}

# ---------------------------------------------------------------------------
# Clean allure3 working copy
# ---------------------------------------------------------------------------

clean_allure_install() {
    _step "Removing allure3 working copy"

    _info "Directory: ${ALLURE_DIR}"
    _info "On the next run the CLI will re-download, apply the updated patch,"
    _info "and rebuild allure3 automatically."

    if [ "${DRY_RUN}" = true ]; then
        _dry_run "rm -rf ${ALLURE_DIR}"
        return 0
    fi

    rm -rf "${ALLURE_DIR}"
    _ok "Removed: ${ALLURE_DIR}"
}

# ---------------------------------------------------------------------------
# Patch generation helpers
# ---------------------------------------------------------------------------

count_orig_files() {
    find "${ALLURE_DIR}" -name "*.orig" -not -path "${ALLURE_DIR}/.git/*" | wc -l
}

# Warn about stray *.patch files in the allure3 working copy.
# These originate from the old manual workflow (git diff --cached > *.patch
# run from within the allure3 repo directory).  They must NOT be included
# in the generated patch.
warn_stray_patch_files() {
    local stray
    stray="$(find "${ALLURE_DIR}" -maxdepth 1 -name "*.patch" 2>/dev/null || true)"

    if [ -n "${stray}" ]; then
        _warn "Stray *.patch file(s) found in the allure3 working copy:"
        printf '%s\n' "${stray}" | while IFS= read -r line; do echo "  ${line}"; done
        _warn "These are left over from the old manual workflow and should be"
        _warn "deleted.  They will be excluded from the generated patch."
        if [ "${DRY_RUN}" = false ]; then
            find "${ALLURE_DIR}" -maxdepth 1 -name "*.patch" -delete
            _ok "Stray patch file(s) removed."
        fi
    fi
}

# Reconstruct the vanilla allure3 state inside TMPDIR using .orig backup files.
# After this call TMPDIR holds the pre-patch (vanilla) source tree.
restore_vanilla_to_tmpdir() {
    local tmpdir="$1"

    # Start with a full copy of the current (patched) working copy
    cp -r "${ALLURE_DIR}/." "${tmpdir}/"

    # Walk every .orig file and restore the pre-patch content
    while IFS= read -r -d '' orig; do
        local rel_orig="${orig#"${ALLURE_DIR}/"}"
        local rel_target="${rel_orig%.orig}"
        local tmp_orig="${tmpdir}/${rel_orig}"
        local tmp_target="${tmpdir}/${rel_target}"

        if [ -s "${tmp_orig}" ]; then
            # Non-empty .orig: file was modified by the patch; restore original
            cp "${tmp_orig}" "${tmp_target}"
        else
            # Empty .orig: file was newly created by the patch; remove it
            rm -f "${tmp_target}"
        fi

        # Remove the .orig metadata file from the vanilla copy
        rm -f "${tmp_orig}"
    done < <(find "${ALLURE_DIR}" -name "*.orig" -not -path "${ALLURE_DIR}/.git/*" -print0)
}

# Generate the patch using a temporary git repository so that the output is a
# clean unified-diff compatible with "patch --strip=1".
generate_patch_via_tmpgit() {
    local patch_dest="$1"

    local tmpdir
    tmpdir="$(mktemp -d)"
    # SC2064: intentional early expansion to capture tmpdir value in trap
    # shellcheck disable=SC2064
    trap "rm -rf '${tmpdir}'" EXIT

    _info "Setting up temporary git repo to reconstruct vanilla state..."

    # Stage 1: Vanilla (pre-patch) state
    restore_vanilla_to_tmpdir "${tmpdir}"

    (
        cd "${tmpdir}"
        git init -b main -q
        git config user.email "allure3-dev@local"
        git config user.name "Allure3 Dev Script"

        # Exclude build artefacts and stray files so only sources are tracked
        cat > .git/info/exclude << 'GITIGNORE'
node_modules/
.yarn/cache/
.yarn/unplugged/
.yarn/install-state.gz
.pnp.cjs
.pnp.loader.mjs
dist/
out/
data/
*.tgz
*.tar.gz
*.patch
.allure/
GITIGNORE

        git add -A
        git commit -q -m "vanilla-allure3-base"
    )

    # Stage 2: Overlay the current (patched + developer-modified) state,
    # excluding build artefacts, .orig backup files, and stray *.patch files
    rsync -a \
        --exclude=".git/" \
        --exclude="*.orig" \
        --exclude="*.patch" \
        --exclude="node_modules/" \
        --exclude=".yarn/cache/" \
        --exclude=".yarn/unplugged/" \
        --exclude=".yarn/install-state.gz" \
        --exclude=".pnp.cjs" \
        --exclude=".pnp.loader.mjs" \
        --exclude="dist/" \
        --exclude="out/" \
        --exclude="data/" \
        "${ALLURE_DIR}/" "${tmpdir}/"

    # Stage 2b: Exclude Tier 2 overlay files from the diff.
    # These are managed as full source files in the overlay directory and must
    # not appear in the core patch.  For each overlay file:
    #   - If it existed in vanilla (committed), restore it to the vanilla version.
    #   - If it is new (not in vanilla), remove it from the tmpdir entirely.
    if [ -d "${OVERLAY_DIR}" ]; then
        while IFS= read -r -d '' overlay_file; do
            local rel_path="${overlay_file#"${OVERLAY_DIR}/"}"
            if (cd "${tmpdir}" && git ls-files --error-unmatch "${rel_path}" >/dev/null 2>&1); then
                if ! (cd "${tmpdir}" && git checkout HEAD -- "${rel_path}" 2>/dev/null); then true; fi
            else
                rm -f "${tmpdir}/${rel_path}"
            fi
        done < <(find "${OVERLAY_DIR}" -type f -print0)
    fi

    # Stage 3: Stage everything (so new files are included) and capture diff
    local diff_stat
    if ! (cd "${tmpdir}" && git add -A); then true; fi
    diff_stat="$(cd "${tmpdir}" && git diff --cached --stat 2>/dev/null)" || true

    if [ -z "${diff_stat}" ]; then
        _warn "No differences found between vanilla allure3 and current state."
        _warn "The patch file was not written."
        rm -rf "${tmpdir}"
        trap - EXIT
        return 1
    fi

    _info "Changes captured in patch:"
    printf '%s\n' "${diff_stat}" | while IFS= read -r line; do echo "  ${line}"; done

    # Write the patch (git diff format, compatible with "patch --strip=1")
    (cd "${tmpdir}" && git diff --cached) > "${patch_dest}"

    rm -rf "${tmpdir}"
    trap - EXIT
    return 0
}

# Main patch generation orchestrator
generate_patch() {
    _step "Generating patch"

    if [ ! -d "${PATCHES_DIR}" ]; then
        if [ "${DRY_RUN}" = true ]; then
            _dry_run "mkdir -p ${PATCHES_DIR}"
        else
            mkdir -p "${PATCHES_DIR}"
            _ok "Created patches directory: ${PATCHES_DIR}"
        fi
    fi

    local patch_name
    patch_name="$(resolve_patch_name)"
    local patch_dest="${PATCHES_DIR}/${patch_name}"

    _info "Patch name:    ${patch_name}"
    _info "Destination:   ${patch_dest}"

    # Warn about and remove stray *.patch files from the working copy
    warn_stray_patch_files

    # -----------------------------------------------------------------------
    # Strategy 1: allure3 dir is a git repo (standalone developer clone)
    # -----------------------------------------------------------------------
    if [ -d "${ALLURE_DIR}/.git" ]; then
        _info "Allure3 dir is a git repository; using git diff."

        local changed_files
        changed_files="$(cd "${ALLURE_DIR}" && git status --short | grep -c -v '^?')"

        if [ "${changed_files}" -eq 0 ]; then
            _warn "No uncommitted changes found in ${ALLURE_DIR}."
            _warn "Nothing to patch."
            return 0
        fi

        _info "Files with changes (${changed_files}):"
        (cd "${ALLURE_DIR}" && git status --short | grep -v '^?' | sed 's/^/  /')

        if [ "${DRY_RUN}" = true ]; then
            _dry_run "cd ${ALLURE_DIR} && git add -A"
            _dry_run "cd ${ALLURE_DIR} && git restore --staged '**/*.patch' (exclude stray patch files)"
            _dry_run "cd ${ALLURE_DIR} && git diff --cached > ${patch_dest}"
            _dry_run "cd ${ALLURE_DIR} && git restore --staged ."
            return 0
        fi

        (cd "${ALLURE_DIR}" && git add -A)
        # Unstage stray *.patch files and all Tier 2 overlay files
        if ! (cd "${ALLURE_DIR}" && git restore --staged '**/*.patch' 2>/dev/null); then true; fi
        if [ -d "${OVERLAY_DIR}" ]; then
            while IFS= read -r -d '' overlay_file; do
                local rel_path="${overlay_file#"${OVERLAY_DIR}/"}"
                if ! (cd "${ALLURE_DIR}" && git restore --staged "${rel_path}" 2>/dev/null); then true; fi
            done < <(find "${OVERLAY_DIR}" -type f -print0)
        fi
        _ok "Staged all changes (Tier 2 overlay files and stray *.patch files excluded)"

        (cd "${ALLURE_DIR}" && git diff --cached) > "${patch_dest}"

        (cd "${ALLURE_DIR}" && git restore --staged .)
        _ok "Changes unstaged (working tree unchanged)"

    # -----------------------------------------------------------------------
    # Strategy 2: CLI-managed working copy with .orig backup files
    # -----------------------------------------------------------------------
    elif [ "$(count_orig_files)" -gt 0 ]; then
        local orig_count
        orig_count="$(count_orig_files)"
        _info "Found ${orig_count} .orig backup file(s)."
        _info "Reconstructing vanilla allure3 state for diff generation..."

        if [ "${DRY_RUN}" = true ]; then
            _dry_run "mktemp -d  # create temporary git repo"
            _dry_run "restore vanilla state from .orig files"
            _dry_run "rsync patched state into temp repo (*.patch files excluded)"
            _dry_run "git diff --cached > ${patch_dest}"
            return 0
        fi

        if ! command -v rsync &>/dev/null; then
            _error "rsync is required for patch generation but was not found."
            _error "Install it with:  sudo apt-get install rsync"
            exit 1
        fi

        if ! generate_patch_via_tmpgit "${patch_dest}"; then
            _warn "Patch generation produced no output."
            return 0
        fi

    # -----------------------------------------------------------------------
    # Strategy 3: No .orig files, not a git repo — cannot proceed
    # -----------------------------------------------------------------------
    else
        _error "Cannot generate a patch:"
        _error "  - ${ALLURE_DIR} is not a Git repository, AND"
        _error "  - no *.orig backup files were found."
        _error ""
        _error "Possible fixes:"
        _error "  1. Remove ${ALLURE_DIR} and run the CLI (e.g. '<app-name> run')"
        _error "     to re-download and re-apply patches (patch --backup creates"
        _error "     the required .orig files)."
        _error "  2. Point to a standalone allure3 git clone:"
        _error "     bash scripts/allure3-dev.sh patch --allure-dir /path/to/allure3"
        exit 1
    fi

    if [ -f "${patch_dest}" ]; then
        local patch_lines
        patch_lines="$(wc -l < "${patch_dest}")"
        _ok "Patch written to: ${patch_dest}"
        _info "Patch size: ${patch_lines} lines"
    fi
}

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

cmd_test() {
    _info "Mode: TEST  (build web-awesome + generate preview report)"
    _info "Allure3 dir:   ${ALLURE_DIR}"
    _info "Results dir:   ${RESULTS_DIR}"
    _info "Overlay dir:   ${OVERLAY_DIR}"

    verify_paths
    setup_node_path

    # Multi-file mode: allure3 serves report as a web site from out/ directory.
    # This lets you inspect all generated assets and is faster to iterate on
    # than building a single-file HTML every time.
    set_single_file false

    # Copy Tier 2 overlay files into the allure3 working copy before building.
    # This ensures any edits made directly to the overlay source directory are
    # reflected in the preview without an extra manual copy step.
    apply_component_files

    copy_results_to_allure3
    build_web_awesome
    generate_report
    open_report

    _step "Done"
    _ok "Test preview complete."
    _info ""
    _info "To edit Tier 2 component files (recommended):"
    _info "  ${OVERLAY_DIR}/packages/web-awesome/src/"
    _info "  Then re-run 'test' — overlay files are copied automatically."
    _info ""
    _info "To edit core/plugin files (Tier 1):"
    _info "  ${ALLURE_DIR}/packages/{core,plugin-awesome,plugin-log}/src/"
    _info "  Run 'build' then 'test' after changes."
    _info ""
    _info "When satisfied, export your changes:"
    _info "  bash scripts/allure3-dev.sh patch"
    _info "  bash scripts/allure3-dev.sh patch --clean-allure  (also removes allure3 working copy)"
}

cmd_build() {
    _info "Mode: BUILD  (full Allure3 project rebuild)"
    _info "Allure3 dir: ${ALLURE_DIR}"
    _info "Overlay dir: ${OVERLAY_DIR}"

    verify_paths
    setup_node_path

    # Apply overlay files before a full rebuild so all component sources are current.
    apply_component_files

    build_allure3

    _step "Done"
    _ok "Allure3 full build complete."
    _info "To preview the report, run:"
    _info "  bash scripts/allure3-dev.sh test"
}

cmd_patch() {
    local patch_name
    patch_name="$(resolve_patch_name)"

    _info "Mode: PATCH  (sync overlay files + regenerate core patch)"
    _info "Allure3 dir:   ${ALLURE_DIR}"
    _info "Patches dir:   ${PATCHES_DIR}"
    _info "Overlay dir:   ${OVERLAY_DIR}"
    _info "Patch file:    ${patch_name}"
    [ "${CLEAN_ALLURE}" = true ] && _info "Clean allure:  YES (will remove working copy after patch)"
    _info "App name:      ${APP_NAME}"

    verify_paths

    # Ensure singleFile: true so the production (single-file HTML) config is
    # captured in the patch rather than the development multi-file setting.
    set_single_file true

    # Sync Tier 2 overlay files from allure3 working copy back to the source
    # directory.  This captures any edits made directly in the working copy.
    sync_component_files

    # Regenerate the Tier 1 core patch (overlay files are excluded from the diff).
    generate_patch

    # Optionally remove the allure3 working copy after patching
    if [ "${CLEAN_ALLURE}" = true ]; then
        clean_allure_install
    fi

    local patch_dest="${PATCHES_DIR}/${patch_name}"

    _step "Done"
    _ok "Patch generation complete."
    echo ""
    _info "NEXT STEPS:"
    _info "  1. Review the core patch:"
    _info "       cat ${patch_dest}"
    _info "  2. Review updated overlay files:"
    _info "       ls ${OVERLAY_DIR}/packages/web-awesome/src/"
    if [ "${CLEAN_ALLURE}" = false ]; then
        _info "  3. Remove the allure3 working copy so the next run"
        _info "     re-downloads, applies the updated patch + overlay, and rebuilds:"
        _info "       rm -rf ${ALLURE_DIR}"
        _info "     Or re-run with --clean-allure to do this automatically:"
        _info "       bash scripts/allure3-dev.sh patch --clean-allure"
        _info "  4. Run the CLI to verify the patch applies correctly:"
        _info "       ${APP_NAME} run"
        _info "  5. Commit both the patch file and changed overlay files."
    else
        _info "  3. Run the CLI to verify the patch applies correctly:"
        _info "       ${APP_NAME} run"
        _info "  4. Commit both the patch file and changed overlay files."
    fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

main() {
    parse_args "$@"

    case "${MODE}" in
        test)  cmd_test  ;;
        patch) cmd_patch ;;
        build) cmd_build ;;
        *)
            _error "Unhandled mode: ${MODE}"
            exit 1
            ;;
    esac
}

main "$@"
