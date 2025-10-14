# Intel® Edge System Qualification (Intel® ESQ) - Repository Instructions

## Project Overview

Intel® ESQ is a dual-package Python CLI tool designed for comprehensive evaluation and qualification of Intel edge computing systems. The project provides extensible test suites to assess system capabilities across AI, media, and system performance domains.

**Repository Structure:**
- **sysagent** - Core framework providing CLI infrastructure, utilities, and plugin system
- **esq** - Domain-specific test suites for Intel edge hardware qualification (AI, media, system tests)

Both packages are interconnected but can be used independently. The main CLI entry point is `esq` command (via `sysagent.cli:main`).

## Project Architecture

### Dual-Package System

```
src/
├── sysagent/           # Core framework
│   ├── cli.py          # Main CLI entry point
│   ├── configs/        # Framework configurations
│   ├── suites/         # Core test suites (examples, system tests)
│   └── utils/          # Framework utilities
│       ├── cli/        # CLI command handlers and parsers
│       ├── config/     # Configuration management
│       ├── core/       # Core abstractions (Result, Metrics, Cache)
│       ├── infrastructure/ # Docker, Node.js setup
│       ├── logging/    # Logging configuration
│       ├── plugins/    # Pytest plugins
│       ├── reporting/  # Report generation (Allure, visualizations)
│       ├── system/     # System information gathering
│       └── testing/    # Test execution utilities
└── esq/                # ESQ-specific extensions
    ├── configs/        # ESQ configurations
    │   └── profiles/   # Test profiles (qualifications, suites, verticals)
    ├── suites/         # ESQ test suites
    │   ├── ai/         # AI tests (vision, audio, gen)
    │   ├── media/      # Media processing tests
    │   ├── system/     # System-level tests
    │   └── vertical/   # Vertical-specific tests
    └── utils/          # ESQ-specific utilities
```

### Key Components

1. **CLI System** (`src/sysagent/utils/cli/`)
   - `parsers.py` - Argument parsing
   - `handlers.py` - Command routing
   - `commands/*.py` - Individual command implementations (run, info, list, clean, deps, etc.)

2. **Configuration System** (`src/sysagent/utils/config/`)
   - `config_loader.py` - YAML profile loading
   - `config.py` - Configuration management

3. **Test Profiles** (`src/esq/configs/profiles/`)
   - **qualifications/** - Tests with KPI-based pass/fail criteria
   - **suites/** - Tests for data collection without KPI validation
   - **verticals/** - Vertical-specific tests without KPI validation

4. **Plugin System** (`src/sysagent/utils/plugins/`)
   - pytest plugins for allure integration, caching, parameterization, validation

## Development Setup

### Prerequisites
- Python 3.10+
- uv package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker Engine (for containerized tests)
- Intel drivers (for GPU/NPU tests)

### CRITICAL: Project Setup Workflow

**ALWAYS verify project setup before running ANY commands**. Never use one-liner shortcuts that assume setup is complete.

#### ✅ CORRECT Workflow - Check Setup State First

```bash
# Step 1: Check if project is set up
if [ ! -d ".venv" ] || [ ! -f ".venv/bin/activate" ]; then
    echo "Setting up project..."
    
    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    
    # Create virtual environment
    uv venv
    
    # Activate and install
    source .venv/bin/activate
    uv pip install -e .
else
    echo "Project already set up"
    # Only activate if not already in venv
    if [ -z "$VIRTUAL_ENV" ]; then
        source .venv/bin/activate
    fi
fi

# Step 2: Verify installation
esq --version

# Step 3: Run your command
esq info
```

**For Verification/Checking Only (No Actions):**
```bash
# When you just need to CHECK something (not run tests), use simple test commands:
echo "Checking Project Setup..."

# Check venv exists
if [ ! -d ".venv" ]; then
    echo "[MISSING] Virtual environment not found"
    echo "Run: uv venv && source .venv/bin/activate && uv pip install -e ."
else
    echo "[OK] Virtual environment exists"
fi

# Check if venv is activated  
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[WARNING] Virtual environment not activated"
    echo "Run: source .venv/bin/activate"
else
    echo "[OK] Virtual environment activated: $VIRTUAL_ENV"
fi

# Check esq command
if command -v esq &> /dev/null; then
    echo "[OK] esq command available"
    esq --version
else
    echo "[MISSING] esq not installed"
    echo "Run: uv pip install -e ."
fi

echo "Setup check complete"
```

#### ❌ WRONG Approach - Assuming Setup Exists

```bash
# NEVER do this - assumes .venv exists and uses shortcuts
cd /path/to/project && if [ -z "$VIRTUAL_ENV" ]; then source .venv/bin/activate; fi && esq info

# NEVER use exit in one-liner chains
if [ ! -d ".venv" ]; then echo "ERROR: No virtual environment found"; exit 1; fi && esq info

# NEVER chain multiple commands with && without proper structure
if [ ! -d ".venv" ]; then exit 1; fi && source .venv/bin/activate && esq run
```

#### Environment Setup Commands (First Time Only)

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies in editable mode - CRITICAL: Use uv pip, NOT pip
uv pip install -e .

# Verify installation
esq --version
```

### CRITICAL Rules for Command Generation

**NEVER:**
- Chain multiple commands with `&&` without verifying setup state first
- Use `pip install` - ALWAYS use `uv pip install -e .`
- Assume `.venv` directory exists without checking
- Activate venv if `$VIRTUAL_ENV` is already set (check first with `test -z "$VIRTUAL_ENV"`)
- Skip checking `esq --version` after installation
- **Use `exit` or `return` in verification commands** - verification should only echo status
- **Create one-liner commands** - ALWAYS use separate, multi-line commands
- **Mix checking and executing** - separate verification from action
- **Use `timeout` command** with esq commands - tests have built-in timeout
- **Truncate output** with `head`, `tail`, or `2>&1 |` pipes - use `-v`/`-d` flags instead
- **Use unicode symbols** in bash output (avoid ✅❌⚠️) - use [OK]/[MISSING]/[WARNING]/[ERROR]/[INFO] instead

**ALWAYS:**
- Check if `.venv` exists before trying to activate
- Use separate commands, not one-liners, for clarity and error handling
- Verify `esq` is installed with `esq --version` before running tests
- Use `uv pip install -e .` for editable installation (never `pip install`)
- **Check `$VIRTUAL_ENV` before activating** - only activate if variable is empty: `test -z "$VIRTUAL_ENV"`
- **For verification commands**: Use echo-only pattern - show status without exit/return
- **Structure commands properly**: Use proper if/then/else blocks with clear formatting
- **Run esq commands directly** - no timeout wrapper, no output truncation
- **Use `-v` or `-d` flags** for output control, check `esq_data/logs/` for complete logs
- **Use plain text markers** in bash output: [OK]/[MISSING]/[WARNING]/[ERROR]/[INFO] - NOT unicode symbols

### Common Development Workflow

**BEFORE running any commands, verify project setup:**

```bash
# === Verification Pattern (Echo-Only) ===
# When checking setup, use echo to show status - NEVER use exit or return

if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment not created"
    echo "Setup required: uv venv && source .venv/bin/activate && uv pip install -e ."
else
    echo "[OK] Virtual environment exists"
    
    # Check if activation needed
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "[INFO] Activating virtual environment..."
        source .venv/bin/activate
    else
        echo "[OK] Virtual environment already activated"
    fi
fi

# Verify esq is installed
if command -v esq &> /dev/null; then
    echo "[OK] esq command available"
    esq --version
else
    echo "[ERROR] esq not installed"
    echo "Install required: uv pip install -e ."
fi

# === Execution Pattern (Only After Verification) ===
# After verification shows [OK], run your commands

esq list
esq -v run --profile profile.suite.ai.vision
```

**For Quick Status Check (Echo-Only, No Actions):**
```bash
# When user asks to "check" or "verify" setup (not run tests):
echo "Project Status Check..."

test -d ".venv" && echo "[OK] .venv exists" || echo "[MISSING] .venv - run: uv venv"

test -n "$VIRTUAL_ENV" && echo "[OK] venv activated: $VIRTUAL_ENV" || echo "[WARNING] venv not activated - run: source .venv/bin/activate"

command -v esq &> /dev/null && echo "[OK] esq installed: $(esq --version 2>&1)" || echo "[MISSING] esq - run: uv pip install -e ."
```

**Standard workflow commands** (after setup verification):

```bash
# ALWAYS list profiles first to verify correct profile names
esq list

# Run all profiles
esq -v run

# Run specific profile (use exact name from 'esq list')
esq -d run --profile profile.suite.ai.vision

# Run specific test with filter
esq -d run --profile profile.suite.ai.vision --filter test_id=T0001

# Run without cache (force fresh execution)
esq -d run -nc --profile profile.suite.ai.vision --filter test_id=T0001

# View system information
esq info

# Clean up data directories
esq clean --all
```

**Profile Naming Convention**:
- Profile names use full dotted notation: `profile.suite.ai.vision`, `profile.suite.ai.audio`, `profile.suite.ai.gen`
- **Always run `esq list` first** to verify the exact profile name before executing tests
- Common profiles: `profile.suite.ai.vision`, `profile.suite.ai.audio`, `profile.suite.ai.gen`
- Do NOT use shortened names like `ai_vision` - these will fail

### Verbose and Debug Modes
- `-v` or `--verbose` - Show detailed execution logs
- `-d` or `--debug` - Show debug-level logs with maximum detail
- Always use `-v` or `-d` in development to verify test execution

### CRITICAL: Output and Timeout Handling

**ESQ CLI has built-in output and timeout management. DO NOT wrap commands with external tools.**

**NEVER:**
- Use `timeout` command (e.g., `timeout 300 esq run`) - ESQ has built-in timeout per test
- Truncate output with `head` (e.g., `esq run 2>&1 | head -100`)
- Truncate output with `tail` (e.g., `esq run 2>&1 | tail -50`)
- Pipe stderr to stdout and truncate (e.g., `2>&1 | head`)

**WHY:**
- ESQ tests have configurable `timeout` values in profile YAML (typically 180-600 seconds)
- External `timeout` command interferes with test cleanup and result collection
- Output truncation prevents proper error analysis and debugging
- ESQ uses `-v`/`-d` flags for verbosity control instead of piping
- Complete output is automatically saved to `<cli-name>_data/logs/` for review

**CORRECT - Run ESQ Commands Directly:**
```bash
# Let ESQ handle all output and timeout
esq -d run --profile profile.suite.ai.vision

# For specific test with debug output
esq -d run --profile profile.suite.ai.vision --filter test_id=T0001

# Check logs after test completion (log filename matches CLI name)
cat esq_data/logs/esq_run.log              # For 'esq' CLI
cat test-esq_data/logs/test-esq_run.log    # For 'test-esq' CLI
cat sysagent_data/logs/sysagent_run.log    # For 'sysagent' CLI
```

**WRONG - External Timeout/Truncation:**
```bash
# NEVER use timeout command
timeout 300 esq run --profile profile.suite.ai.vision

# NEVER truncate output
esq run 2>&1 | head -100
esq run 2>&1 | tail -50

# NEVER combine with timeout and truncation
timeout 600 esq run --profile profile.suite.ai.vision 2>&1 | head -200
```

### Profile Configuration

Profiles are YAML files that define test structure, parameters, and requirements.

#### Profile Structure

**Location**: `src/{package}/configs/profiles/{type}/`
- `qualifications/` - Tests with KPI pass/fail criteria
- `suites/` - Data collection tests without KPI validation
- `verticals/` - Industry-specific tests

**Profile Maps to Test Files**:
```
Profile YAML key → Test file name
├── suites:
│   ├── name: "ai"              → src/{package}/suites/ai/
│   │   └── sub_suites:
│   │       └── name: "vision"  → src/{package}/suites/ai/vision/
│   │           └── tests:
│   │               └── test_dlstreamer:     → test_dlstreamer.py
│   │                   └── params: [...]    → Passed to test function
```

#### How It Works

1. **File Discovery**: CLI reads YAML test keys (e.g., `test_dlstreamer`) and searches for `{key}.py` in suite path
2. **Test Routing**: Each param can specify `test:` field to target specific function within file
3. **Config Passing**: All profile params merged and passed to pytest function via `configs` fixture

**Example - Single Function Per File**:
```yaml
suites:
  - name: "ai"
    sub_suites:
      - name: "vision"
        tests:
          test_dlstreamer:  # Discovers: ai/vision/test_dlstreamer.py
            params:
              - test_id: "T0001"
                devices: ["cpu", "gpu"]
                timeout: 300
```

**Example - Multiple Functions Per File**:
```yaml
tests:
  test_cache:  # Discovers: cache/test_cache.py
    params:
      - test: "test_cache_import"     # Routes to specific function
        test_id: "UNIT-001"
      - test: "test_cache_creation"   # Different function, same file
        test_id: "UNIT-002"
```

#### Parameter Flow

```
Profile YAML → Consolidator → Pytest Plugin → Test Function
     ↓              ↓              ↓               ↓
  params       merge with     filter by      configs fixture
               defaults    function name    (all merged params)
```

Test functions receive all parameters via `configs` fixture:
```python
def test_dlstreamer(configs, ...):
    devices = configs.get("devices", [])
    timeout = configs.get("timeout", 300)
```

## Testing and Validation

### Test Output Locations

After running tests, results are generated in:
- **CLI Summary**: Terminal output
- **JSON Summary**: `<cli_data_folder>/results/core/test_summary.json`
- **Allure Report**: `<cli_data_folder>/reports/allure/index.html`

Default `<cli_data_folder>` is `esq_data/` in the current working directory.

### Test Structure

ESQ tests are pytest-based with custom fixtures following a standard 7-step pattern:
1. Extract parameters from configs
2. Validate system requirements
3. Prepare assets/dependencies
4. Execute test logic
5. Collect metrics
6. Validate against KPIs (if applicable)
7. Cache results

**For detailed test verification:** See `.github/instructions/testing-verification.instructions.md` which provides:
- Complete 6-step verification process
- Test structure best practices with code examples
- Profile and test development checklists
- Comprehensive troubleshooting guides
- Full command reference for all scenarios

## Build and Validation

### Build Commands

```bash
# Install in editable mode (development)
uv pip install -e .

# Build distribution packages
python -m build

# Install from built package
uv pip install dist/esq-*.whl
```

### Testing Commands

```bash
# Run all tests
esq run

# Run specific suite
esq run --profile profile.suite.ai.vision

# Run with pytest directly (advanced)
pytest src/esq/suites/ai/vision/test_dlstreamer.py -v
```

### Validation Steps

1. **Dependency Check**: Verify all system dependencies
   ```bash
   esq deps
   ```

2. **System Info**: Validate system meets requirements
   ```bash
   esq info
   ```

3. **Profile Listing**: Ensure profiles load correctly
   ```bash
   esq list
   ```

4. **Test Execution**: Run tests with verbose output
   ```bash
   esq -v run
   ```

5. **Coverity Scan**: Run security scanning (see instructions)
   ```bash
   ./tests/scans/coverity_scan.sh --package-dir=./src/esq
   ```

## Security Considerations

- Follow Coverity security fix patterns in `.github/instructions/coverity-security-fixes.instructions.md`
- Apply character-by-character string copying to break taint chains
- Validate all user inputs with allow-lists
- Use restrictive file permissions (0o750 or 0o770)
- Never use `shell=True` in subprocess calls

## Common Issues and Solutions

1. **Docker Permission Errors**: Add user to docker group
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **Missing Dependencies**: Run dependency installer
   ```bash
   esq deps
   ```

3. **Test Failures**: Check system requirements
   ```bash
   esq info  # Verify hardware/software requirements
   ```

4. **Stale Cache**: Clean and re-run
   ```bash
   esq clean --all
   esq run
   ```

5. **Import Errors**: Reinstall in editable mode
   ```bash
   uv pip install -e . --force-reinstall
   ```

## Key Principles

### Setup and Installation Principles

- **ALWAYS verify setup state** before running commands - never assume `.venv` exists
- **NEVER use `pip install`** - ALWAYS use `uv pip install -e .` for this project
- **NEVER use one-liner command chains** without verifying setup first
- **Check virtual environment** before activation - never activate if `$VIRTUAL_ENV` is already set
- **Use editable installation** for development: `uv pip install -e .`
- **Verify installation** with `esq --version` after setup

### Testing and Execution Principles

- **Always list profiles first** with `esq list` before running tests to ensure correct profile names
- **Use full profile names** (e.g., `profile.suite.ai.vision`) not shortened names (e.g., `ai_vision`)
- **Always use `-v` or `-d` flags** when running tests in development
- **Clean cache** before testing new versions: `esq clean --all`
- **Test both with and without cache** to ensure functionality works correctly
- **Verify test results** in JSON summary and Allure report, not just CLI output
- **Check existing tests** don't regress when making changes
- **Follow security patterns** for Coverity-identified vulnerabilities

## Detailed Instructions

For specific tasks, refer to these comprehensive guides:

### Testing and Verification
**File**: `.github/instructions/testing-verification.instructions.md`

Use this when:
- Adding or modifying tests
- Verifying test functionality
- Debugging test failures
- Creating test profiles

Contains:
- Complete 6-step verification process
- Test execution workflows with cache management
- Profile and test development checklists
- Comprehensive troubleshooting for common test issues
- Full command reference with examples

### Security Fixes
**File**: `.github/instructions/coverity-security-fixes.instructions.md`

Use this when:
- Fixing Coverity security issues
- Addressing code vulnerabilities

Contains:
- Security fix patterns for specific vulnerability types
- Verification procedures using Coverity scan
- Best practices for secure coding

## Additional Resources

- Documentation: `docs/` directory (built with MkDocs)
- Example profiles: `src/sysagent/configs/profiles/examples/`
- Test examples: `src/sysagent/suites/examples/`
