# Chat Q&A Core

Runs the Chat Question-and-Answer Core sample application of Open Edge Platform (RAG over a document corpus), ingests documents, sends questions, and measures query latency.

| | |
|---|---|
| **Test suite** | `test_chatqna_core` (`src/esq/suites/ai/gen/test_chatqna_core.py`) |
| **Integration type** | [Type 2: reproduce the original flow](../../guides/developer/integration-types.md#type-2-reproduce-the-original-flow). It runs the application's own Docker Compose stack and configuration from `open-edge-platform/edge-ai-libraries`, pinned to a commit. |
| **Key metric** | `p95_query_latency_ms` (ms) |
| **Devices** | CPU and GPU, selected by the compose profile and backend service |

## How It Works

1. Download the application's compose file, NGINX template, and model configuration from `edge-ai-libraries` at `assets_repo_ref`, and bind every published port to `127.0.0.1`.
2. Start the compose stack with the selected backend (`compose_profile`, `backend_service`) and wait until it is healthy.
3. Ingest the documents of `corpus_dir`, then send the prompts of `prompt_file` (up to `prompt_limit`).
4. Report startup time, ingestion time, p50 and p95 query latency, time to first token, and the query success rate. A `scenario_matrix` entry runs several backends in one test case and reports them together.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `assets_repo_ref` | `main` | `edge-ai-libraries` commit of the downloaded application files. The profile pins the OEP 2026.2 commit. |
| `backend_runtime`, `compute_device` | `openvino`, `cpu` | Model backend (`openvino` or `ollama`) and device |
| `compose_profile`, `backend_service` | `OPENVINO`, `chatqna-core-ov-cpu` | Compose profile and backend service to start |
| `backend_tag`, `ui_tag`, `registry` | `core_2026.2.0`, `core_2026.2.0`, `intel/` | Application image tags and registry |
| `include_ui`, `pull_images` | `false`, `true` | Start the UI service, pull images before starting |
| `compose_file`, `model_config_path`, `corpus_dir`, `prompt_file` | downloaded defaults | Override the application files, documents, and prompts |
| `prompt_limit`, `warmup_query`, `keep_documents_after_run` | `5`, unset, `false` | Prompts sent, optional warm-up question, keep ingested documents |
| `scenario_matrix` | `[]` | Several backend scenarios, each with its own `test_id` and overrides, run in one test case |
| `startup_timeout_seconds`, `startup_attempts`, `startup_poll_interval_seconds` | `2700`, `2`, `15` | Waiting for the stack to become healthy |
| `request_timeout_seconds`, `timeout` | `180`, `1800` | Per-request and per-test time limits in seconds |
| `ref_p50_query_latency_ms`, `ref_p95_query_latency_ms`, `ref_ttft_ms` | `-1.0` | Reference values shown with the results (`-1` means none) |
| `preserve_environment`, `compose_project_name` | `false`, `esq-chatqna-core` | Keep the stack after the test, compose project name |

## Profiles

<!-- esq:suite-profiles test_chatqna_core -->

## Run

```bash
esq run --profile profile.horizontal.ai.gen-chatqna-core --filter test_id=GEN-CHAT-001
```
