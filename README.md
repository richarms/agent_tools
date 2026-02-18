# agent_tools

agent_tools is a shell-based REPL that fronts either the OpenAI Responses API or Gemini with a narrow system prompt focused on MeerKAT infrastructure telemetry. It wraps a curated toolbelt (Grafana, Kibana, Elasticsearch, Logtrail, Mesos, HTTP helpers, and search) and enforces read-only safeguards so the agent can investigate without mutating production systems.

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` is generated from `requirements.in`:

```bash
uv pip compile requirements.in -o requirements.txt
```

Provide the backing API keys (`OPENAI_API_KEY` for OpenAI, `GEMINI_API_KEY` for Gemini) in the environment before starting the CLI.

## Usage (modular CLI)

```bash
python -m agent_tools \
  [--backend {gemini|openai}] \
  [--gemini-model GEMINI_MODEL] \
  [--openai-model OPENAI_MODEL] \
  [--model MODEL_OVERRIDE] \
  [--target-mc BASE_HOST_OR_URL] \
  [--resume SESSION_ID]
```

- `--backend` (or `AGENT_BACKEND`) selects the LLM family. Use `--model` to override the family-specific defaults in one flag.
- `--gemini-model`, `--openai-model`, `GEMINI_MODEL`, and `OPENAI_MODEL` control the concrete model names if no override is supplied.
- `--target-mc` (or `TARGET_MC`) defines the base host used to build Grafana/Kibana/Elasticsearch URLs when dedicated endpoints are not provided.
- `--resume` loads a JSON transcript that was saved automatically on the previous exit. Sessions live under the system temp dir (`agent_tools_sessions`) and are keyed by the printed hash.

The REPL prints tool and model responses prefixed with `%;`. A per-turn token report follows whenever the OpenAI backend returns usage data, and a cumulative report is shown on exit. Ctrl+C/Ctrl+D cleanly exit after saving the session context.

## Usage (single-file OpenAI runner)

```bash
python agent_tools.py \
  [--model OPENAI_MODEL] \
  [--target-mc BASE_HOST_OR_URL] \
  [--resume SESSION_ID]
```

This runner is OpenAI-only and includes the same infrastructure tools plus a `mattermost_channels_status` helper.

## Tooling

The modular CLI tools live in `agent_tools/tools.py` and are exposed to both LLM backends:

- `ping` and `http_request` provide basic network inspection.
- `google_search` uses Custom Search for live lookups when `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are configured.
- `grafana_request`, `kibana_request`, `logtrail_request`, and `elasticsearch_read` proxy Grafana, Kibana, Logtrail, and Elasticsearch APIs with read-only guards and bearer/basic auth support.
- `mesos_frameworks` summarizes active and completed Mesos frameworks, automatically scoping to the configured MC host when `MESOS_MASTER_URL` is unset.

Schemas for the OpenAI function-calling API and Gemini Tools API are generated from the same `ToolSpec` definitions, ensuring identical capabilities regardless of backend.

## Credentials and environment

- `OPENAI_API_KEY` and `GEMINI_API_KEY` are required depending on backend choice.
- `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` enable `google_search`.
- `GRAFANA_API_TOKEN` (and optional `GRAFANA_API_URL`) authorize Grafana queries.
- Kibana helpers need either `KIBANA_API_KEY` or `KIBANA_BASIC_AUTH`; `KIBANA_ALLOW_NO_AUTH=1` allows anonymous access for test stacks.
- Optional overrides: `ELASTICSEARCH_URL`, `LOGTRAIL_API_URL`, `MESOS_MASTER_URL`.
- `TARGET_MC` controls the default host that `config.mc_service_url()` uses when a per-service override is absent.
- `MATTERMOST_API_URL`, `MATTERMOST_API_ID`, and `MATTERMOST_API_TOKEN` are used by `mattermost_channels_status` in `agent_tools.py` (optional).
- `MATTERMOST_TEAM_NAME` optionally overrides the default team name (`sarao`) for Mattermost checks.

Unset options fall back to lab infrastructure defaults (`http://lab-mc.sdp.kat.ac.za`). All inputs are sanitized before serialization so saved sessions only include the data needed to recreate context.

## Sessions

Session transcripts are stripped of transient status fields, serialized to JSON, and stored under the system temp directory with a short SHA-256 identifier. Use `python -m agent_tools --resume <id>` to pick up where you left off. The saved payload also records the MC target that was active when the session ended.

## Repository layout

- `agent_tools/` contains the modular package used by `python -m agent_tools`.
- `agent_tools.py` is a single-file OpenAI runner.
- `agent_tools/dev/` contains development-only helper scripts.
- `logs/` stores ad-hoc operational notes/log captures.
