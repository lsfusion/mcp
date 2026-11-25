
# lsfusion-mcp

An extensible MCP server hosting multiple tools for lsFusion development. Ships with RAG search tools, syntax validation, and mandatory guidance.

Transports:
- **STDIO** for local development / desktop MCP clients.
- **Streamable HTTP** for production via Uvicorn (mounted at `/mcp`).

## Core Tools
- `lsfusion_get_guidance(rules?: string, brief?: string)`: Read ONE guidance article whole. With no arguments, the top article of each branch — base material plus the complete map of that branch. With a name, that area's article entire: no search, no ranking, no excerpt.
- `lsfusion_retrieve_docs(query: string | string[], type?: "language" | "paradigm" | "how-to")`: Official documentation search (English content only). All three reference branches are searched when `type` is omitted. `brief` and `rules` are still accepted for compatibility but are no longer part of the corpus — those two branches are addressed by name through `lsfusion_get_guidance`, and passing one here returns a pointer to that tool.
- `lsfusion_report_feedback(report)`: Submit one anonymous, depersonalized reinforcement-quality signal (the feedback-loop sink) — classified by `signal_type` (doc gap, expectation-mismatch, unclear `eval` error, missing capability, RAG miss, other). The agent calls it — per the `get_guidance` workflow rule and only with user consent — when a task hit action-affecting friction. Server-side: anti-abuse caps, best-effort redaction, server-computed `dedup_fingerprint`, then each report is stored as its own JSON file in `<REPORTS_DIR>/inbox/` (see [Feedback reports](#feedback-reports)). `status` is `recorded` only once that file is written; with no reports dir, or a failed write, it is `rejected` with `detail` "the server could not store the report". Gated by `FEEDBACK_ENABLED`. Returns `{report_id, status, dedup_fingerprint}`. See `MCP-FEEDBACK-PLAN.md`.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-... RAG_SNAPSHOT_PATH=./corpus.npz  # built by tools/rag_build_snapshot.py

# STDIO transport
python server.py stdio

# HTTP transport
python server.py http --host 0.0.0.0 --port 8000
```

### Claude Desktop / MCP Inspector (STDIO)
```bash
mcp install server.py
mcp dev server.py
```

## Tests

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
pytest
```

No OpenAI key is needed: the root `conftest.py` sets a dummy one, and no test calls the API. CI and the image use Python 3.12.

- **Sibling checks.** Tests marked `siblings` read the repositories mcp is mirrored into: the drift guard (`tools/tests/test_proxy_schema_drift.py`) reads the platform's `MCPDispatcher.java` and guidance docs (`docs/rules`, `docs/brief`) and the plugin's `McpBaseService.java` / `McpToolset.kt`; the chunker test reads real platform docs.
- **Released plugin.** The plugin's master is what its next release will ship, but mcp deploys on every commit and users keep the plugin they have. The sibling checks also marked `released` hold the server to what a released plugin depends on: the tools it proxies still exist, the server takes the arguments it sends and sends nothing its result schemas reject. Point `LSF_PLUGIN_ROOT` at a checkout of a release tag (`v<version>`, set when the plugin is published) and run `pytest -m released`.
- **Roots.** By default they look for `platform/` and `plugin-idea/` (or `plugin_idea/`) next to mcp, as in the super-workspace. `LSF_PLATFORM_ROOT` / `LSF_PLUGIN_ROOT` point them elsewhere, e.g. at a worktree with a mirroring change. A sibling that is not there skips its checks; under a root set explicitly, a missing file fails. `--forbid-skips` turns every skip into a failure.
- **CI** (`.github/workflows/docker.yml`). On every push, job `test` runs `pytest -m "not siblings" --forbid-skips`, and job `siblings` checks out the master of platform and plugin-idea (only the files the checks read), sets both roots and runs `pytest -m siblings --forbid-skips`. Job `released-plugin` checks out the newest `v*` tag of plugin-idea and runs `pytest -m released --forbid-skips` against it. The image is built only after all three pass; on `main` it is then pushed and `deployMcp` redeploys it. Daily and on demand (Run workflow) only the test jobs run. To deploy while the siblings disagree, drop `siblings` or `released-plugin` from `needs` of `build`.
- **Changes a released plugin cannot take.** Prefer making the server take both the old and the new form, so the change deploys at once and the plugin moves over in its next release. Mark what stays only for the old form with `COMPAT(plugin <= <version>)` in a comment, so it can be found and removed once users have moved on. A change that has to wait for a plugin release stays on its branch, where `released-plugin` is red until that release is out. A new check whose rule the current release breaks gets the `released` mark only once a release passes it.

## Adding new tools
Create a new module under `tools/` and register it with `@mcp.tool()` in `server.py` (or build an auto-discovery
if you prefer). Keep tool signatures simple and JSON-serializable.

## Contract / output for `lsfusion_retrieve_docs`
Returns an array of objects:
```json
{
  "docs": [
    { "source": "documentation-language", "text": "....", "score": 0.73 },
    { "source": "documentation-paradigm", "text": "....", "score": 0.69 }
  ]
}
```
Ranked by `score` — the similarity of the chunk to the query. Structured
output is enabled.

`query` accepts a list of distinct queries when the caller needs several
independent things at once (on real traffic 85% of call bursts carry two or
more topics). The batch is one embeddings round trip instead of several, and
shares ONE budget (`RESULT_MAX_CHARS` characters of text, at most
`BATCH_MAX_QUERIES` queries) rather than multiplying it, so it never costs
more context than the separate calls it replaces. `type` and `exclude_ids`
apply to every query in the batch. A chunk answering two queries is returned
once, credited to the query it scored higher for, and every chunk carries the
`query` it is credited to (`null` when only one was submitted). The per-query
share is a ceiling, not a guarantee: a query whose best chunks all belong to a
neighbour returns fewer.

## Environment variables
- `OPENAI_API_KEY` — OpenAI API key (required: `lsfusion_retrieve_docs` embeds each query with it).
- `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` — the model queries are embedded with and its vector size; both must match the snapshot, which is refused otherwise (default `text-embedding-3-large` / 3072).
- `MCP_BUILD_SHA` — the build identifier stamped into every event log line and report as `server_version`: the git commit the image was built from (CI passes `github.sha` as the `GIT_SHA` build arg). A build without it says `unknown`.
- `LOG_DIR` — directory for dated JSONL event files. Empty (default) ⇒ event logs go to **stderr only**. Set it to a writable (uid 10001) bind-mount to also append `<stream>-YYYYMMDD.jsonl` (`retrieval`, `reports`; Phase A2). It is also the default parent of `REPORTS_DIR`.
- `REPORTS_DIR` — where `lsfusion_report_feedback` stores reports, one file each in `<REPORTS_DIR>/inbox/`. Default `<LOG_DIR>/reports`, so the `LOG_DIR` bind mount holds them; empty when neither is set, and then every report is refused.
- `FEEDBACK_ENABLED` — `0` turns `lsfusion_report_feedback` off (it answers `disabled` and stores nothing). Default on.
- `RAG_SNAPSHOT_PATH` — the corpus snapshot `lsfusion_retrieve_docs` searches in-process (default `/data/snapshot/corpus.npz`). Built by `tools/rag_build_snapshot.py` (the `ragBuildSnapshot` Jenkins step); without it the tool answers with an error.
- `SNAPSHOT_MAX_AGE_DAYS` — how old a snapshot may get before every load warns (default 7; a warning, never a refusal).
- `RESULT_MAX_CHARS` — the most chunk text one call returns, in characters, split evenly across its (query × branch) cells (default 36000).
- `SCORE_FLOOR_GAP` — how far below a query's best score a chunk may sit and still be returned (default 0.10).
- `BATCH_MAX_QUERIES` / `ARTICLE_MAX_NAMES` — the most queries / articles one call may carry (default 16 / 12).
- `SECTIONS_MAX_CHARS` — characters the `ARTICLE_PARTIAL` status may spend naming the sections a walk did not return (default 1500).
- `QUERY_LOG_MAX_CHARS` / `ERROR_LOG_MAX_CHARS` — caps on the verbatim query / error text stored in logs (default 2000 / 500).

## Event logging (`retrieve_docs`)
Every `lsfusion_retrieve_docs` call emits one structured JSON line (best-effort — a logging failure never breaks the tool). Envelope `{schema_version, event, ts (ISO-8601 UTC ms), server_version, ok}` plus `{query (capped), type, n_requested, n_results, top_score = max(results[].score), latency_ms, results:[{rank, source, score}]}` — no chunk text; `error_class`/`error_message` on failure. Written to **stderr** (keeps the STDIO transport's stdout protocol channel clean; Docker's `json-file` captures stderr anyway), and additionally to a dated file when `LOG_DIR` is set. See `MCP-FEEDBACK-PLAN.md`.

## Feedback reports
Each report `lsfusion_report_feedback` records becomes one file, `<REPORTS_DIR>/inbox/<UTC YYYYMMDDTHHMMSS>-<report_id>.json` (e.g. `20261119T101500-rpt_3f9c2a71d04e8b65.json`), so the names sort by time. The file holds the redacted report plus `report_id`, `ts`, `schema_version`, `server_version`, `dedup_fingerprint`, and `client` / `client_version` from the MCP handshake when the client gave them; a report from the IntelliJ plugin also carries `ide_build`, the IDE build its User-Agent names (e.g. `IU-253.28294.334`). It is written as `.<name>.tmp` in the same folder and renamed into place, so a reader that takes `*.json` never sees half a report.

The worker `tools/feedback_issues.py` turns each file, oldest first, into one issue in `lsfusion/platform-feedback`, then writes the record with `issue_number` and `issue_url` added to `<REPORTS_DIR>/processed/` and removes it from the inbox. It runs in the mcp container, the GitHub token on stdin:

```bash
docker exec -i stack-mcp-1 python -m tools.feedback_issues < /etc/lsfusion-feedback/github-token
python -m tools.feedback_issues --dry-run --reports-dir DIR [--out PREVIEW_DIR]   # no token, no network
```

- **Issue:** the title is the start of `problem_summary` (about 100 characters, cut at a word), without leading tags such as `[CONVEYOR][S3]`; the body keeps them. The body has short sections (problem, expected vs actual, suggested change, eval errors, outcome, who reported it), the full record in a collapsed JSON block, and ends with `<!-- report_id: rpt_… -->`. Agent text cannot ping or link: mentions, `#123` references, URLs and link brackets get a zero-width space, and HTML is escaped outside code. Over GitHub's size limit the JSON block is cut first.
- **Labels:** three per issue: `raw` (an agent's report as received — primary data, kept unedited; the work it leads to goes into issues of its own), `area:docs|platform|search` and `target:<primary_target>`. The worker creates any of its 12 labels that are missing, each with a description, and never changes other labels.
- **No duplicates:** before posting, it lists the repo's issues created since the oldest pending report (the issues list, not search, so it is never stale) and only moves a file whose issue already carries its marker. Runs must not overlap.
- **Limits:** writes are 2 s apart. On a rate-limit answer (403/429) it stops and leaves the rest for the next run. It prints `posted N, skipped N, failed N, left N` and exits 1 if anything failed, 2 if there is no token.

The `reports` event stream carries no report content: a stored report logs one `report_stored` line with `report_id`, `signal_type` and `payload_size`, and a refused one logs `report_feedback` with `ok: false` and its `reason`. Full `report_feedback` records with `ok: true` are what schema v4 and earlier wrote, before reports moved to files.

## Docker

Build and run:
```bash
docker build -t lsfusion/mcp:latest .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v "$PWD/snapshot:/data/snapshot:ro" \
  lsfusion/mcp:latest
```

Or via Compose:
```bash
docker compose up --build
```

## Production secrets (where to store keys)
**Do not hardcode secrets**. Options:
1. **Kubernetes Secrets + external secret store**  
   - Store secrets in AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault.  
   - Sync into K8s as `Secret` via External Secrets Operator.  
   - Mount as env vars in the Deployment:
     ```yaml
     env:
       - name: OPENAI_API_KEY
         valueFrom: { secretKeyRef: { name: mcp-secrets, key: openai } }
     ```

2. **Docker Swarm / Compose secrets**  
   - Use `secrets:` and mount files into the container, then export into env at entrypoint:
     ```yaml
     services:
       mcp:
         image: lsfusion/mcp:latest
         secrets: [openai_key]
     secrets:
       openai_key: { file: ./secrets/openai_key.txt }
     ```
   - Read them in an entrypoint script:
     ```bash
     export OPENAI_API_KEY="$(cat /run/secrets/openai_key)"
     exec python server.py http --host 0.0.0.0 --port 8000
     ```

3. **Cloud run / App services** (ECS, Cloud Run, App Service)  
   - Inject as environment variables wired to a managed secret store (e.g., AWS Parameter Store / Secrets Manager).
   - Rotate periodically; grant least-privilege IAM.

4. **CI/CD (GitHub Actions)**  
   - Store in **Actions Secrets**.  
   - At build/deploy time pass them into the container as env vars or bake only into the runtime environment (never into the image).

**This app reads credentials from environment variables**, so your orchestrator should inject them from a secure store.
Prefer secret stores over committing `.env` files.

## Hardening checklist
- Run as non-root (done in Dockerfile).
- Keep logs to stdout/stderr; in STDIO mode, avoid extra prints (MCP uses stdio).
- Set request timeouts and retries in your MCP client / reverse proxy.
- Add health endpoint (optional) and readiness checks on `/mcp` handshake.


## HTTP transport configuration (FastMCP)
FastMCP reads host/port from environment variables:
- `MCP_HOST` (default: `0.0.0.0`)
- `MCP_PORT` (default: `8000`)

Examples:

**Local run**
```bash
export OPENAI_API_KEY=sk-... RAG_SNAPSHOT_PATH=./corpus.npz
export MCP_HOST=0.0.0.0 MCP_PORT=8000
python server.py http
```

**Docker**
```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v "$PWD/snapshot:/data/snapshot:ro" \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_PORT=8000 \
  ghcr.io/<org>/<repo>/lsfusion-mcp:latest
```
