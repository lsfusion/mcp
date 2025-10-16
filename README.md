
# lsfusion-mcp

An extensible MCP server hosting multiple tools. Ships with `retrieve_docs(query: string)` for RAG search
(OpenAI embeddings -> Pinecone), and a structure ready for future tools (e.g., code syntax checks).

Transports:
- **STDIO** for local development / desktop MCP clients.
- **Streamable HTTP** for production via Uvicorn (mounted at `/mcp`).

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...        PINECONE_API_KEY=...        PINECONE_INDEX=lsfusion        PINECONE_NAMESPACE=""

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

## Adding new tools
Create a new module under `tools/` and register it with `@mcp.tool()` in `server.py` (or build an auto-discovery
if you prefer). Keep tool signatures simple and JSON-serializable.

## Contract / output for `retrieve_docs`
Returns an array of objects:
```json
[
  { "source": "documentation-how-to", "text": "....", "score": 0.73 },
  { "source": "articles", "text": "....", "score": 0.69 }
]
```
Sorted by `score` descending.

## Environment variables
- `OPENAI_API_KEY` — OpenAI API key
- `PINECONE_API_KEY` — Pinecone API key
- `PINECONE_INDEX` — Pinecone index name (default `lsfusion`)
- `PINECONE_NAMESPACE` — Pinecone namespace (default empty)
- `EMBEDDING_MODEL` — OpenAI embedding model (default `text-embedding-3-large`)

## Docker

Build and run:
```bash
docker build -t lsfusion/mcp:latest .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e PINECONE_API_KEY=$PINECONE_API_KEY \
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
       - name: PINECONE_API_KEY
         valueFrom: { secretKeyRef: { name: mcp-secrets, key: pinecone } }
     ```

2. **Docker Swarm / Compose secrets**  
   - Use `secrets:` and mount files into the container, then export into env at entrypoint:
     ```yaml
     services:
       mcp:
         image: lsfusion/mcp:latest
         secrets: [openai_key, pinecone_key]
     secrets:
       openai_key: { file: ./secrets/openai_key.txt }
       pinecone_key: { file: ./secrets/pinecone_key.txt }
     ```
   - Read them in an entrypoint script:
     ```bash
     export OPENAI_API_KEY="$(cat /run/secrets/openai_key)"
     export PINECONE_API_KEY="$(cat /run/secrets/pinecone_key)"
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
- `MCP_HOST` (default: `127.0.0.1`)
- `MCP_PORT` (default: `8000`)

Examples:

**Local run**
```bash
export OPENAI_API_KEY=sk-... PINECONE_API_KEY=...
export MCP_HOST=0.0.0.0 MCP_PORT=8000
python server.py http
```

**Docker**
```bash
docker run --rm -p 8000:8000 \  -e OPENAI_API_KEY=$OPENAI_API_KEY \  -e PINECONE_API_KEY=$PINECONE_API_KEY \  -e MCP_HOST=0.0.0.0 \  -e MCP_PORT=8000 \  ghcr.io/<org>/<repo>/lsfusion-mcp:latest
```
