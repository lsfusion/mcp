## lsFusion MCP brief

Before generating or modifying lsFusion code, call the `get_brief` tool and read these notes to the model:

- Context: this MCP exposes helpers for lsFusion docs search and DSL validation.
- Transports: FastMCP over streamable HTTP (`/`, SSE at `/sse`), host/port via `MCP_HOST`/`MCP_PORT`.
- Tools:
  - `retrieve_docs(query)`: prioritized documentation/reference chunks.
  - `retrieve_samples(query)`: how-tos and code samples.
  - `retrieve_learning(query)`: tutorials and articles.
  - `validate_dsl_statements(text)`: syntax validation for lsFusion statements.
- Workflow for the model:
  1) Invoke `get_brief` and read this content.
  2) If unsure about syntax/usage, call the retrieve_* tools with a focused query.
  3) Validate generated statements with `validate_dsl_statements`.
- Run server: `python server.py` (defaults to HTTP); switch transport via env `MCP_TRANSPORT`.
