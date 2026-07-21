
import os

# === Environment variables ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RAG_VECTOR_STORE_ID = os.environ.get("RAG_VECTOR_STORE_ID", "")

# Embedding model (kept for potential future direct-embedding flows;
# OpenAI vector_stores.search handles embedding server-side).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

# Attribute key under which sourceType is stored on each VS file
# (chunker writes the bare category value "language" / "paradigm" / "how-to" /
# "brief" / "rules" — the folder name, no combined-form prefix; see
# fill/ingest.py:_section_attributes).
SOURCETYPE = "sourceType"

# Attribute key under which the article slug is stored on each VS file (the
# article's published slug, e.g. "Brief" / "Rules_export"; see
# fill/ingest.py:_section_attributes).
SLUG = "slug"

# Bare category values (the docs/<lang> folder names) exposed by the chunker on
# VS file attributes.
SOURCETYPE_DOCUMENTATION = "documentation"
SOURCETYPE_DOCUMENTATION_PARADIGM = "paradigm"
SOURCETYPE_DOCUMENTATION_LANGUAGE = "language"
SOURCETYPE_DOCUMENTATION_HOWTO = "how-to"
SOURCETYPE_DOCUMENTATION_BRIEF = "brief"
SOURCETYPE_DOCUMENTATION_RULES = "rules"

# Combined identifiers returned in `DocItem.source` for backward
# compatibility with consumers that key on the legacy spelling.
SOURCETYPE_DOC_PARADIGM = f"{SOURCETYPE_DOCUMENTATION}-{SOURCETYPE_DOCUMENTATION_PARADIGM}"
SOURCETYPE_DOC_LANGUAGE = f"{SOURCETYPE_DOCUMENTATION}-{SOURCETYPE_DOCUMENTATION_LANGUAGE}"
SOURCETYPE_DOC_HOWTO = f"{SOURCETYPE_DOCUMENTATION}-{SOURCETYPE_DOCUMENTATION_HOWTO}"
SOURCETYPE_DOC_BRIEF = f"{SOURCETYPE_DOCUMENTATION}-{SOURCETYPE_DOCUMENTATION_BRIEF}"
SOURCETYPE_DOC_RULES = f"{SOURCETYPE_DOCUMENTATION}-{SOURCETYPE_DOCUMENTATION_RULES}"

# Per-sourceType chunk budget. Each retrieve_docs call issues one
# vector_stores.search per entry and merges the results by score.
TOP_K = {
    SOURCETYPE_DOC_PARADIGM: 3,
    SOURCETYPE_DOC_LANGUAGE: 3,
    SOURCETYPE_DOC_HOWTO: 3,
    SOURCETYPE_DOC_BRIEF: 3,
    SOURCETYPE_DOC_RULES: 3,
}

# === Structured event logging (feedback loop, Phase A; see MCP-FEEDBACK-PLAN.md) ===
# Bump when the log envelope/record shape changes.
LOG_SCHEMA_VERSION = 1
# Stamped into every event so analytics can attribute records to a build. Ops
# should set this (image digest / git sha) in the deployment env.
SERVER_VERSION = os.environ.get("MCP_SERVER_VERSION", "unknown")
# Directory for dated JSONL event files. Empty => stderr only (A1). A2 points
# this at the bind-mounted host dir on ai.lsfusion.org.
LOG_DIR = os.environ.get("LOG_DIR", "")
# Cap on the verbatim query stored in retrieval logs (privacy / prompt-stuffing guard).
QUERY_LOG_MAX_CHARS = int(os.environ.get("QUERY_LOG_MAX_CHARS", "2000"))
# Cap on error text stored when a call fails.
ERROR_LOG_MAX_CHARS = int(os.environ.get("ERROR_LOG_MAX_CHARS", "500"))

# === report_feedback (feedback loop, Phase B; see MCP-FEEDBACK-PLAN.md) ===
# Master switch. Off => the tool returns status="disabled" and stores nothing.
FEEDBACK_ENABLED = os.environ.get("FEEDBACK_ENABLED", "1").lower() not in ("0", "false", "no", "")
# Anti-abuse caps (reports from agents are noisy; reject pathological payloads).
REPORT_MAX_EVAL_ERRORS = int(os.environ.get("REPORT_MAX_EVAL_ERRORS", "50"))
REPORT_MAX_QUERIES = int(os.environ.get("REPORT_MAX_QUERIES", "50"))
# A single code excerpt longer than this => reject (no source dumps).
REPORT_CODE_EXCERPT_MAX_CHARS = int(os.environ.get("REPORT_CODE_EXCERPT_MAX_CHARS", "2000"))
# Total serialized payload cap.
REPORT_MAX_TOTAL_CHARS = int(os.environ.get("REPORT_MAX_TOTAL_CHARS", "65536"))
