
import os

# === Environment variables ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RAG_VECTOR_STORE_ID = os.environ.get("RAG_VECTOR_STORE_ID", "")

# Embedding model (kept for potential future direct-embedding flows;
# OpenAI vector_stores.search handles embedding server-side).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

# Attribute key under which sourceType is stored on each VS file
# (chunker writes the bare manifest value "language" / "paradigm" — no
# combined-form prefix; see fill/ingest.py:_section_attributes).
SOURCETYPE = "sourceType"

# Bare manifest values exposed by the chunker on VS file attributes.
SOURCETYPE_DOCUMENTATION = "documentation"
SOURCETYPE_DOCUMENTATION_PARADIGM = "paradigm"
SOURCETYPE_DOCUMENTATION_LANGUAGE = "language"

# Combined identifiers returned in `DocItem.source` for backward
# compatibility with consumers that key on the legacy spelling.
SOURCETYPE_DOC_PARADIGM = f"{SOURCETYPE_DOCUMENTATION}-{SOURCETYPE_DOCUMENTATION_PARADIGM}"
SOURCETYPE_DOC_LANGUAGE = f"{SOURCETYPE_DOCUMENTATION}-{SOURCETYPE_DOCUMENTATION_LANGUAGE}"

# Per-sourceType chunk budget. Each retrieve_docs call issues one
# vector_stores.search per entry and merges the results by score.
TOP_K = {
    SOURCETYPE_DOC_PARADIGM: 3,
    SOURCETYPE_DOC_LANGUAGE: 3,
}
