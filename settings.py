
import os

# === Environment variables ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RAG_VECTOR_STORE_ID = os.environ.get("RAG_VECTOR_STORE_ID", "")

# The model every chunk AND every query is embedded with when the local index
# serves retrieval. Both sides must use the same one — a snapshot built by
# another model is refused at load, because mixing them does not fail loudly,
# it just returns nonsense. (The vector store embeds server-side and ignores
# this.)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "3072"))

# Attribute key under which sourceType is stored on each VS file
# (chunker writes the bare category value "language" / "paradigm" / "how-to" /
# "brief" / "rules" — the folder name, no combined-form prefix; see
# fill/ingest.py:_section_attributes).
SOURCETYPE = "sourceType"

# Attribute key under which the article slug is stored on each VS file (the
# article's published slug, e.g. "Brief" / "Rules_export"; see
# fill/ingest.py:_section_attributes).
SLUG = "slug"

# Attribute key under which the article's heading path is stored on each VS
# file ("Rules: navigator > Navigator rules"; see fill/ingest.py:_section_attributes).
# It is what a query is matched against for the title rerank.
HEADING_PATH = "heading_path"

# Attribute key under which the stable chunk id is stored on each VS file
# ("{slug}::{kebab-section}", written for every uploaded section by
# fill/ingest.py:_section_attributes). It is the id returned in `DocItem.id`
# and the one `retrieve_docs(exclude_ids=...)` filters on.
SECTION_ID = "section_id"

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

# Per-sourceType chunk budget. One entry costs one search, so this is also the
# per-branch slice of a merged result.
#
# There used to be a second, larger table for the case where `type` was passed
# explicitly, because `brief` and `rules` were split into per-area articles and
# a typed lookup was meant to bring a whole article back. Those two branches are
# no longer searched — an article is named and delivered whole by get_guidance —
# and for the three that remain the two tables held the same numbers, so the
# distinction went with them.
TOP_K = {
    SOURCETYPE_DOC_PARADIGM: 3,
    SOURCETYPE_DOC_LANGUAGE: 3,
    SOURCETYPE_DOC_HOWTO: 3,
}

# Ceiling on the chunks ONE call returns when it carries several queries.
# A batch must not be a way to buy context: four separate calls return 60
# chunks (~70 KB), and the whole reason a per-branch quota exists is the
# caller's context, not its patience. Under this cap a four-query batch costs
# ~28 KB — less than the calls it replaces. A single query is unaffected: its
# own quota (15 untyped, 7-8 typed) is already below the cap.
BATCH_TOTAL_CAP = int(os.environ.get("BATCH_TOTAL_CAP", "24"))

# Most queries a batch may carry. Beyond this the split leaves each query too
# little to be worth asking, so the call is refused rather than quietly
# truncated. Measured on real traffic: bursts of unrelated needs are 2-4 deep,
# and 8 covers all but a handful.
# Measured depth is 2-4 (452 bursts of two topics, 200 of three, 105 of four),
# and four is what has actually been verified end to end. Raise it only once
# per-query completeness has been measured at five and beyond.
BATCH_MAX_QUERIES = max(1, min(
    int(os.environ.get("BATCH_MAX_QUERIES", "4")),
    # More queries than the cap would give each less than one chunk, and the
    # max(1, ...) floor would then quietly break the cap instead.
    BATCH_TOTAL_CAP))

# === Local dense index (fill/snapshot.py, tools/local_index.py) ===
# Where the server looks for the snapshot. Empty (or a missing file) => the
# local index is simply not used and retrieval goes to the vector store, which
# is the behaviour this server has always had.
SNAPSHOT_PATH = os.environ.get("RAG_SNAPSHOT_PATH", "/data/snapshot/corpus.npz")

# Which backend serves retrieve_docs: "store" (OpenAI Vector Store, the way it
# has always worked) or "local" (the snapshot). Canary switch — the code ships
# first and defaults to the old path; flipping this env var is the rollout, and
# flipping it back is the rollback. "local" still falls back to the store when
# the snapshot is missing or the embedding call fails.
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "store")

# How old a snapshot may get before every load says so. A stale index does not
# fail — it answers from documentation that has moved on — so age is WARNED
# about rather than refused: refusing would take retrieval down for a weekend
# over a docs change that may not matter. The docs move on 2-3 days a week, so
# a week without a rebuild means the delivery pipeline is broken, not idle.
SNAPSHOT_MAX_AGE_DAYS = float(os.environ.get("SNAPSHOT_MAX_AGE_DAYS", "7"))

# === Structured event logging (feedback loop, Phase A; see MCP-FEEDBACK-PLAN.md) ===
# Bump when the log envelope/record shape changes, or when a field's MEANING
# does. v3: `top_score` is max(results[].score); in v2 it was the first hit's
# score, which stopped being the maximum while a heading bonus reordered the
# list (that bonus is gone, but the field keeps the v3 meaning). v4: a call may
# carry several queries — `query` is then a JOIN of them rather than anything a
# caller typed, and `queries` / `query_stats` / `results[].query_index` carry
# what actually happened. Group analytics by `queries`, never by the join.
LOG_SCHEMA_VERSION = 4
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
