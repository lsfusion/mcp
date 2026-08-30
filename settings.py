
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
# The article's own `keywords` frontmatter — the words a reader searches for
# that its title does not use ("validation" for an article titled
# "constraints"). Stamped by fill/ingest.py only on articles that declare any.
KEYWORDS = "keywords"

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

# Per-sourceType chunk budget for the DEFAULT search — the one that runs when
# `type` is omitted and searches every branch, merging the results by score.
# Each entry costs one vector_stores.search, so this is also the per-branch
# slice of the merged result.
TOP_K = {
    SOURCETYPE_DOC_PARADIGM: 3,
    SOURCETYPE_DOC_LANGUAGE: 3,
    SOURCETYPE_DOC_HOWTO: 3,
    SOURCETYPE_DOC_BRIEF: 3,
    SOURCETYPE_DOC_RULES: 3,
}

# Per-sourceType chunk budget used ONLY when `type` is passed explicitly, i.e.
# when the whole response comes from that single branch.
# `rules` and `brief` get more than the other branches: they are split into
# per-area articles, and one targeted query is meant to cover the area. Raising
# it in TOP_K instead would apply to the untyped search too, and drag those
# chunks into every generic result at the expense of the other branches — hence
# the separate table.
# Sized against the corpus, not guessed: a typed lookup should be able to bring
# back a whole per-area article. Measured on the current corpus the biggest are
# 7 sections (brief) and 6 (rules); the quota is set one above each, so a normal
# amount of growth does not silently start truncating. A quota below the article
# guarantees it arrives incomplete, with the embedding choosing what is missing.
TYPED_TOP_K = {
    SOURCETYPE_DOC_PARADIGM: 3,
    SOURCETYPE_DOC_LANGUAGE: 3,
    SOURCETYPE_DOC_HOWTO: 3,
    SOURCETYPE_DOC_BRIEF: 8,
    SOURCETYPE_DOC_RULES: 7,
}

# How many candidates each branch search asks the store for before the title
# rerank picks the quota out of them. Separate from the quota on purpose: a
# short query's own article is often outside the first few, and promoting it
# is only possible among candidates we were given. Bigger costs response size
# from the store, not an extra call.
TITLE_RERANK_CANDIDATE_K = 30

# What a heading match adds to a chunk's store score, per fraction of the
# query's words carried by its heading path. Bounded on purpose: a named
# article overtakes a near neighbour without passing a hit the store scored
# decisively higher, so a query that names an article and one thing besides
# ("navigator caching") is helped rather than dropped back to nothing.
# Measured against the live store: at 0.3 queries that are just an area name go
# from 0 first places in 12 to 12, while natural-language questions (5/5) and
# questions sharing no wording with their article (6 of 8) stay exactly at the
# unmodified baseline, chunk for chunk. At 0.4 and above the latter start to
# erode; at 0.2 the area names only reach 10.
TITLE_MATCH_BOOST = 0.3

# Weight of a match in the SECTION part of the heading path, relative to the
# article's own title. A path reads "<article title> > <H2> > <H3>": the title
# says what the article is about, a section name is a detail of it, and a
# generic one ("Operator", "Examples") repeats across dozens of articles. One
# depth weight for the whole tail, not a list of exceptions.
HEADING_SECTION_WEIGHT = 0.4

# Words dropped before matching a query against a heading path. They are the
# words a caller adds to say WHICH KIND of answer it wants rather than what
# about — `rules`/`brief` name this corpus's own branches. An empty set after
# this means the query says nothing to match a title on, and nothing is
# promoted.
TITLE_MATCH_STOPWORDS = frozenset({
    "a", "an", "and", "brief", "for", "in", "of", "on", "rules", "the", "to", "with",
})

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

# The heading bonus again, on the LOCAL backend's scale. TITLE_MATCH_BOOST is
# 0.3 because that is what moves a vector-store score; a cosine's neighbours sit
# much closer together, and the same 0.3 there would let a heading match
# overturn the ranking outright. Measured on the local index: at 0.05 one-word
# area names go from 9 first places in 12 to 11, while 120 real queries move by
# +0.003 nDCG@5 (interval [-0.004, +0.009]) — i.e. it costs nothing and saves a
# narrow class.
LOCAL_TITLE_MATCH_BOOST = float(os.environ.get("LOCAL_TITLE_MATCH_BOOST", "0.05"))

# How old a snapshot may get before every load says so. A stale index does not
# fail — it answers from documentation that has moved on — so age is WARNED
# about rather than refused: refusing would take retrieval down for a weekend
# over a docs change that may not matter. The docs move on 2-3 days a week, so
# a week without a rebuild means the delivery pipeline is broken, not idle.
SNAPSHOT_MAX_AGE_DAYS = float(os.environ.get("SNAPSHOT_MAX_AGE_DAYS", "7"))

# === Structured event logging (feedback loop, Phase A; see MCP-FEEDBACK-PLAN.md) ===
# Bump when the log envelope/record shape changes, or when a field's MEANING
# does. v3: `top_score` is max(results[].score); in v2 it was the first hit's
# score, which stopped being the maximum once the heading bonus reorders.
LOG_SCHEMA_VERSION = 3
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
