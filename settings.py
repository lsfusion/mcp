
import os

# === Environment variables ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

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


# Relevance floor: how far below the best score of its own query a chunk may
# sit and still be worth returning. An absolute gap in cosine space, not a
# fraction of the best score.
#
# It was a fraction, calibrated on probe queries whose vector WAS a chunk's
# vector — so the best score was 1.0 and 0.75 of it sat far above noise. Real
# queries never match a chunk that closely: measured over 200 taken from the
# live log, the best score is 0.558 median and 0.458 at p10. The same fraction
# there lands at 0.34, and the corpus's own noise is 0.262 median with a p99 of
# 0.551 — so on a flat query the floor was cutting almost nothing and the size
# budget went back to filling itself with near-noise.
#
# That p99 is worth stating plainly: two RANDOM chunks of this corpus score
# about what a typical query's BEST match does. Dense retrieval here has a low
# ceiling, and a rule anchored to the top score rather than to a fraction of it
# is the one that survives that.
#
# Measured over the same 200 live queries, chunks returned:
#                   =1    2-4   5-9  10-19   20+   median   hits the budget
#   0.75 x max       1     12    54     91    42       12         27%
#   max - 0.05      36    106    46     12     0        3          0%
#   max - 0.07      18     80    73     27     2        5          2%
#   max - 0.10       7     39    81     56    17        7         11%
#
# 0.10 is chosen at the EDGES, not the median. A single chunk is the failure
# this whole design is about — a chunk cannot show the constraint it depends on
# — and 0.05 leaves 36 queries of 200 holding exactly one. At the other end it
# keeps a median of 7, near the 9 the old per-branch quota returned, and cuts
# budget-bound responses from 27% to 11%. Nothing is ever returned empty.
SCORE_FLOOR_GAP = float(os.environ.get("SCORE_FLOOR_GAP", "0.10"))

# Ceiling on the total TEXT one call returns, in characters.
#
# The only limit that bounds what a caller actually spends. A chunk count never
# did: chunk length spans 15x — median 718 characters, p99 ~7k, max ~11k — so
# the same 24 chunks are ~27 KB of average text and ~183 KB of the largest.
# Responses of 70 KB were reaching callers, whose clients paged them out to a
# file to be parsed by hand.
#
# What this bounds is the TRANSPORT, not the token cost, and it is the honest
# unit for that: 35,297 characters of text serialise to 40,293 of JSON (1.142x),
# which clears the 50,000-character threshold above which clients stop handling
# a result inline. Against a token cap it is merely conservative — measured
# density runs from 2.75 to 20.59 characters per token across real articles, so
# this is at most ~13,000 tokens and usually far less. Counting real tokens
# would mean guessing at a client-side tokenizer we do not know.
#
# It is split evenly across the (query x branch) cells of a call and unused
# share flows to the others, so no branch and no query in a batch can crowd out
# the rest.
RESULT_MAX_CHARS = int(os.environ.get("RESULT_MAX_CHARS", "36000"))

# Most articles one call may read, by the same rule as the query cap: the point
# where giving every article a foothold and staying inside the budget stop being
# jointly keepable.
#
# Measured, 300 trials per size — even share per article, each taking a
# contiguous prefix, every article guaranteed its first remaining chunk:
#     2 articles   0% over budget    2.0 of  2 complete
#     4 articles   0% over budget    3.8 of  4
#     8 articles   0% over budget    7.1 of  8
#    12 articles   0% over budget    8.7 of 12
#    16 articles   2% over budget    9.3 of 16
#    24 articles  11% over budget    9.1 of 24
#    32 articles  30% over budget    8.4 of 32
#
# 12 is the last size that never overflowed. It is an admission limit and a
# proxy, not a proof: the real condition is whether the sum of each article's
# next mandatory chunk fits, which depends on WHICH articles and which page, and
# is checked when the response is built. The cap keeps the common case from
# reaching that check at all. Past 12 the completed count also stops rising —
# more articles asked for, fewer finished — but that is a usefulness argument,
# and only the broken promise justifies refusing.
ARTICLE_MAX_NAMES = max(1, int(os.environ.get("ARTICLE_MAX_NAMES", "12")))

# Most queries a batch may carry — the point where this server's two promises
# stop being jointly keepable, not a view about how many questions an agent
# ought to have.
#
# The promises: every query in a batch gets at least its best chunk (a query
# answered with nothing is indistinguishable from a query nothing matched), and
# the response fits `RESULT_MAX_CHARS` (a response the client pages out to a
# file is a response nobody reads). Both hold easily while a cell's share is
# bigger than a chunk. Past that they start to fight: keeping every query
# answered means exceeding the budget, and holding the budget means starving
# the tail of the batch.
#
# Measured, 60 batches at each size, with the starvation guarantee in place:
#     8 queries  median 17,158  max 35,280  over budget  0/60
#    16 queries  median 34,839  max 35,964  over budget  0/60
#    20 queries  median 35,791  max 40,456  over budget  3/60
#    24 queries  median 35,879  max 46,276  over budget  9/60
#    28 queries  median 35,959  max 59,005  over budget 16/60
#
# So 16. It is the last size at which both promises held in every trial, and it
# is derived rather than chosen — the earlier 4 and 8 were traffic observations
# and the arithmetic of a budget that no longer exists.
#
# Refusing past it is not distrust of the caller. A batch is context-neutral —
# measured, it costs exactly what the separate calls it replaces cost — so the
# caller loses nothing by splitting, and gains an answer to every query instead
# of silence for some of them.
BATCH_MAX_QUERIES = max(1, int(os.environ.get("BATCH_MAX_QUERIES", "16")))

# === Local dense index (fill/snapshot.py, tools/local_index.py) ===
# Where the server looks for the snapshot. Empty (or a missing file) => the
# local index is simply not used and retrieval goes to the vector store, which
# is the behaviour this server has always had.
SNAPSHOT_PATH = os.environ.get("RAG_SNAPSHOT_PATH", "/data/snapshot/corpus.npz")

# Which backend serves retrieve_docs: "store" (OpenAI Vector Store, the way it
# has always worked) or "local" (the snapshot). Canary switch — the code ships
# first and defaults to the old path; flipping this env var is the rollout, and

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
