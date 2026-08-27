"""Pipeline-version constants used in section_payload_hash + state.pipeline_versions.

Bumping any of these triggers a full reindex via Step-0 version-drift detection
(see RAG-PLAN.md §"Step 0"). Bumps are intentional — pin tests in `tests/`
prevent accidental edits.

`SOURCE_URL_VERSION` lives in `fill.config` because it travels with the URL
contract; re-imported here so the unified tuple is in one place.
"""

from __future__ import annotations

from fill.config import SOURCE_URL_VERSION  # re-exported, do not edit here

# Bump when MarkdownHeaderTextSplitter config, how-to grouping rules, kebab
# conversion, secondary-split parameters, or section_id grammar changes.
# v4 (current): no chunker-logic change — the VS attach now explicitly
#   requests chunking_strategy static max_chunk_size_tokens=4096 /
#   chunk_overlap_tokens=0 (see fill.real_openai_client._CHUNKING_STRATEGY).
#   Previously the strategy was omitted, so OpenAI applied `auto` (=800/400)
#   and re-split every section into 50%-overlapping near-duplicate chunks.
#   The bump forces a full re-scan AND re-upload so the whole index is
#   rebuilt under the correct strategy.
# v3: undersized-run merge — trigger on a single short section
#   (cur < SHORT_SECTION_FLOOR_TOKENS=50), absorb consecutive same-parent
#   siblings while accumulated body stays ≤ MERGE_SOFT_CAP_TOKENS=256 and
#   either acc < floor or next < floor (so a stub merges with its
#   explanatory neighbor even when the neighbor itself isn't tiny);
#   stub detection — files whose body is exactly `### (Under development)`
#   emit zero sections.
# v2: _merge_short_siblings — both-short gate.
CHUNKER_VERSION: str = "v4"

# Reserved for future glossary preprocessing (DSL term aliasing, synonym
# injection). Placeholder until ingest needs it.
GLOSSARY_VERSION: str = "v0"

# Bump when the deterministic section prefix shape changes
# (`# {sourceType}: {heading_path}\n\n{content}`).
PREFIX_VERSION: str = "v1"


def pipeline_versions() -> dict[str, str]:
    """Return the full version tuple as a dict, matching the shape persisted
    in `platform/.rag/openai-state.json::pipeline_versions`."""
    return {
        "chunker_version": CHUNKER_VERSION,
        "glossary_version": GLOSSARY_VERSION,
        "prefix_version": PREFIX_VERSION,
        "source_url_version": SOURCE_URL_VERSION,
    }
