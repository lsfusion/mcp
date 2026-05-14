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
CHUNKER_VERSION: str = "v1"

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
