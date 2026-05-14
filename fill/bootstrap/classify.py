"""Bootstrap classifier — Docusaurus sidebars.js → platform/docs/manifest.json.

Logic (matches RAG-PLAN.md "Bootstrap" section):
  1. Snapshot sidebars.js + record provenance (commit_sha, git_dirty, fetched_at).
  2. Parse JS via Node subprocess (one-line `node -e "console.log(JSON.stringify(...))"`).
  3. Walk the tree; the closest ancestor with label "Paradigm" / "Language" / "How-to"
     determines sourceType for the leaf doc-id. Other tops → "unclassified" sentinel.
  4. Cross-check with `ls <docs-dir>/*.md`:
       - slug present in sidebar tree with recognized ancestor → use mapped sourceType
       - slug present in sidebar tree but ancestor category is NOT in
         {Paradigm, Language, How-to} → default to "paradigm" (legacy DocsStruct),
         recorded as "defaulted_no_category".
       - slug NOT in sidebar tree at all → default to "paradigm", recorded as
         "defaulted_not_in_sidebars".
       - slug in sidebar tree but NO .md file → orphan, record in report.
       - if any sidebars node has unexpected SHAPE (parser confusion) → BLOCKING
         (classify.py exits 2 before writing manifest).
  5. Emit:
       <out-manifest>  : platform/docs/manifest.json
       <out-snapshot>  : <dir>/sidebars-snapshot.js + sidebars-snapshot.meta.json
       <out-report>    : <dir>/bootstrap-report.md

Run example (from aggregate root):
  python -m fill.bootstrap.classify \\
    --sidebars docusaurus/sidebars.js \\
    --docs-dir platform/docs/en \\
    --out-manifest platform/docs/manifest.json \\
    --out-snapshot platform/.rag/bootstrap/ \\
    --out-report platform/.rag/bootstrap/bootstrap-report.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED_SOURCE_TYPES = ("paradigm", "language", "how-to")
DEFAULT_SOURCE_TYPE = "paradigm"   # legacy DocsStruct.java behavior for slugs without recognized category
NO_CATEGORY = "__no_category__"     # internal sentinel: slug is in sidebars but ancestor isn't Paradigm/Language/How-to

# Category labels in sidebars.js that determine sourceType for descendant leaves.
LABEL_TO_SOURCE_TYPE = {
    "Paradigm": "paradigm",
    "Language": "language",
    "How-to": "how-to",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sidebars", required=True, type=Path,
                   help="Path to docusaurus/sidebars.js (will be snapshotted)")
    p.add_argument("--docs-dir", required=True, type=Path,
                   help="Path to platform/docs/en (source of truth for actual files)")
    p.add_argument("--out-manifest", required=True, type=Path,
                   help="Where to write manifest.json (typically platform/docs/manifest.json)")
    p.add_argument("--out-snapshot", required=True, type=Path,
                   help="Directory to write sidebars-snapshot.js and sidebars-snapshot.meta.json")
    p.add_argument("--out-report", required=True, type=Path,
                   help="Where to write bootstrap-report.md")
    return p.parse_args()


def _git_info(path: Path) -> tuple[str | None, bool]:
    """Return (commit_sha, git_dirty) for the git repo containing `path`.
    Returns (None, False) if not in a git repo or git is unavailable.
    """
    resolved = path.resolve()
    repo = resolved if resolved.is_dir() else resolved.parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
            check=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
            check=True, timeout=10,
        ).stdout.strip()
        return sha, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None, False


def _sidebars_to_json(sidebars_path: Path) -> dict:
    """Convert sidebars.js (CommonJS module.exports) to a Python dict via Node."""
    script = (
        f"const s = require({json.dumps(str(sidebars_path.resolve()))}); "
        "process.stdout.write(JSON.stringify(s));"
    )
    try:
        result = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, check=True, timeout=30,
        )
    except FileNotFoundError:
        sys.exit("ERROR: `node` is required to parse sidebars.js. Install Node.js and retry.")
    except subprocess.TimeoutExpired:
        sys.exit("ERROR: node timed out (30s) parsing sidebars.js — possible infinite require() loop.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ERROR: node failed to evaluate sidebars.js:\n{e.stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"node returned non-JSON output (first 200 chars): {result.stdout[:200]!r}; "
                         f"json error: {e}") from e


def _walk(node, ancestor_source_type: str | None, slug_to_source: dict[str, str]) -> None:
    """Recursively walk the sidebars tree.

    `node` is one of:
      - str: leaf doc-id (slug)
      - dict: {type: 'category', label: ..., link: ..., items: [...]}
              or {type: 'doc', id: ...}
              or {type: 'link', ...} etc.
      - list: items collection (top-level sidebar)

    `ancestor_source_type` is the closest matched sourceType from a parent
    category's `label`; None until we hit Paradigm/Language/How-to.
    """
    if isinstance(node, list):
        for item in node:
            _walk(item, ancestor_source_type, slug_to_source)
        return

    if isinstance(node, str):
        # Leaf slug — route through _record_slug for consistent duplicate-warning diagnostics.
        _record_slug(slug_to_source, node, ancestor_source_type)
        return

    if isinstance(node, dict):
        ntype = node.get("type")
        label = node.get("label")

        # Inline doc node: {"type":"doc","id":"slug"} — record the leaf directly.
        if ntype == "doc":
            doc_id = node.get("id")
            if not isinstance(doc_id, str):
                raise ValueError(f"sidebars 'doc' node without string id: {_short_repr(node)}")
            _record_slug(slug_to_source, doc_id, ancestor_source_type)
            return

        # Reference / external link nodes — non-doc, no slug recorded.
        if ntype in ("link", "ref", "html"):
            return

        # Category — bump sourceType context if label matches; record link.id if present;
        # recurse into items.
        if ntype == "category":
            new_st = ancestor_source_type
            if label in LABEL_TO_SOURCE_TYPE:
                new_st = LABEL_TO_SOURCE_TYPE[label]

            link = node.get("link")
            if isinstance(link, dict) and link.get("type") == "doc":
                link_id = link.get("id")
                if isinstance(link_id, str):
                    _record_slug(slug_to_source, link_id, new_st)
            # link types other than 'doc' (e.g. generated-index, none) — no slug to record.

            items = node.get("items")
            if items is None:
                raise ValueError(f"sidebars 'category' node without items: {_short_repr(node)}")
            _walk(items, new_st, slug_to_source)
            return

        # Unknown / autogenerated / typo'd node shape — block (do not silently default).
        raise ValueError(f"unsupported sidebars node type {ntype!r}: {_short_repr(node)}")

    # Non-list/dict/str — block.
    raise ValueError(f"unexpected sidebars node shape: {_short_repr(node)}")


def _short_repr(node, limit: int = 200) -> str:
    """Truncated repr for error messages — sidebars subtrees can be huge."""
    s = repr(node)
    if len(s) <= limit:
        return s
    return s[:limit] + f"... ({len(s)} chars)"


def _record_slug(slug_to_source: dict[str, str], slug: str, source_type: str | None) -> None:
    """Record slug → sourceType. Warn on conflicting reassignment to a different sourceType."""
    new_value = source_type or NO_CATEGORY
    existing = slug_to_source.get(slug)
    if existing is not None and existing != new_value:
        print(
            f"WARNING: slug {slug!r} appears multiple times in sidebars with different sourceTypes "
            f"({existing!r} → {new_value!r}); last occurrence wins.",
            file=sys.stderr,
        )
    slug_to_source[slug] = new_value


def _classify(sidebars_tree: dict) -> dict[str, str]:
    """Walk all top-level sidebars and build slug → sourceType map."""
    slug_to_source: dict[str, str] = {}
    for _top_key, branch in sidebars_tree.items():
        _walk(branch, None, slug_to_source)
    return slug_to_source


def _scan_docs(docs_dir: Path) -> set[str]:
    """Scan top-level .md files in docs_dir. Plan assumes flat layout for `docs/en`.

    Warns if .md files exist in subdirectories — those would be silently excluded.
    If subdirectories appear later (e.g. `docs/en/forms/Display.md`), slug derivation
    in plan §"Chunking algorithm" needs to be revisited; this scan stays top-level
    until that change.
    """
    if not docs_dir.is_dir():
        sys.exit(f"ERROR: docs-dir not found: {docs_dir}")
    top_level = {p.stem for p in docs_dir.glob("*.md")}
    nested = list(p for p in docs_dir.rglob("*.md") if p.parent != docs_dir)
    if nested:
        print(
            f"WARNING: {len(nested)} .md file(s) found in subdirectories of {docs_dir} "
            "(top-level scan ignores them). Examples: " + ", ".join(str(p.relative_to(docs_dir)) for p in nested[:3]),
            file=sys.stderr,
        )
    return top_level


def _write_snapshot(sidebars_path: Path, out_snapshot_dir: Path) -> dict:
    out_snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_target = out_snapshot_dir / "sidebars-snapshot.js"
    shutil.copy2(sidebars_path, snapshot_target)

    commit_sha, git_dirty = _git_info(sidebars_path)
    meta = {
        "source_repo": "lsfusion/docusaurus",
        "source_path": "sidebars.js",
        "commit_sha": commit_sha,
        "git_dirty": git_dirty,
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetched_from": str(sidebars_path.resolve().parent),
    }
    (out_snapshot_dir / "sidebars-snapshot.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def _build_manifest(
    files: set[str], sidebar_map: dict[str, str]
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Compose final manifest and bucket lists for the report.

    Returns:
      manifest: {slug: {"sourceType": "..."}}
      buckets:
        - "classified":               slugs mapped via recognized ancestor category
        - "defaulted_no_category":    slugs in sidebars but ancestor != Paradigm/Language/How-to → paradigm
        - "defaulted_not_in_sidebars":slugs present as .md but absent from sidebars → paradigm
        - "orphan_sidebars":          slugs in sidebars with no matching .md file
    """
    manifest: dict[str, dict] = {}
    buckets: dict[str, list[str]] = defaultdict(list)

    for slug in sorted(files):
        if slug in sidebar_map:
            st = sidebar_map[slug]
            if st == NO_CATEGORY:
                manifest[slug] = {"sourceType": DEFAULT_SOURCE_TYPE}
                buckets["defaulted_no_category"].append(slug)
            else:
                manifest[slug] = {"sourceType": st}
                buckets["classified"].append(slug)
        else:
            manifest[slug] = {"sourceType": DEFAULT_SOURCE_TYPE}
            buckets["defaulted_not_in_sidebars"].append(slug)

    for slug in sorted(sidebar_map.keys()):
        if slug not in files:
            buckets["orphan_sidebars"].append(slug)

    return manifest, buckets


def _write_manifest(out_manifest: Path, manifest: dict[str, dict]) -> None:
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_report(
    out_report: Path,
    meta: dict,
    manifest: dict[str, dict],
    buckets: dict[str, list[str]],
) -> None:
    out_report.parent.mkdir(parents=True, exist_ok=True)
    total = len(manifest)
    per_type = Counter(rec["sourceType"] for rec in manifest.values())
    defaulted_no_cat = buckets.get("defaulted_no_category", [])
    defaulted_not_in_sb = buckets.get("defaulted_not_in_sidebars", [])
    defaulted_total = len(defaulted_no_cat) + len(defaulted_not_in_sb)
    orphans = buckets.get("orphan_sidebars", [])

    lines: list[str] = []
    lines.append("# RAG Bootstrap Report")
    lines.append("")
    lines.append("## Acceptance Required")
    lines.append("")
    if defaulted_total:
        lines.append(
            f"This bootstrap classified **{defaulted_total}** files as `paradigm` by default "
            f"({len(defaulted_no_cat)} via unrecognized sidebars category, "
            f"{len(defaulted_not_in_sb)} absent from sidebars). "
            "Misclassification will appear as retrieval bugs later."
        )
        lines.append("")
        lines.append("**To merge this PR, the reviewer must add to the PR description:**")
        lines.append("")
        lines.append(f"    BOOTSTRAP_DEFAULTS_REVIEWED: {defaulted_total}")
        lines.append("")
        lines.append(
            "Where the number matches the count shown above. "
            "This is an explicit acknowledgement that the defaulted lists were reviewed, not skimmed. "
            "Alternatively, add the same line to `.rag/bootstrap/acceptance.md`."
        )
    else:
        lines.append("No files defaulted to `paradigm` — sidebars.js covers every `.md` in docs/en "
                     "with a recognized category. No acceptance marker required.")
    lines.append("")

    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- Total `.md` files in docs/en: **{total}**")
    for st in ALLOWED_SOURCE_TYPES:
        n = per_type.get(st, 0)
        if n:
            lines.append(f"- `{st}`: {n}")
    lines.append("")

    if defaulted_no_cat:
        lines.append('## Files defaulted to "paradigm" (sidebars category not Paradigm/Language/How-to)')
        lines.append("")
        lines.append(
            "These files appear in `sidebars.js` but under categories like `install`, `Learning materials`, "
            "or single leaves (`IDE`, `Online_demo`) — none of {Paradigm, Language, How-to} ancestor. "
            "Classified as `paradigm` by fallback. If any are actually how-to or language reference, "
            "edit `manifest.json` before merge:"
        )
        lines.append("")
        for s in defaulted_no_cat:
            lines.append(f"- `{s}`")
        lines.append("")

    if defaulted_not_in_sb:
        lines.append('## Files defaulted to "paradigm" (slug not found in sidebars at all)')
        lines.append("")
        lines.append(
            "These `.md` files exist in docs/en but `sidebars.js` doesn't list them. "
            "Classified as `paradigm` by fallback. Check whether they should be added to sidebars "
            "(in docusaurus repo) or whether classification needs override here:"
        )
        lines.append("")
        for s in defaulted_not_in_sb:
            lines.append(f"- `{s}`")
        lines.append("")
    if orphans:
        lines.append("## Orphan sidebars-entries (in sidebars, no `.md` file)")
        lines.append("")
        lines.append(
            "These slugs appear in `sidebars.js` but have no matching `.md` file in docs/en. "
            "Non-blocking; may indicate stale sidebar entries."
        )
        lines.append("")
        for s in orphans:
            lines.append(f"- `{s}`")
        lines.append("")

    lines.append("## Sidebars snapshot provenance")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta, indent=2))
    lines.append("```")
    lines.append("")

    out_report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    # Parse + classify BEFORE writing snapshot — if parsing fails we don't want a stale
    # snapshot artifact pointing at a sidebars.js we can't actually use.
    try:
        sidebars_tree = _sidebars_to_json(args.sidebars)
        sidebar_map = _classify(sidebars_tree)
    except ValueError as e:
        # Parser confusion is a documented blocker: exit 2 with a clean message
        # (no traceback, no snapshot written yet).
        print(f"ERROR: sidebars.js parser confusion: {e}", file=sys.stderr)
        return 2

    meta = _write_snapshot(args.sidebars, args.out_snapshot)
    files = _scan_docs(args.docs_dir)
    manifest, buckets = _build_manifest(files, sidebar_map)
    _write_manifest(args.out_manifest, manifest)
    _write_report(args.out_report, meta, manifest, buckets)

    # Summary on stdout for quick eyeballing.
    print(f"manifest: {args.out_manifest} ({len(manifest)} entries)")
    print(f"snapshot: {args.out_snapshot}")
    print(f"report:   {args.out_report}")
    for bucket, items in buckets.items():
        if items:
            print(f"  {bucket}: {len(items)}")

    # Defaulted-to-paradigm buckets are intentional — they require BOOTSTRAP_DEFAULTS_REVIEWED
    # acceptance marker, NOT blocking exit.
    return 0


if __name__ == "__main__":
    sys.exit(main())
