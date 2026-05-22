"""Docs folder-structure validator + CLI helpers for platform/docs.

Category comes from the first subfolder under docs/<lang> (language/paradigm/
how-to/brief/rules) — there is no manifest. `SLUG_RE` lives here as the single
source of truth for slug shape (imported by `fill.config`). Three subcommands:

  - `validate`  — folder-structure + slug-shape check of docs/<lang>
  - `estimate`  — rough size estimate for changed .md files (PR comment)
  - `check-bootstrap`  — verify BOOTSTRAP_DEFAULTS_REVIEWED marker on bootstrap PRs

Exit codes (CLI):
  0  — checks passed
  1  — validation failed (errors present, file missing, schema bad, etc.)
  2  — argparse usage error (bad/missing subcommand or argument) — set by argparse,
       not by this module. Jenkins consumers can treat 2 as a wiring/config bug.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_SOURCE_TYPES = frozenset({"paradigm", "language", "how-to", "brief", "rules"})
SLUG_RE = re.compile(r"^[A-Za-z0-9_\-=.]+$")  # platform/docs/<lang> slugs allow letters/digits/_/-/=/. (no /, no whitespace, no :)

# Soft caps to bound memory in CI on any single read.
# .md files in platform/docs are <50 KB each; 2 MB per file is a generous safety cap.
MAX_MD_BYTES       = 2 * 1024 * 1024
MAX_REPORT_BYTES   = 1 * 1024 * 1024


def _read_text_capped(path: Path, max_bytes: int) -> tuple[str | None, str | None]:
    """Read text with a size cap. Returns (text, error_message). One of the two is None.

    Refuses to read non-regular files (devices, FIFOs, procfs entries) — `stat().st_size`
    can lie for those, allowing a small reported size with unbounded actual read.
    """
    try:
        st = path.stat()
    except OSError as e:
        return None, f"could not stat {path}: {e}"
    import stat as _stat
    if not _stat.S_ISREG(st.st_mode):
        return None, f"{path}: not a regular file (mode={oct(st.st_mode)})"
    if st.st_size > max_bytes:
        return None, f"{path}: file size {st.st_size} bytes exceeds cap {max_bytes} bytes"
    try:
        return path.read_text(encoding="utf-8"), None
    except (UnicodeDecodeError, OSError) as e:
        return None, f"could not read {path}: {e}"

# Recognized marker inside PR description, .rag/bootstrap/acceptance.md, OR
# the report's expected-count line. Same shape everywhere: own line, leading
# whitespace OK. Anchoring on its own line prevents false-positive matches on
# quoted/inline prose like "I confirm BOOTSTRAP_DEFAULTS_REVIEWED: 18 today".
MARKER_RE = re.compile(r"^\s*BOOTSTRAP_DEFAULTS_REVIEWED:\s*(\d+)\s*$", re.MULTILINE)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}


# ─── validate ──────────────────────────────────────────────────────────────────


def validate(docs_dir: Path) -> ValidationResult:
    """Folder-structure + slug-shape check of docs/<lang> (the manifest is gone;
    category is the first subfolder).

    Errors (any → ok=False):
      - docs_dir missing
      - an .md file directly under docs_dir (no category folder)
      - an .md file under a folder that is not an allowed sourceType
      - a slug (filename stem) failing SLUG_RE
      - two .md files sharing a slug (case-insensitive) — they would collide on
        the flat public URL / RAG section_id, even across different folders
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not docs_dir.is_dir():
        return ValidationResult(False, [f"docs-dir not found: {docs_dir}"])

    # Recurse: docs live one level down by category folder. Accept any-case .md.
    md_paths = sorted(
        p for p in docs_dir.rglob("*.md") if p.is_file() and p.suffix.lower() == ".md"
    )
    stem_to_rels: dict[str, list[str]] = {}
    per_type: Counter = Counter()
    for p in md_paths:
        rel = p.relative_to(docs_dir)
        parts = rel.parts
        if len(parts) < 2:
            errors.append(f"{rel}: .md file directly under docs-dir has no category folder")
        else:
            folder = parts[0]
            if folder not in ALLOWED_SOURCE_TYPES:
                errors.append(
                    f"{rel}: folder {folder!r} not in {sorted(ALLOWED_SOURCE_TYPES)}"
                )
            else:
                per_type[folder] += 1
        # fullmatch (not match) — `$` would otherwise accept a trailing newline.
        if not SLUG_RE.fullmatch(p.stem) or p.stem in (".", ".."):
            errors.append(f"{rel}: slug {p.stem!r} does not match {SLUG_RE.pattern!r}")
        stem_to_rels.setdefault(p.stem.casefold(), []).append(str(rel))

    for _stem_cf, rels in sorted(stem_to_rels.items()):
        if len(rels) > 1:
            errors.append(
                f"duplicate slug (case-insensitive) across files: {sorted(rels)}"
            )

    stats = {"total": len(md_paths), "per_type": dict(per_type)}
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, stats=stats)


# ─── estimate ──────────────────────────────────────────────────────────────────


def estimate(changed_files: list[Path]) -> dict:
    """Rough size estimate for a PR-comment summary.

    Returns dict with: files (deduped, sorted), total_chars, estimated_tokens_rough,
    skipped (count of inputs that were filtered: missing, non-.md, decode error).
    `skipped > 0` with `files == []` is the wiring-bug signal for the Jenkins consumer
    (e.g. wrong CWD, paths relative to wrong root).
    Token estimate is a 4-chars-per-token heuristic — coarse but good enough for
    "10s vs 100s of files" PR comments. The real tokenizer runs at ingest time.
    """
    seen: set[Path] = set()
    files: list[str] = []
    total_chars = 0
    skipped = 0
    for f in changed_files:
        if f in seen:
            continue
        seen.add(f)
        if not f.exists() or f.suffix != ".md":
            skipped += 1
            continue
        content, err = _read_text_capped(f, MAX_MD_BYTES)
        if err:
            skipped += 1
            continue
        total_chars += len(content)
        files.append(str(f))
    files.sort()
    return {
        "files": files,
        "total_chars": total_chars,
        "estimated_tokens_rough": total_chars // 4,
        "skipped": skipped,
    }


# ─── check-bootstrap ───────────────────────────────────────────────────────────


def check_bootstrap_acceptance(
    report_path: Path,
    *,
    pr_description: str | None = None,
    acceptance_file: Path | None = None,
) -> ValidationResult:
    """Verify the BOOTSTRAP_DEFAULTS_REVIEWED: <N> marker matches the report's expected count.

    Expected count is read from `report_path` (the BOOTSTRAP_DEFAULTS_REVIEWED:<N> snippet
    embedded in the "Acceptance Required" section of bootstrap-report.md). Marker can be
    in either `pr_description` (provided by Jenkins from `gh pr view`) or
    `acceptance_file` (`.rag/bootstrap/acceptance.md`). At least one must contain it.

    Special case: if the report contains `BOOTSTRAP_DEFAULTS_REVIEWED: 0`, no marker
    is required — passes automatically.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not report_path.exists():
        return ValidationResult(False, [f"bootstrap-report not found: {report_path}"])
    report_text, err = _read_text_capped(report_path, MAX_REPORT_BYTES)
    if err:
        return ValidationResult(False, [err])
    matches = MARKER_RE.findall(report_text)
    if not matches:
        return ValidationResult(
            False,
            [f"{report_path}: no `BOOTSTRAP_DEFAULTS_REVIEWED: <N>` line found "
             f"(must be on its own line). Classify.py emits one even when N=0."],
        )
    # Multiple matches in the report indicate either a duplicate marker or a stale snippet
    # in commentary — both ambiguous, refuse to guess.
    if len(set(matches)) > 1:
        return ValidationResult(
            False,
            [f"{report_path}: multiple conflicting BOOTSTRAP_DEFAULTS_REVIEWED values: {set(matches)}"],
        )
    expected = int(matches[0])
    stats = {"expected": expected}

    if expected == 0:
        # Nothing was defaulted; no acceptance required.
        return ValidationResult(ok=True, stats=stats)

    found: list[tuple[str, int]] = []  # (source, value)

    def _scan_marker(text: str, source: str) -> ValidationResult | None:
        """Append marker hits from `text` to `found`. Fail-fast if same source has conflicting values."""
        vals = [int(v) for v in MARKER_RE.findall(text)]
        unique = set(vals)
        if len(unique) > 1:
            return ValidationResult(
                False,
                [f"{source}: multiple conflicting BOOTSTRAP_DEFAULTS_REVIEWED values: {sorted(unique)}"],
            )
        if vals:
            found.append((source, vals[0]))
        return None

    if pr_description:
        conflict = _scan_marker(pr_description, "pr_description")
        if conflict:
            return conflict

    if acceptance_file and acceptance_file.exists():
        acceptance_text, err = _read_text_capped(acceptance_file, MAX_REPORT_BYTES)
        if err:
            return ValidationResult(False, [err])
        conflict = _scan_marker(acceptance_text, str(acceptance_file))
        if conflict:
            return conflict

    if not found:
        errors.append(
            f"BOOTSTRAP_DEFAULTS_REVIEWED marker not found in PR description or "
            f"{acceptance_file or '.rag/bootstrap/acceptance.md'}; expected: {expected}"
        )
    else:
        # Strict contract: EVERY provided source must match expected. If any source disagrees
        # (stale value), reject — even if another source happens to match. Prevents silent
        # acceptance when, say, .rag/bootstrap/acceptance.md got out of date but PR body has
        # the fresh marker.
        mismatches = [(src, val) for src, val in found if val != expected]
        if mismatches:
            seen = ", ".join(f"{src}={val}" for src, val in found)
            errors.append(
                f"BOOTSTRAP_DEFAULTS_REVIEWED mismatch: expected {expected}, found {seen}"
            )
        else:
            stats["marker_source"] = ",".join(src for src, _ in found)

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, stats=stats)


# ─── CLI ───────────────────────────────────────────────────────────────────────


def _print_result(result: ValidationResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.as_dict(), indent=2))
        return
    for w in result.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in result.errors:
        print(f"ERROR: {e}", file=sys.stderr)
    if result.stats:
        print(f"stats: {json.dumps(result.stats, sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fill.manifest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="folder-structure + slug-shape check of docs/<lang>")
    pv.add_argument("--docs-dir", required=True, type=Path)
    pv.add_argument("--json", action="store_true", help="machine-readable output")

    pe = sub.add_parser("estimate", help="rough size estimate for changed .md files (PR comments)")
    pe.add_argument("files", nargs="+", type=Path, help="changed .md files")
    pe.add_argument("--json", action="store_true")

    pb = sub.add_parser("check-bootstrap", help="verify BOOTSTRAP_DEFAULTS_REVIEWED marker")
    pb.add_argument("--report", required=True, type=Path,
                    help="bootstrap-report.md (contains the expected count)")
    pb.add_argument("--pr-description", default=None,
                    help="PR description body (from `gh pr view --json body`)")
    pb.add_argument("--acceptance-file", default=None, type=Path,
                    help="fallback marker file, e.g. .rag/bootstrap/acceptance.md")
    pb.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "validate":
        r = validate(args.docs_dir)
        _print_result(r, args.json)
        return 0 if r.ok else 1

    if args.cmd == "estimate":
        est = estimate(args.files)
        if args.json:
            print(json.dumps(est, indent=2))
        else:
            print(f"files: {len(est['files'])}, chars: {est['total_chars']}, "
                  f"~tokens: {est['estimated_tokens_rough']}")
        return 0

    if args.cmd == "check-bootstrap":
        r = check_bootstrap_acceptance(
            args.report,
            pr_description=args.pr_description,
            acceptance_file=args.acceptance_file,
        )
        _print_result(r, args.json)
        return 0 if r.ok else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
