"""Manifest validator + CLI for platform/docs/manifest.json.

Used by Jenkins `ragValidateManifest.groovy` (PR-builder) and locally.
Three subcommands:

  - `validate`  — schema and consistency checks against docs/en
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

ALLOWED_SOURCE_TYPES = frozenset({"paradigm", "language", "how-to"})
UNCLASSIFIED_VALUE = "unclassified"          # sentinel — accepted only with --allow-unclassified
SLUG_RE = re.compile(r"^[A-Za-z0-9_\-=.]+$")  # platform/docs/en slugs allow letters/digits/_/-/=/. (no /, no whitespace, no :)

# Soft caps to bound memory in CI on any single read.
# manifest.json today is ~20 KB for 345 entries; 5 MB allows ~100x growth.
# .md files in platform/docs/en are <50 KB each; 2 MB per file is a generous safety cap.
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
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


def validate(
    manifest_path: Path,
    docs_dir: Path,
    *,
    allow_unclassified: bool = False,
) -> ValidationResult:
    """Schema + consistency check of manifest.json against docs/en files.

    Errors (any → ok=False):
      - manifest file missing / invalid JSON / wrong root type
      - entry not an object, or missing `sourceType` key
      - sourceType not in {paradigm, language, how-to}; `unclassified` only when allowed
      - slug fails SLUG_RE
      - .md file in docs_dir without manifest entry
      - manifest entry without matching .md file

    Warnings (don't fail):
      - .md files under subdirectories of docs_dir (silently ignored by top-level scan)
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        return ValidationResult(False, [f"manifest not found: {manifest_path}"])
    manifest_text, err = _read_text_capped(manifest_path, MAX_MANIFEST_BYTES)
    if err:
        return ValidationResult(False, [err])
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as e:
        return ValidationResult(False, [f"manifest is not valid JSON: {e}"])
    except RecursionError:
        # Deeply-nested JSON bombs hit Python's recursion limit during decode.
        return ValidationResult(False, [f"manifest exceeds JSON nesting limit: {manifest_path}"])
    if not isinstance(manifest, dict):
        return ValidationResult(
            False, [f"manifest root must be an object, got {type(manifest).__name__}"]
        )

    for slug, rec in manifest.items():
        # fullmatch (not match) — `re.match` with `^...$` accepts a trailing
        # newline because `$` matches before the final \n by default.
        if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug) or slug in (".", ".."):
            errors.append(f"manifest key {slug!r} does not match {SLUG_RE.pattern!r}")
        if not isinstance(rec, dict):
            errors.append(f"manifest[{slug!r}] must be an object, got {type(rec).__name__}")
            continue
        st = rec.get("sourceType")
        if st is None:
            errors.append(f"manifest[{slug!r}] missing 'sourceType'")
            continue
        if not isinstance(st, str):
            errors.append(f"manifest[{slug!r}].sourceType must be a string, got {type(st).__name__}")
            continue
        if st == UNCLASSIFIED_VALUE:
            if not allow_unclassified:
                errors.append(
                    f"manifest[{slug!r}].sourceType is 'unclassified' "
                    "(allowed only in bootstrap branch via --allow-unclassified)"
                )
        elif st not in ALLOWED_SOURCE_TYPES:
            errors.append(
                f"manifest[{slug!r}].sourceType={st!r} not in {sorted(ALLOWED_SOURCE_TYPES)}"
            )

    if not docs_dir.is_dir():
        errors.append(f"docs-dir not found: {docs_dir}")
        return ValidationResult(False, errors)

    # Accept any-case .md extension (Foo.md, Foo.MD, Foo.Md) — be tolerant of mixed-case files
    # since plan doesn't require lowercase-only. Case-insensitive stem-collision detection below
    # catches the actual hazard (two files mapping to the same logical slug on case-insensitive FS).
    md_paths = [p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md"]
    stem_to_paths: dict[str, list[str]] = {}
    for p in md_paths:
        stem_to_paths.setdefault(p.stem.casefold(), []).append(p.name)
    for stem_cf, names in stem_to_paths.items():
        if len(names) > 1:
            errors.append(
                f"duplicate filenames mapping to slug (case-insensitive collision): {sorted(names)}"
            )
    file_slugs = {p.stem for p in md_paths}
    manifest_slugs = set(manifest.keys())

    for slug in sorted(file_slugs - manifest_slugs):
        errors.append(f"{slug}.md present in {docs_dir} but missing manifest entry")
    for slug in sorted(manifest_slugs - file_slugs):
        errors.append(f"manifest[{slug!r}] has no corresponding .md file in {docs_dir}")

    # Cap nested-scan so a miswired Jenkins CWD (e.g. repo root, mounted tree) doesn't
    # blow up validation time. Stops after first 100 nested .md files; warning indicates "100+".
    nested: list[Path] = []
    NESTED_CAP = 100
    for p in docs_dir.rglob("*.md"):
        if p.parent != docs_dir:
            nested.append(p)
            if len(nested) >= NESTED_CAP:
                break
    if nested:
        sample = ", ".join(str(p.relative_to(docs_dir)) for p in nested[:3])
        count_label = f"{NESTED_CAP}+" if len(nested) >= NESTED_CAP else str(len(nested))
        warnings.append(
            f"{count_label} .md file(s) in subdirectories of {docs_dir} are NOT indexed: "
            f"e.g. {sample}"
        )

    # Stats: count only hashable string sourceTypes (non-string ones were already flagged above).
    per_type = Counter(
        rec["sourceType"]
        for rec in manifest.values()
        if isinstance(rec, dict) and isinstance(rec.get("sourceType"), str)
    )
    stats = {
        "total": len(manifest),
        "per_type": dict(per_type),
        "missing_in_manifest": len(file_slugs - manifest_slugs),
        "orphan_in_manifest": len(manifest_slugs - file_slugs),
    }

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

    pv = sub.add_parser("validate", help="schema + consistency check of manifest vs docs/en")
    pv.add_argument("--manifest", required=True, type=Path)
    pv.add_argument("--docs-dir", required=True, type=Path)
    pv.add_argument("--allow-unclassified", action="store_true",
                    help="permit sourceType='unclassified' (bootstrap branch only)")
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
        r = validate(args.manifest, args.docs_dir, allow_unclassified=args.allow_unclassified)
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
