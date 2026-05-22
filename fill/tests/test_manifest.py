"""Tests for fill.manifest — validate (folder structure), estimate, check-bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fill.manifest import (
    check_bootstrap_acceptance,
    estimate,
    main,
    validate,
)


# ─── validate (folder structure) ─────────────────────────────────────────────


def _make_docs(tmp_path: Path, relpaths: list[str]) -> Path:
    """Create docs/ with the given paths (relative to docs/, e.g.
    'language/AGGR.md'). Returns docs_dir."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for rel in relpaths:
        p = docs_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {Path(rel).stem}\n\nbody\n", encoding="utf-8")
    return docs_dir


def test_validate_ok(tmp_path):
    dd = _make_docs(tmp_path, [
        "language/AGGR_operator.md",
        "paradigm/Aggregations.md",
        "how-to/How-to_GROUP_SUM.md",
        "brief/Brief.md",
        "rules/Workflow.md",
    ])
    r = validate(dd)
    assert r.ok is True, r.errors
    assert r.errors == []
    assert r.stats["total"] == 5
    assert r.stats["per_type"] == {
        "language": 1, "paradigm": 1, "how-to": 1, "brief": 1, "rules": 1,
    }


def test_validate_file_without_category_folder(tmp_path):
    dd = _make_docs(tmp_path, ["language/AGGR.md", "STRAY.md"])
    r = validate(dd)
    assert r.ok is False
    assert any("STRAY.md" in e and "no category folder" in e for e in r.errors)


def test_validate_unknown_category_folder(tmp_path):
    dd = _make_docs(tmp_path, ["bogus/Foo.md"])
    r = validate(dd)
    assert r.ok is False
    assert any("bogus" in e and "not in" in e for e in r.errors)


def test_validate_bad_slug_format(tmp_path):
    # A space in the filename → stem fails SLUG_RE, even in a valid folder.
    dd = _make_docs(tmp_path, ["language/foo bar.md"])
    r = validate(dd)
    assert r.ok is False
    assert any("does not match" in e for e in r.errors)


def test_validate_real_world_slugs_accepted(tmp_path):
    real_slugs = [
        "=_statement",
        "How-to_GROUP_SUM",
        "IF_..._THEN_operator",
        "AGGR_operator",
        "Access_to_an_external_system_EXTERNAL",
    ]
    dd = _make_docs(tmp_path, [f"language/{s}.md" for s in real_slugs])
    r = validate(dd)
    assert r.ok is True, r.errors


def test_validate_docs_dir_missing(tmp_path):
    r = validate(tmp_path / "no-such-dir")
    assert r.ok is False
    assert any("docs-dir not found" in e for e in r.errors)


def test_validate_duplicate_slug_across_folders(tmp_path):
    # Same stem in two folders → same flat URL / section_id → collision.
    dd = _make_docs(tmp_path, ["language/Foo.md", "paradigm/Foo.md"])
    r = validate(dd)
    assert r.ok is False
    assert any("duplicate slug" in e for e in r.errors)


def test_validate_no_spurious_duplicate_for_unique_stems(tmp_path):
    dd = _make_docs(tmp_path, ["language/Foo.md", "language/Bar.md"])
    r = validate(dd)
    assert r.ok is True, r.errors
    assert not any("duplicate" in e for e in r.errors)


# ─── estimate ──────────────────────────────────────────────────────────────────


def test_estimate_basic(tmp_path):
    f1 = tmp_path / "a.md"
    f2 = tmp_path / "b.md"
    f1.write_text("hello")  # 5 chars → 1 token
    f2.write_text("world" * 100)  # 500 chars → 125 tokens
    est = estimate([f1, f2])
    assert est["total_chars"] == 505
    assert est["estimated_tokens_rough"] == 126
    assert set(est["files"]) == {str(f1), str(f2)}


def test_estimate_skips_missing_and_non_md(tmp_path):
    f1 = tmp_path / "a.md"
    f1.write_text("hi")
    f_missing = tmp_path / "missing.md"
    f_other = tmp_path / "c.txt"
    f_other.write_text("ignored")
    est = estimate([f1, f_missing, f_other])
    assert est["files"] == [str(f1)]
    assert est["total_chars"] == 2
    assert est["skipped"] == 2


def test_estimate_dedup_duplicate_path(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("hello")
    est = estimate([f, f, f])
    assert est["total_chars"] == 5  # not 15 — deduped
    assert est["files"] == [str(f)]


def test_estimate_all_invalid_signals_wiring_bug(tmp_path):
    est = estimate([tmp_path / "missing.md", tmp_path / "other.txt"])
    assert est["files"] == []
    assert est["skipped"] == 2
    assert est["total_chars"] == 0


# ─── check-bootstrap ───────────────────────────────────────────────────────────


def _make_report(tmp_path: Path, n: int) -> Path:
    p = tmp_path / "bootstrap-report.md"
    p.write_text(
        "## Acceptance Required\n\n"
        f"This bootstrap classified **{n}** files as `paradigm` by default ...\n\n"
        f"    BOOTSTRAP_DEFAULTS_REVIEWED: {n}\n\n"
    )
    return p


def test_check_bootstrap_zero_is_auto_pass(tmp_path):
    report = _make_report(tmp_path, 0)
    r = check_bootstrap_acceptance(report)
    assert r.ok is True
    assert r.stats["expected"] == 0


def test_check_bootstrap_marker_in_pr_description(tmp_path):
    report = _make_report(tmp_path, 18)
    r = check_bootstrap_acceptance(report, pr_description="...\nBOOTSTRAP_DEFAULTS_REVIEWED: 18\n...")
    assert r.ok is True
    assert r.stats["expected"] == 18
    assert r.stats["marker_source"] == "pr_description"


def test_check_bootstrap_marker_in_acceptance_file(tmp_path):
    report = _make_report(tmp_path, 18)
    accept = tmp_path / "acceptance.md"
    accept.write_text("BOOTSTRAP_DEFAULTS_REVIEWED: 18\n")
    r = check_bootstrap_acceptance(report, acceptance_file=accept)
    assert r.ok is True
    assert r.stats["marker_source"] == str(accept)


def test_check_bootstrap_marker_missing(tmp_path):
    report = _make_report(tmp_path, 18)
    r = check_bootstrap_acceptance(report)
    assert r.ok is False
    assert any("not found" in e for e in r.errors)


def test_check_bootstrap_marker_mismatch(tmp_path):
    report = _make_report(tmp_path, 18)
    r = check_bootstrap_acceptance(report, pr_description="BOOTSTRAP_DEFAULTS_REVIEWED: 5")
    assert r.ok is False
    assert any("mismatch" in e and "expected 18" in e and "5" in e for e in r.errors)


def test_check_bootstrap_missing_report(tmp_path):
    r = check_bootstrap_acceptance(tmp_path / "no-report.md")
    assert r.ok is False
    assert any("bootstrap-report not found" in e for e in r.errors)


def test_check_bootstrap_report_without_placeholder(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("no marker here")
    r = check_bootstrap_acceptance(report, pr_description="BOOTSTRAP_DEFAULTS_REVIEWED: 5")
    assert r.ok is False
    assert any("BOOTSTRAP_DEFAULTS_REVIEWED" in e and "line found" in e for e in r.errors)


# ─── CLI exit codes ────────────────────────────────────────────────────────────


def test_cli_validate_ok_exits_0(tmp_path, capsys):
    dd = _make_docs(tmp_path, ["language/Foo.md"])
    rc = main(["validate", "--docs-dir", str(dd)])
    assert rc == 0


def test_cli_validate_fail_exits_1(tmp_path, capsys):
    dd = _make_docs(tmp_path, ["bogus/Foo.md"])
    rc = main(["validate", "--docs-dir", str(dd)])
    assert rc == 1


def test_cli_estimate_json(tmp_path, capsys):
    f = tmp_path / "a.md"
    f.write_text("abcd")
    rc = main(["estimate", "--json", str(f)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_chars"] == 4
    assert out["estimated_tokens_rough"] == 1


def test_cli_check_bootstrap_zero(tmp_path, capsys):
    report = _make_report(tmp_path, 0)
    rc = main(["check-bootstrap", "--report", str(report)])
    assert rc == 0
