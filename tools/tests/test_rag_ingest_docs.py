"""Tests for tools/rag_ingest_docs.py — the Jenkins driver script.

Sets up a fake `platform_root` (with `docs/en/`, `.rag/`, `docs/manifest.json`)
in a tmp dir, injects a `FakeGitRunner` so no real git calls happen, and
runs the orchestration core (`run()`) end-to-end against a
`FakeVectorStoreClient`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from fill.openai_client import FakeVectorStoreClient
from fill.state import State, load
from tools.rag_ingest_docs import (
    EXIT_INGEST_ERRORS,
    EXIT_OK,
    EXIT_SETUP_ERROR,
    run,
)


# ─── fakes ───────────────────────────────────────────────────────────────────


@dataclass
class FakeGitRunner:
    """Mirrors GitRunner's surface; tests preload the responses."""

    head: str = "head-sha-default"
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    head_calls: int = 0
    changed_calls: list[tuple[str, str, str]] = field(default_factory=list)

    def head_sha(self) -> str:
        self.head_calls += 1
        return self.head

    def changed_under(self, base_sha: str, head_sha: str, subdir: str) -> tuple[list[str], list[str]]:
        self.changed_calls.append((base_sha, head_sha, subdir))
        return list(self.changed), list(self.removed)


# ─── platform_root fixture ─────────────────────────────────────────────────


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _platform_root(tmp_path: Path, *, files: dict[str, str], manifest: dict) -> Path:
    """Build a fake platform repo at tmp_path."""
    root = tmp_path / "platform"
    docs_en = root / "docs" / "en"
    docs_en.mkdir(parents=True)
    for name, body in files.items():
        _write(docs_en / name, body)
    _write(root / "docs" / "manifest.json", json.dumps(manifest, indent=2))
    (root / ".rag").mkdir()
    return root


def _md(title: str, body: str) -> str:
    return f"---\ntitle: {title}\n---\n\n{body}\n"


# ─── setup errors ──────────────────────────────────────────────────────────


def test_returns_setup_error_when_docs_root_missing(tmp_path):
    root = tmp_path / "platform"
    root.mkdir()
    code, _ = run(
        platform_root=root,
        client=FakeVectorStoreClient(),
        git=FakeGitRunner(),
        vector_store_id_override="vs_x",
    )
    assert code == EXIT_SETUP_ERROR


def test_returns_setup_error_when_manifest_missing(tmp_path):
    root = tmp_path / "platform"
    (root / "docs" / "en").mkdir(parents=True)
    code, _ = run(
        platform_root=root,
        client=FakeVectorStoreClient(),
        git=FakeGitRunner(),
        vector_store_id_override="vs_x",
    )
    assert code == EXIT_SETUP_ERROR


def test_returns_setup_error_when_no_vector_store_id(tmp_path):
    root = _platform_root(tmp_path, files={"AGGR.md": _md("AGGR", "## S\n\nx")},
                          manifest={"AGGR": {"sourceType": "language"}})
    code, _ = run(
        platform_root=root,
        client=FakeVectorStoreClient(),
        git=FakeGitRunner(),
        vector_store_id_override=None,  # not in state either
    )
    assert code == EXIT_SETUP_ERROR


# ─── forced full scan (first run) ──────────────────────────────────────────


def test_first_run_does_forced_full_scan_and_stamps_sentinels(tmp_path):
    root = _platform_root(
        tmp_path,
        files={
            "AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "OTHER.md": _md("OTHER", "## B\n\nbar"),
        },
        manifest={
            "AGGR": {"sourceType": "language"},
            "OTHER": {"sourceType": "paradigm"},
        },
    )
    client = FakeVectorStoreClient()
    git = FakeGitRunner(head="commit-001")

    code, stats = run(
        platform_root=root,
        client=client,
        git=git,
        vector_store_id_override="vs_x",
    )

    assert code == EXIT_OK
    assert stats.files_processed == 2
    assert stats.errors == []
    # Full-scan path means git diff is NOT consulted.
    assert git.changed_calls == []

    # State now stamps the cycle.
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_x"
    assert state.last_indexed_docs_commit == "commit-001"
    assert state.pipeline_versions is not None
    assert "AGGR.md" in state.files
    assert "OTHER.md" in state.files
    assert state.files["AGGR.md"].indexed_sourceType == "language"
    assert state.files["OTHER.md"].indexed_sourceType == "paradigm"


# ─── incremental (git diff) ────────────────────────────────────────────────


def test_incremental_uses_git_diff_after_baseline(tmp_path):
    root = _platform_root(
        tmp_path,
        files={
            "AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "OTHER.md": _md("OTHER", "## B\n\nbar"),
            "QUIET.md": _md("QUIET", "## C\n\nbaz"),
        },
        manifest={
            "AGGR": {"sourceType": "language"},
            "OTHER": {"sourceType": "paradigm"},
            "QUIET": {"sourceType": "language"},
        },
    )
    client = FakeVectorStoreClient()
    # Baseline run.
    run(
        platform_root=root,
        client=client,
        git=FakeGitRunner(head="commit-001"),
        vector_store_id_override="vs_x",
    )
    uploads_after_baseline = len(client.upload_calls)

    # Second run: only AGGR changed. Git says so.
    code, stats = run(
        platform_root=root,
        client=client,
        git=FakeGitRunner(head="commit-002", changed=["docs/en/AGGR.md"]),
        vector_store_id_override=None,  # use the state's stored vs_id
    )
    assert code == EXIT_OK
    # AGGR's content didn't actually change → fast-path skips it.
    assert stats.files_fast_path_skipped == 1
    # OTHER and QUIET not in git diff → not seen at all.
    assert stats.files_seen == 1
    # No new uploads expected.
    assert len(client.upload_calls) == uploads_after_baseline


def test_incremental_uploads_actually_changed_content(tmp_path):
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\noriginal")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")
    initial_uploads = len(client.upload_calls)

    # Actually change the content.
    (root / "docs" / "en" / "AGGR.md").write_text(
        _md("AGGR", "## S\n\nMODIFIED"), encoding="utf-8")

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(head="c2", changed=["docs/en/AGGR.md"]),
    )
    assert code == EXIT_OK
    assert stats.files_processed == 1
    assert stats.files_fast_path_skipped == 0
    assert len(client.upload_calls) > initial_uploads


# ─── stale handling ────────────────────────────────────────────────────────


def test_incremental_unions_stale_files(tmp_path):
    """A file marked stale by reconcile (or a prior failure) is processed
    even when git says nothing changed."""
    root = _platform_root(
        tmp_path,
        files={
            "AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "OTHER.md": _md("OTHER", "## B\n\nbar"),
        },
        manifest={
            "AGGR": {"sourceType": "language"},
            "OTHER": {"sourceType": "paradigm"},
        },
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")

    # Hand-edit state to mark OTHER stale.
    state_path = root / ".rag" / "openai-state.json"
    state = load(state_path)
    state.files["OTHER.md"].stale = True
    from fill.state import save as save_state
    save_state(state_path, state)

    # Git says nothing changed; stale union must still process OTHER.
    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(head="c2"),
    )
    assert code == EXIT_OK
    assert stats.files_seen == 1  # only the stale OTHER
    # OTHER's hash didn't actually change → fast-path skip is gated by stale=True,
    # so it actually re-runs. After this, stale should clear.
    state_after = load(state_path)
    assert state_after.files["OTHER.md"].stale is False


def test_stale_but_missing_file_is_skipped(tmp_path):
    """A stale entry whose underlying file no longer exists isn't processed
    via files_to_process (it'd error). It stays in state until a later run
    sees it in files_removed (or reconcile handles it)."""
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")

    # Manually inject a stale-but-missing file into state.
    state_path = root / ".rag" / "openai-state.json"
    state = load(state_path)
    from fill.state import FileRecord, save as save_state
    state.files["DELETED.md"] = FileRecord(stale=True)
    save_state(state_path, state)

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(head="c2"),
    )
    assert code == EXIT_OK
    # No error, no processing — DELETED.md was skipped because the file
    # isn't on disk.
    assert all("DELETED.md" not in err for err in stats.errors)


# ─── case B: files removed ─────────────────────────────────────────────────


def test_incremental_removes_files_per_git_diff(tmp_path):
    root = _platform_root(
        tmp_path,
        files={
            "AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "OTHER.md": _md("OTHER", "## B\n\nbar"),
        },
        manifest={
            "AGGR": {"sourceType": "language"},
            "OTHER": {"sourceType": "paradigm"},
        },
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")

    # Actually delete the file on disk, mirror the git diff.
    (root / "docs" / "en" / "OTHER.md").unlink()

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(head="c2", removed=["docs/en/OTHER.md"]),
    )
    assert code == EXIT_OK
    assert stats.files_removed == 1
    state = load(root / ".rag" / "openai-state.json")
    assert "OTHER.md" not in state.files


# ─── errors leave sentinels unstamped ──────────────────────────────────────


def test_errors_do_not_stamp_sentinels(tmp_path):
    """If any per-file error occurred, last_indexed_docs_commit and
    pipeline_versions must NOT be updated. The next run reprocesses."""
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nfoo")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    # Establish baseline.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x")

    # Now force an upload error on the next run.
    client = FakeVectorStoreClient()
    (root / "docs" / "en" / "AGGR.md").write_text(
        _md("AGGR", "## S\n\nchanged content"), encoding="utf-8")
    client.fail_upload_for_section_id.add("AGGR::s")

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(head="c2", changed=["docs/en/AGGR.md"]),
    )
    assert code == EXIT_INGEST_ERRORS
    assert stats.errors  # non-empty

    state = load(root / ".rag" / "openai-state.json")
    # Sentinels must still point at c1 / old versions — pinned to the exact
    # baseline value, not just "not c2".
    assert state.last_indexed_docs_commit == "c1"
    assert state.pipeline_versions is not None  # also pinned from baseline
    assert state.files["AGGR.md"].stale is True


# ─── manifest miss ─────────────────────────────────────────────────────────


def test_file_not_in_manifest_is_reported_as_setup_error(tmp_path):
    """A doc that exists on disk but has no manifest entry should appear in
    `stats.errors` (via the per-file setup try/except in fill.ingest), and
    sentinels must NOT be stamped."""
    root = _platform_root(
        tmp_path,
        files={
            "AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "UNTRACKED.md": _md("UNTRACKED", "## C\n\nx"),
        },
        manifest={"AGGR": {"sourceType": "language"}},  # UNTRACKED missing
    )
    code, stats = run(
        platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x",
    )
    assert code == EXIT_INGEST_ERRORS
    assert any("UNTRACKED" in err for err in stats.errors)
    # AGGR was still processed fine.
    state = load(root / ".rag" / "openai-state.json")
    assert state.files["AGGR.md"].stale is False
    assert state.last_indexed_docs_commit != "c1"  # NOT stamped (errors present)


# ─── vector_store_id resolution ────────────────────────────────────────────


def test_vector_store_id_override_persists_to_state(tmp_path):
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_first")

    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_first"

    # Second run with no override uses stored value.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c2", changed=["docs/en/AGGR.md"]))
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_first"


def test_vector_store_id_switch_clears_state_and_does_full_reindex(tmp_path):
    """Switching to a different VS must wipe state.files + engage forced
    full scan so the new (presumed empty) VS gets every section uploaded."""
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    # Baseline against vs_old.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_old")
    state_old = load(root / ".rag" / "openai-state.json")
    assert state_old.files["AGGR.md"]  # established
    assert state_old.vector_store_id == "vs_old"

    # Switch to vs_new. Old state must be wiped + a new full scan uploads
    # against the new (fresh) client.
    new_client = FakeVectorStoreClient()
    code, stats = run(
        platform_root=root, client=new_client,
        git=FakeGitRunner(head="c2"), vector_store_id_override="vs_new",
    )
    assert code == EXIT_OK
    assert stats.files_processed == 1  # full re-upload, not fast-pathed
    assert stats.files_fast_path_skipped == 0

    state_new = load(root / ".rag" / "openai-state.json")
    assert state_new.vector_store_id == "vs_new"
    # Sections were uploaded to the NEW client (not just rewired to the
    # old file_ids). Assert via new_client.upload_calls since the two
    # clients share the same fake-file-NN id scheme.
    assert len(new_client.upload_calls) == 1
    new_file_ids = {srec.file_id for srec in state_new.files["AGGR.md"].sections.values()}
    assert all(fid in {vs.file_id for vs in new_client.list_sections()} for fid in new_file_ids)


# ─── rename handling (round 1 review) ────────────────────────────────────────


def test_rename_drops_old_and_uploads_new(tmp_path):
    """A `git mv FOO.md BAR.md` shows up as old-deleted + new-added with
    --no-renames. The state should drop the FOO record and upload BAR."""
    root = _platform_root(
        tmp_path,
        files={"FOO.md": _md("FOO", "## A\n\nfoo")},
        manifest={
            "FOO": {"sourceType": "language"},
            "BAR": {"sourceType": "language"},
        },
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")

    # Simulate the rename on disk; git diff (with --no-renames) would report:
    #   deleted: docs/en/FOO.md
    #   added:   docs/en/BAR.md
    (root / "docs" / "en" / "FOO.md").rename(root / "docs" / "en" / "BAR.md")

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(
            head="c2",
            changed=["docs/en/BAR.md"],
            removed=["docs/en/FOO.md"],
        ),
    )
    assert code == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert "FOO.md" not in state.files  # old record dropped
    assert "BAR.md" in state.files       # new file ingested


# ─── setup errors (round 1 review) ───────────────────────────────────────────


def test_malformed_manifest_returns_setup_error(tmp_path):
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    # Clobber the manifest with non-JSON.
    (root / "docs" / "manifest.json").write_text("not json {{", encoding="utf-8")

    code, _ = run(
        platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x",
    )
    assert code == EXIT_SETUP_ERROR
    # State should not have been mutated by a half-run.
    state_path = root / ".rag" / "openai-state.json"
    assert not state_path.exists()


def test_manifest_with_non_object_root_returns_setup_error(tmp_path):
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    (root / "docs" / "manifest.json").write_text("[]", encoding="utf-8")

    code, _ = run(
        platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x",
    )
    assert code == EXIT_SETUP_ERROR


# ─── main() smoke (CLI/env vs_id resolution) ─────────────────────────────────


def _init_git_repo(root: Path) -> str:
    """Make `root` a git repo with one commit so `main()`'s real GitRunner
    can call `git rev-parse HEAD`. Returns the HEAD sha."""
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()


def test_main_uses_stored_vector_store_id_when_env_unset(tmp_path, monkeypatch):
    """Once state has a vs_id, neither --vector-store-id nor RAG_VECTOR_STORE_ID
    is required. This is the steady-state Jenkins flow."""
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    head_sha = _init_git_repo(root)
    # Bootstrap state via run(), stamping HEAD as the baseline commit so
    # main()'s subsequent git diff base..HEAD has a real base to use.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head=head_sha), vector_store_id_override="vs_persisted")

    monkeypatch.delenv("RAG_VECTOR_STORE_ID", raising=False)

    from tools.rag_ingest_docs import main
    rc = main([
        "--platform-root", str(root),
        "--dry-run",
    ])
    assert rc == EXIT_OK


def test_main_first_run_dry_run_works_without_any_vs_id(tmp_path, monkeypatch):
    """First-run --dry-run with no CLI/env/state vs_id must succeed and
    persist a sentinel into state.json (steady-state assumption: a
    successful run always leaves vs_id populated)."""
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    _init_git_repo(root)
    monkeypatch.delenv("RAG_VECTOR_STORE_ID", raising=False)

    from tools.rag_ingest_docs import main
    rc = main(["--platform-root", str(root), "--dry-run"])
    assert rc == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id  # populated (sentinel or otherwise)


def test_main_returns_setup_error_when_no_vs_id_anywhere(tmp_path, monkeypatch):
    """First run with nothing set + no --dry-run → setup error."""
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    _init_git_repo(root)
    monkeypatch.delenv("RAG_VECTOR_STORE_ID", raising=False)

    from tools.rag_ingest_docs import main
    rc = main(["--platform-root", str(root)])
    assert rc == EXIT_SETUP_ERROR


def test_git_diff_call_uses_correct_base_and_subdir(tmp_path):
    """Pin the exact arguments to git.changed_under so a refactor can't
    silently widen the diff to docs/ru/ or use the wrong base."""
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    # Baseline.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="commit-base"), vector_store_id_override="vs_x")

    git = FakeGitRunner(head="commit-new")
    run(platform_root=root, client=FakeVectorStoreClient(), git=git)
    assert git.changed_calls == [("commit-base", "commit-new", "docs/en")]


def test_stale_but_missing_entry_remains_in_state(tmp_path):
    """A stale entry whose file no longer exists doesn't trigger an error
    but also isn't auto-removed; reconcile or a later git-diff handles it."""
    root = _platform_root(
        tmp_path,
        files={"AGGR.md": _md("AGGR", "## S\n\nx")},
        manifest={"AGGR": {"sourceType": "language"}},
    )
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x")

    state_path = root / ".rag" / "openai-state.json"
    state = load(state_path)
    from fill.state import FileRecord, save as save_state
    state.files["DELETED.md"] = FileRecord(stale=True)
    save_state(state_path, state)

    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c2"))

    state_after = load(state_path)
    assert "DELETED.md" in state_after.files  # entry persists until cleaned
