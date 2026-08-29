"""Tests for tools/rag_ingest_docs.py — the Jenkins driver script.

Sets up a fake `platform_root` (with `docs/<category>/` and `.rag/`) in a
tmp dir, injects a `FakeGitRunner` so no real git calls happen, and runs the
orchestration core (`run()`) end-to-end against a `FakeVectorStoreClient`.

Category comes from the first folder under `docs/` (language/paradigm/
how-to/brief/rules) — there is no manifest. State keys (source_file) are the
path relative to `docs`, e.g. `language/AGGR.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from fill.openai_client import FakeVectorStoreClient
from fill.state import State, load, save
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


def _platform_root(tmp_path: Path, *, files: dict[str, str]) -> Path:
    """Build a fake platform repo at tmp_path with the TYPE-FIRST layout.

    `files` keys are LOGICAL source keys "<type>/<stem>.md" (e.g.
    `language/AGGR.md`) — the same keys the vector store uses. Physically each
    is written to docs/<type>/en/<stem>.md. A key with no "/" (e.g. `STRAY.md`)
    is written directly under docs/ to simulate a misplaced, untyped file.
    """
    root = tmp_path / "platform"
    docs = root / "docs"
    docs.mkdir(parents=True)
    for relpath, body in files.items():
        if "/" in relpath:
            t, _, name = relpath.partition("/")
            _write(docs / t / "en" / name, body)
        else:
            _write(docs / relpath, body)
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


def test_returns_setup_error_when_no_vector_store_id(tmp_path):
    root = _platform_root(tmp_path, files={"language/AGGR.md": _md("AGGR", "## S\n\nx")})
    code, _ = run(
        platform_root=root,
        client=FakeVectorStoreClient(),
        git=FakeGitRunner(),
        vector_store_id_override=None,  # not in state either
    )
    assert code == EXIT_SETUP_ERROR


# ─── forced full scan (first run) ──────────────────────────────────────────


def test_forced_full_scan_removes_docs_the_ledger_still_tracks(tmp_path):
    """A doc deleted or renamed while the ledger was stale must not survive a
    forced full scan: there is no base commit to diff, so the removals come
    from the ledger vs what is on disk. The orphan sweep cannot catch these —
    it only looks at files the ledger does NOT know."""
    root = _platform_root(
        tmp_path,
        files={"rules/Rules_execution.md": _md("Rules execution", "## S\n\nfoo")},
    )
    client = FakeVectorStoreClient()

    # First cycle indexes the doc under its old name.
    old_root = _platform_root(
        tmp_path / "old",
        files={"rules/Rules_physical_model.md": _md("Rules physical model", "## S\n\nfoo")},
    )
    code, _ = run(platform_root=old_root, client=client, git=FakeGitRunner(head="c1"),
                  vector_store_id_override="vs_x")
    assert code == EXIT_OK
    state_path = old_root / ".rag" / "openai-state.json"
    assert "rules/Rules_physical_model.md" in load(state_path).files

    # Carry that ledger over to the renamed tree and force a full scan.
    (root / ".rag").mkdir(parents=True, exist_ok=True)
    (root / ".rag" / "openai-state.json").write_text(
        state_path.read_text(encoding="utf-8"), encoding="utf-8")
    st = load(root / ".rag" / "openai-state.json")
    st.pipeline_versions = {"chunker_version": "stale"}
    save(root / ".rag" / "openai-state.json", st)

    code, stats = run(platform_root=root, client=client, git=FakeGitRunner(head="c2"),
                      vector_store_id_override="vs_x")

    assert code == EXIT_OK
    assert stats.files_removed == 1
    assert "rules/Rules_physical_model.md" not in load(root / ".rag" / "openai-state.json").files


def test_first_run_does_forced_full_scan_and_stamps_sentinels(tmp_path):
    root = _platform_root(
        tmp_path,
        files={
            "language/AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "paradigm/OTHER.md": _md("OTHER", "## B\n\nbar"),
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
    assert "language/AGGR.md" in state.files
    assert "paradigm/OTHER.md" in state.files
    # sourceType is derived from the folder, not a manifest.
    assert state.files["language/AGGR.md"].indexed_sourceType == "language"
    assert state.files["paradigm/OTHER.md"].indexed_sourceType == "paradigm"


# ─── incremental (git diff) ────────────────────────────────────────────────


def test_incremental_uses_git_diff_after_baseline(tmp_path):
    root = _platform_root(
        tmp_path,
        files={
            "language/AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "paradigm/OTHER.md": _md("OTHER", "## B\n\nbar"),
            "language/QUIET.md": _md("QUIET", "## C\n\nbaz"),
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
        git=FakeGitRunner(head="commit-002", changed=["docs/language/en/AGGR.md"]),
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
        files={"language/AGGR.md": _md("AGGR", "## S\n\noriginal")},
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")
    initial_uploads = len(client.upload_calls)

    # Actually change the content.
    (root / "docs" / "language" / "en" / "AGGR.md").write_text(
        _md("AGGR", "## S\n\nMODIFIED"), encoding="utf-8")

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(head="c2", changed=["docs/language/en/AGGR.md"]),
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
            "language/AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "paradigm/OTHER.md": _md("OTHER", "## B\n\nbar"),
        },
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")

    # Hand-edit state to mark OTHER stale.
    state_path = root / ".rag" / "openai-state.json"
    state = load(state_path)
    state.files["paradigm/OTHER.md"].stale = True
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
    assert state_after.files["paradigm/OTHER.md"].stale is False


def test_stale_but_missing_file_is_skipped(tmp_path):
    """A stale entry whose underlying file no longer exists isn't processed
    via files_to_process (it'd error). It stays in state until a later run
    sees it in files_removed (or reconcile handles it)."""
    root = _platform_root(
        tmp_path,
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")

    # Manually inject a stale-but-missing file into state.
    state_path = root / ".rag" / "openai-state.json"
    state = load(state_path)
    from fill.state import FileRecord, save as save_state
    state.files["language/DELETED.md"] = FileRecord(stale=True)
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
            "language/AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "paradigm/OTHER.md": _md("OTHER", "## B\n\nbar"),
        },
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")

    # Actually delete the file on disk, mirror the git diff.
    (root / "docs" / "paradigm" / "en" / "OTHER.md").unlink()

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(head="c2", removed=["docs/paradigm/en/OTHER.md"]),
    )
    assert code == EXIT_OK
    assert stats.files_removed == 1
    state = load(root / ".rag" / "openai-state.json")
    assert "paradigm/OTHER.md" not in state.files


# ─── errors leave sentinels unstamped ──────────────────────────────────────


def test_errors_do_not_stamp_sentinels(tmp_path):
    """If any per-file error occurred, last_indexed_docs_commit and
    pipeline_versions must NOT be updated. The next run reprocesses."""
    root = _platform_root(
        tmp_path,
        files={"language/AGGR.md": _md("AGGR", "## S\n\nfoo")},
    )
    # Establish baseline.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x")

    # Now force an upload error on the next run.
    client = FakeVectorStoreClient()
    (root / "docs" / "language" / "en" / "AGGR.md").write_text(
        _md("AGGR", "## S\n\nchanged content"), encoding="utf-8")
    client.fail_upload_for_section_id.add("AGGR::s")

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(head="c2", changed=["docs/language/en/AGGR.md"]),
    )
    assert code == EXIT_INGEST_ERRORS
    assert stats.errors  # non-empty

    state = load(root / ".rag" / "openai-state.json")
    # Sentinels must still point at c1 / old versions — pinned to the exact
    # baseline value, not just "not c2".
    assert state.last_indexed_docs_commit == "c1"
    assert state.pipeline_versions is not None  # also pinned from baseline
    assert state.files["language/AGGR.md"].stale is True


# ─── category-folder errors (replaces the old manifest-miss tests) ──────────


def test_untyped_file_is_ignored_not_ingested(tmp_path):
    """A misplaced .md outside docs/<type>/en/ (here directly under docs/) is
    not part of the ingested English slice, so it is silently ignored — no
    error, not indexed. The valid sibling is processed and sentinels stamp.
    (Type-first scopes the glob to docs/<type>/en; the old layout instead
    errored on an uncategorized docs/en/STRAY.md.)"""
    root = _platform_root(
        tmp_path,
        files={
            "language/AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "STRAY.md": _md("STRAY", "## C\n\nx"),  # docs/STRAY.md — not under <type>/en
        },
    )
    code, stats = run(
        platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x",
    )
    assert code == EXIT_OK
    assert stats.errors == []
    state = load(root / ".rag" / "openai-state.json")
    assert "language/AGGR.md" in state.files
    assert "STRAY.md" not in state.files
    assert state.last_indexed_docs_commit == "c1"  # clean run → stamped


def test_unknown_type_folder_is_ignored(tmp_path):
    """A folder that is not one of the five types (docs/bogus/en/X.md) is not
    part of the ingested slice → ignored, no error."""
    root = _platform_root(
        tmp_path,
        files={
            "language/AGGR.md": _md("AGGR", "## S\n\nfoo"),
            "bogus/X.md": _md("X", "## C\n\nx"),  # docs/bogus/en/X.md — 'bogus' not a type
        },
    )
    code, stats = run(
        platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x",
    )
    assert code == EXIT_OK
    assert stats.errors == []
    state = load(root / ".rag" / "openai-state.json")
    assert "bogus/X.md" not in state.files


def test_brief_and_rules_folders_are_valid_sourcetypes(tmp_path):
    """brief/ and rules/ are first-class categories (Phase-1 single files)."""
    root = _platform_root(
        tmp_path,
        files={
            "brief/Brief.md": _md("Brief", "## Map\n\noverview"),
            "rules/Workflow.md": _md("Workflow", "## Rule\n\ndo this"),
        },
    )
    code, stats = run(
        platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x",
    )
    assert code == EXIT_OK
    assert stats.errors == []
    state = load(root / ".rag" / "openai-state.json")
    assert state.files["brief/Brief.md"].indexed_sourceType == "brief"
    assert state.files["rules/Workflow.md"].indexed_sourceType == "rules"


# ─── vector_store_id resolution ────────────────────────────────────────────


def test_vector_store_id_override_persists_to_state(tmp_path):
    root = _platform_root(
        tmp_path,
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
    )
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_first")

    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_first"

    # Second run with no override uses stored value.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c2", changed=["docs/language/en/AGGR.md"]))
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_first"


def test_vector_store_id_switch_clears_state_and_does_full_reindex(tmp_path):
    """Switching to a different VS must wipe state.files + engage forced
    full scan so the new (presumed empty) VS gets every section uploaded."""
    root = _platform_root(
        tmp_path,
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
    )
    # Baseline against vs_old.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_old")
    state_old = load(root / ".rag" / "openai-state.json")
    assert state_old.files["language/AGGR.md"]  # established
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
    new_file_ids = {srec.file_id for srec in state_new.files["language/AGGR.md"].sections.values()}
    assert all(fid in {vs.file_id for vs in new_client.list_sections()} for fid in new_file_ids)


# ─── rename handling ─────────────────────────────────────────────────────────


def test_rename_drops_old_and_uploads_new(tmp_path):
    """A `git mv FOO.md BAR.md` shows up as old-deleted + new-added with
    --no-renames. The state should drop the FOO record and upload BAR."""
    root = _platform_root(
        tmp_path,
        files={"language/FOO.md": _md("FOO", "## A\n\nfoo")},
    )
    client = FakeVectorStoreClient()
    run(platform_root=root, client=client, git=FakeGitRunner(head="c1"),
        vector_store_id_override="vs_x")

    # Simulate the rename on disk; git diff (with --no-renames) would report:
    #   deleted: docs/language/en/FOO.md
    #   added:   docs/language/en/BAR.md
    lang = root / "docs" / "language" / "en"
    (lang / "FOO.md").rename(lang / "BAR.md")

    code, stats = run(
        platform_root=root, client=client,
        git=FakeGitRunner(
            head="c2",
            changed=["docs/language/en/BAR.md"],
            removed=["docs/language/en/FOO.md"],
        ),
    )
    assert code == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert "language/FOO.md" not in state.files  # old record dropped
    assert "language/BAR.md" in state.files       # new file ingested


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
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
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
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
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
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
    )
    _init_git_repo(root)
    monkeypatch.delenv("RAG_VECTOR_STORE_ID", raising=False)

    from tools.rag_ingest_docs import main
    rc = main(["--platform-root", str(root)])
    assert rc == EXIT_SETUP_ERROR


def test_git_diff_call_uses_correct_base_and_subdir(tmp_path):
    """Pin the exact arguments to git.changed_under so a refactor can't
    silently widen the diff to docs/ru/ or use the wrong base. The subdir
    stays `docs` (nested category folders match that prefix)."""
    root = _platform_root(
        tmp_path,
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
    )
    # Baseline.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="commit-base"), vector_store_id_override="vs_x")

    git = FakeGitRunner(head="commit-new")
    run(platform_root=root, client=FakeVectorStoreClient(), git=git)
    assert git.changed_calls == [("commit-base", "commit-new", "docs")]


def test_no_op_ingest_does_not_advance_last_commit(tmp_path):
    """Critical: a fast-path-only ingest (nothing uploaded/deleted/removed)
    MUST NOT update `last_indexed_docs_commit`. Stamping it would change
    state.json, the Jenkins wrapper would commit that change, and the
    commit would itself trigger another webhook → infinite loop (observed
    in production after build #10 — 313 builds in 90 min)."""
    root = _platform_root(
        tmp_path,
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
    )
    # Establish baseline so state has a stamped last_commit.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="commit-A"), vector_store_id_override="vs_x")
    state_after_baseline = load(root / ".rag" / "openai-state.json")
    assert state_after_baseline.last_indexed_docs_commit == "commit-A"

    # Re-run with a NEW head but the docs unchanged — all files fast-path skip.
    code, stats = run(
        platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="commit-B"),
    )
    assert code == EXIT_OK
    # No real work: all uploads/deletes/removals zero.
    assert stats.sections_uploaded == 0
    assert stats.sections_deleted == 0
    assert stats.files_removed == 0

    # last_commit NOT advanced — stays at "commit-A" even though current
    # head is "commit-B". This keeps state.json byte-identical so the
    # Jenkins wrapper's git-diff check skips the commit.
    state_after_noop = load(root / ".rag" / "openai-state.json")
    assert state_after_noop.last_indexed_docs_commit == "commit-A"


def test_no_op_ingest_stamps_pipeline_versions(tmp_path):
    """Companion to the previous test: while `last_indexed_docs_commit`
    must NOT advance on no-op runs (to break the webhook loop),
    `pipeline_versions` MUST advance on clean runs — otherwise a version
    bump that happens to leave every section_payload_hash unchanged would
    leave the sentinel as `None`, forcing forced-full-scan on every
    future ingest."""
    root = _platform_root(
        tmp_path,
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
    )
    # Baseline.
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="commit-A"), vector_store_id_override="vs_x")

    # Hand-set pipeline_versions to None (simulate fresh sentinel after
    # version drift) and re-run with no changes.
    from fill.state import save as save_state
    state = load(root / ".rag" / "openai-state.json")
    state.pipeline_versions = None
    save_state(root / ".rag" / "openai-state.json", state)

    code, stats = run(
        platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="commit-B"),
    )
    assert code == EXIT_OK
    assert stats.sections_uploaded == 0  # no real work

    state_after = load(root / ".rag" / "openai-state.json")
    # `pipeline_versions` IS stamped (otherwise forever-forced-full-scan).
    assert state_after.pipeline_versions is not None
    # `last_indexed_docs_commit` is NOT advanced.
    assert state_after.last_indexed_docs_commit == "commit-A"


def test_entry_whose_file_is_gone_is_removed_even_without_a_diff(tmp_path):
    """A ledger entry whose file no longer exists is removed on the next
    cycle, whether or not the git diff of that window mentions it — the
    deletion may have happened while the ledger was behind. Before this the
    entry survived, and with it the doc's sections in the store."""
    root = _platform_root(
        tmp_path,
        files={"language/AGGR.md": _md("AGGR", "## S\n\nx")},
    )
    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c1"), vector_store_id_override="vs_x")

    state_path = root / ".rag" / "openai-state.json"
    state = load(state_path)
    from fill.state import FileRecord, save as save_state
    state.files["language/DELETED.md"] = FileRecord(stale=True)
    save_state(state_path, state)

    run(platform_root=root, client=FakeVectorStoreClient(),
        git=FakeGitRunner(head="c2"))

    state_after = load(state_path)
    assert "language/DELETED.md" not in state_after.files
