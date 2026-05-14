"""Tests for tools/rag_rebuild_index.py — the Jenkins ragRebuildIndex driver.

Reconcile is much simpler than ingest — no git diff, no manifest, no
files_to_process plan. The driver just plumbs vs_id resolution + state
load/save around `fill.reconcile.reconcile`.
"""

from __future__ import annotations

from pathlib import Path

from fill.openai_client import FakeVectorStoreClient
from fill.state import FileRecord, SectionRecord, State, load
from tools.rag_rebuild_index import EXIT_OK, EXIT_SETUP_ERROR, run


# ─── helpers ─────────────────────────────────────────────────────────────────


def _attrs(section_id: str, source_file: str = "AGGR.md") -> dict:
    return {
        "section_id": section_id,
        "section_payload_hash": f"sha256:{section_id}",
        "heading_path": f"X > {section_id}",
        "source_file": source_file,
        "sourceType": "language",
    }


def _platform_root(tmp_path: Path, *, state: State | None = None) -> Path:
    root = tmp_path / "platform"
    (root / ".rag").mkdir(parents=True)
    if state is not None:
        from fill.state import save as save_state
        save_state(root / ".rag" / "openai-state.json", state)
    return root


# ─── setup errors ──────────────────────────────────────────────────────────


def test_returns_setup_error_when_no_vector_store_id(tmp_path):
    root = _platform_root(tmp_path)  # empty state
    code, stats = run(
        platform_root=root,
        client=FakeVectorStoreClient(),
        vector_store_id_override=None,
    )
    assert code == EXIT_SETUP_ERROR


# ─── happy paths ─────────────────────────────────────────────────────────────


def test_reconcile_rebuilds_state_from_vs(tmp_path):
    root = _platform_root(tmp_path)
    client = FakeVectorStoreClient()
    client.preload("file-1", _attrs("AGGR::s", source_file="AGGR.md"))
    client.preload("file-2", _attrs("AGGR::e", source_file="AGGR.md"))

    code, stats = run(
        platform_root=root,
        client=client,
        vector_store_id_override="vs_x",
    )

    assert code == EXIT_OK
    assert stats.files_in_vs == 2
    assert stats.sections_rebuilt == 2

    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_x"
    assert state.last_indexed_docs_commit is None  # sentinel
    assert state.pipeline_versions is None         # sentinel
    rec = state.files["AGGR.md"]
    assert rec.stale is True
    assert rec.indexed_with is None
    assert set(rec.sections.keys()) == {"AGGR::s", "AGGR::e"}


def test_reconcile_empties_state_when_vs_is_empty(tmp_path):
    initial = State(
        vector_store_id="vs_x",
        last_indexed_docs_commit="abc",
        pipeline_versions={"x": "1"},
        files={"OLD.md": FileRecord(
            file_hash="h",
            sections={"OLD::a": SectionRecord(
                file_id="file-old",
                section_payload_hash="sha256:old",
                heading_path="o",
                source_file="OLD.md",
            )},
        )},
    )
    root = _platform_root(tmp_path, state=initial)
    code, stats = run(
        platform_root=root,
        client=FakeVectorStoreClient(),  # empty VS
    )
    assert code == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert state.files == {}
    assert state.vector_store_id == "vs_x"  # preserved
    assert state.last_indexed_docs_commit is None
    assert state.pipeline_versions is None


# ─── vector_store_id flow ────────────────────────────────────────────────────


def test_uses_stored_vs_id_when_override_none(tmp_path):
    initial = State(vector_store_id="vs_persisted")
    root = _platform_root(tmp_path, state=initial)
    client = FakeVectorStoreClient()
    client.preload("file-1", _attrs("AGGR::s"))

    code, _ = run(platform_root=root, client=client, vector_store_id_override=None)
    assert code == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_persisted"


def test_override_switches_vs_id_without_pre_wipe(tmp_path):
    """Reconcile itself wipes state.files — the driver only needs to set
    the new vs_id; no pre-wipe required (unlike ragIngestDocs)."""
    initial = State(
        vector_store_id="vs_old",
        files={"AGGR.md": FileRecord(file_hash="h")},
    )
    root = _platform_root(tmp_path, state=initial)
    client = FakeVectorStoreClient()
    client.preload("file-1", _attrs("X::s", source_file="X.md"))

    code, _ = run(
        platform_root=root,
        client=client,
        vector_store_id_override="vs_new",
    )
    assert code == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_new"
    # Old state.files["AGGR.md"] gone (reconcile rebuilt from VS, which only
    # has X.md).
    assert "AGGR.md" not in state.files
    assert "X.md" in state.files


# ─── stats surface ───────────────────────────────────────────────────────────


def test_malformed_and_duplicates_appear_in_stats(tmp_path):
    root = _platform_root(tmp_path)
    client = FakeVectorStoreClient()
    # well-formed slot AGGR::s with two candidates → smallest wins
    client.preload("file-zz", _attrs("AGGR::s"))
    client.preload("file-aa", _attrs("AGGR::s"))
    # malformed: section_id missing
    bad = _attrs("AGGR::broken")
    del bad["section_id"]
    client.preload("file-bad", bad)

    code, stats = run(platform_root=root, client=client, vector_store_id_override="vs_x")
    assert code == EXIT_OK
    assert stats.malformed_in_vs == ["file-bad"]
    assert len(stats.duplicate_slots) == 1
    src, sid, losers = stats.duplicate_slots[0]
    assert (src, sid) == ("AGGR.md", "AGGR::s")
    assert losers == ["file-zz"]  # file-aa won

    state = load(root / ".rag" / "openai-state.json")
    assert state.files["AGGR.md"].sections["AGGR::s"].file_id == "file-aa"


# ─── main() smoke ────────────────────────────────────────────────────────────


def test_main_dry_run_first_run(tmp_path, monkeypatch):
    root = _platform_root(tmp_path)
    monkeypatch.delenv("RAG_VECTOR_STORE_ID", raising=False)

    from tools.rag_rebuild_index import main
    rc = main(["--platform-root", str(root), "--dry-run"])
    assert rc == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id  # sentinel persisted


def test_main_setup_error_when_no_vs_id_anywhere(tmp_path, monkeypatch):
    root = _platform_root(tmp_path)
    monkeypatch.delenv("RAG_VECTOR_STORE_ID", raising=False)

    from tools.rag_rebuild_index import main
    rc = main(["--platform-root", str(root)])
    assert rc == EXIT_SETUP_ERROR


def test_main_uses_env_vs_id(tmp_path, monkeypatch):
    root = _platform_root(tmp_path)
    monkeypatch.setenv("RAG_VECTOR_STORE_ID", "vs_from_env")

    from tools.rag_rebuild_index import main
    rc = main(["--platform-root", str(root), "--dry-run"])
    assert rc == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_from_env"


# ─── error propagation ───────────────────────────────────────────────────────


def test_main_returns_setup_error_when_reconcile_raises(tmp_path, monkeypatch):
    """If the VectorStoreClient raises during list_sections, main() must
    catch and return EXIT_SETUP_ERROR, not crash."""
    root = _platform_root(tmp_path)
    monkeypatch.setenv("RAG_VECTOR_STORE_ID", "vs_x")

    import tools.rag_rebuild_index as drv

    class BoomClient:
        def list_sections(self):
            raise RuntimeError("simulated API failure")
        # Other Protocol methods irrelevant; reconcile only calls list_sections.

    monkeypatch.setattr(drv, "_make_client", lambda args, vs_id: BoomClient())

    rc = drv.main(["--platform-root", str(root)])
    assert rc == EXIT_SETUP_ERROR


# ─── recovery from corrupt state (round 1 review) ───────────────────────────


def test_main_recovers_from_corrupt_state_json(tmp_path, monkeypatch):
    """The whole point of reconcile is to recover from a broken state.
    Corrupt JSON must NOT block the run — main() should warn, treat state
    as empty, and rebuild from VS."""
    root = _platform_root(tmp_path)
    (root / ".rag" / "openai-state.json").write_text("not json {{", encoding="utf-8")

    # vs_id has to come from somewhere; corrupt state lost the persisted one.
    monkeypatch.setenv("RAG_VECTOR_STORE_ID", "vs_recovery")

    import tools.rag_rebuild_index as drv

    class CannedClient:
        def list_sections(self):
            from fill.openai_client import VectorStoreSection
            return [VectorStoreSection(
                file_id="file-1",
                attributes={
                    "section_id": "AGGR::s",
                    "section_payload_hash": "sha256:s",
                    "heading_path": "AGGR > S",
                    "source_file": "AGGR.md",
                    "sourceType": "language",
                },
            )]

    monkeypatch.setattr(drv, "_make_client", lambda args, vs_id: CannedClient())
    rc = drv.main(["--platform-root", str(root)])
    assert rc == EXIT_OK

    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_recovery"
    assert "AGGR.md" in state.files


# ─── vs_id resolution: env is fallback, not override ────────────────────────


def test_env_does_not_override_persisted_state_vs_id(tmp_path, monkeypatch):
    """When state already has a vs_id, env must NOT silently switch it.
    Jenkins injects RAG_VECTOR_STORE_ID on every run; treating it as
    override would re-point state every build."""
    initial = State(vector_store_id="vs_persisted")
    root = _platform_root(tmp_path, state=initial)
    monkeypatch.setenv("RAG_VECTOR_STORE_ID", "vs_different_env_value")

    from tools.rag_rebuild_index import main
    rc = main(["--platform-root", str(root), "--dry-run"])
    assert rc == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    # State's persisted vs_id wins over the env.
    assert state.vector_store_id == "vs_persisted"


def test_cli_overrides_persisted_state_vs_id(tmp_path, monkeypatch):
    """Explicit --vector-store-id is the ONLY way to switch from a
    persisted state's vs_id (env is fallback-only)."""
    initial = State(vector_store_id="vs_persisted")
    root = _platform_root(tmp_path, state=initial)
    monkeypatch.delenv("RAG_VECTOR_STORE_ID", raising=False)

    from tools.rag_rebuild_index import main
    rc = main([
        "--platform-root", str(root),
        "--vector-store-id", "vs_explicit",
        "--dry-run",
    ])
    assert rc == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_explicit"


def test_dry_run_with_existing_state_preserves_vs_id(tmp_path, monkeypatch):
    """Dry-run sentinel must NOT fire when state already has a vs_id."""
    initial = State(vector_store_id="vs_real")
    root = _platform_root(tmp_path, state=initial)
    monkeypatch.delenv("RAG_VECTOR_STORE_ID", raising=False)

    from tools.rag_rebuild_index import main
    rc = main(["--platform-root", str(root), "--dry-run"])
    assert rc == EXIT_OK
    state = load(root / ".rag" / "openai-state.json")
    assert state.vector_store_id == "vs_real"
