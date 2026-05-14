"""Tests for fill.real_openai_client.

Uses a hand-rolled `MockOpenAI` (instead of unittest.mock) so call shapes
are typed and assertion failures point at the actual misuse, not generic
"call wasn't made". The OpenAI SDK is never imported in this file — the
wrapper's lazy imports are only triggered inside `delete_section` via the
NotFoundError translation, which we cover by installing a tiny shim
module before that test.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

from fill.openai_client import VectorStoreSection
from fill.real_openai_client import OpenAIVectorStoreClient


# ─── Mock SDK objects ────────────────────────────────────────────────────────


@dataclass
class _MockFile:
    id: str


@dataclass
class _MockVSFile:
    id: str
    attributes: dict[str, Any] | None = None
    status: str = "completed"
    last_error: Any = None


@dataclass
class _MockPage:
    data: list[_MockVSFile]
    has_more: bool = False


class _MockFiles:
    """Mirrors `client.files`."""

    def __init__(self) -> None:
        self._next = 0
        self.create_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.delete_raises: dict[str, Exception] = {}
        self.create_raises: Exception | None = None

    def create(self, *, file: tuple, purpose: str) -> _MockFile:
        if self.create_raises:
            raise self.create_raises
        filename, body, mime = file
        self.create_calls.append({"filename": filename, "body": body, "mime": mime, "purpose": purpose})
        fid = f"file-{self._next:03d}"
        self._next += 1
        return _MockFile(id=fid)

    def delete(self, file_id: str) -> _MockFile:
        self.delete_calls.append(file_id)
        if file_id in self.delete_raises:
            raise self.delete_raises[file_id]
        return _MockFile(id=file_id)


class _MockVSFiles:
    """Mirrors `client.vector_stores.files`."""

    def __init__(self) -> None:
        self.create_and_poll_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[dict[str, Any]] = []
        self.create_and_poll_response: _MockVSFile | None = None
        self.create_and_poll_raises: Exception | None = None
        # Sequential responses for retry tests. Pops one entry per call.
        # Element can be an Exception (raised) or a _MockVSFile (returned).
        self.create_and_poll_side_effects: list[Any] = []
        self.delete_raises: dict[str, Exception] = {}
        self.list_pages: list[_MockPage] = []
        # Probe-retrieve hook for race-recovery testing. Default None →
        # the retrieve falls through to a stub return with status=missing
        # (so the production code's `if status == _TERMINAL_OK` is False).
        self.retrieve_response: _MockVSFile | None = None
        self.retrieve_raises: Exception | None = None

    def create_and_poll(self, **kwargs) -> _MockVSFile:
        self.create_and_poll_calls.append(kwargs)
        if self.create_and_poll_side_effects:
            item = self.create_and_poll_side_effects.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self.create_and_poll_raises:
            raise self.create_and_poll_raises
        if self.create_and_poll_response is None:
            return _MockVSFile(id=kwargs["file_id"], attributes=kwargs.get("attributes"))
        return self.create_and_poll_response

    def retrieve(self, **kwargs) -> _MockVSFile:
        self.retrieve_calls.append(kwargs)
        if self.retrieve_raises:
            raise self.retrieve_raises
        if self.retrieve_response is None:
            # Default: report unknown / not-yet-attached so the recovery
            # branch falls through to either retry or surface the original.
            return _MockVSFile(id=kwargs["file_id"], status="unknown")
        return self.retrieve_response

    def delete(self, **kwargs) -> Any:
        self.delete_calls.append(kwargs)
        fid = kwargs["file_id"]
        if fid in self.delete_raises:
            raise self.delete_raises[fid]
        return _MockFile(id=fid)

    def list(self, **kwargs) -> _MockPage:
        self.list_calls.append(kwargs)
        # Pop pages in order; if exhausted return an empty terminal page.
        if not self.list_pages:
            return _MockPage(data=[], has_more=False)
        return self.list_pages.pop(0)


class _MockVectorStores:
    def __init__(self) -> None:
        self.files = _MockVSFiles()


class MockOpenAI:
    """The bare minimum surface `OpenAIVectorStoreClient` touches."""

    def __init__(self) -> None:
        self.files = _MockFiles()
        self.vector_stores = _MockVectorStores()


# ─── NotFoundError shim ──────────────────────────────────────────────────────


def _install_openai_notfound_shim(monkeypatch):
    """Inject a fake `openai` module exposing `NotFoundError`.

    WHY a shim: `OpenAIVectorStoreClient.delete_section` lazy-imports
    `from openai import NotFoundError` so the test suite can run without
    the real SDK installed. We register a fake module in `sys.modules`
    that satisfies that import, then return the exception class so the
    test can `mock.delete_raises[fid] = NotFoundError(...)`.
    """

    class NotFoundError(Exception):
        pass

    fake_openai = types.ModuleType("openai")
    fake_openai.NotFoundError = NotFoundError
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return NotFoundError


# ─── construction ────────────────────────────────────────────────────────────


def test_rejects_empty_vector_store_id():
    with pytest.raises(ValueError, match="vector_store_id"):
        OpenAIVectorStoreClient("", client=MockOpenAI())


def test_stores_vector_store_id_and_poll_interval():
    mock = MockOpenAI()
    c = OpenAIVectorStoreClient("vs_xxx", client=mock, poll_interval_ms=2500)
    c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert mock.vector_stores.files.create_and_poll_calls[0]["poll_interval_ms"] == 2500
    assert mock.vector_stores.files.create_and_poll_calls[0]["vector_store_id"] == "vs_xxx"


# ─── upload_section ──────────────────────────────────────────────────────────


def test_upload_section_two_step_calls():
    mock = MockOpenAI()
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    fid = c.upload_section(
        content="hello world",
        filename="AGGR--AGGR__syntax.md",
        attributes={"section_id": "AGGR::syntax", "slug": "AGGR"},
    )

    assert fid == "file-000"
    # Step 1: Files.create
    assert mock.files.create_calls == [{
        "filename": "AGGR--AGGR__syntax.md",
        "body": b"hello world",
        "mime": "text/markdown",
        "purpose": "assistants",
    }]
    # Step 2: VS attach + poll
    assert mock.vector_stores.files.create_and_poll_calls == [{
        "vector_store_id": "vs_xxx",
        "file_id": "file-000",
        "attributes": {"section_id": "AGGR::syntax", "slug": "AGGR"},
        "poll_interval_ms": 1000,
    }]


def test_upload_section_attach_failure_deletes_orphan():
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_raises = RuntimeError("attach exploded")
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    with pytest.raises(RuntimeError, match="attach exploded"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})

    # The uploaded File must be cleaned up.
    assert mock.files.delete_calls == ["file-000"]


def test_upload_section_non_completed_status_raises_and_cleans_orphan():
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_response = _MockVSFile(
        id="file-000", status="failed", last_error={"code": "X", "message": "bad"}
    )
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    with pytest.raises(RuntimeError, match="status='failed'"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert mock.files.delete_calls == ["file-000"]


def test_upload_section_orphan_cleanup_failure_does_not_swallow_original_error():
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_raises = RuntimeError("attach exploded")
    mock.files.delete_raises["file-000"] = RuntimeError("cleanup also exploded")
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    # The ORIGINAL error must propagate; cleanup failure is logged, not raised.
    with pytest.raises(RuntimeError, match="attach exploded"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})


def test_upload_section_files_create_failure_propagates_without_attach():
    mock = MockOpenAI()
    mock.files.create_raises = RuntimeError("files API down")
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    with pytest.raises(RuntimeError, match="files API down"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert mock.vector_stores.files.create_and_poll_calls == []
    assert mock.files.delete_calls == []


# ─── delete_section ──────────────────────────────────────────────────────────


def test_delete_section_two_step_calls(monkeypatch):
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    c.delete_section("file-abc")

    assert mock.vector_stores.files.delete_calls == [
        {"vector_store_id": "vs_xxx", "file_id": "file-abc"}
    ]
    assert mock.files.delete_calls == ["file-abc"]


def test_delete_section_translates_vs_notfound_to_keyerror(monkeypatch):
    NotFoundError = _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    mock.vector_stores.files.delete_raises["file-abc"] = NotFoundError("nope")
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    with pytest.raises(KeyError, match="file-abc"):
        c.delete_section("file-abc")
    # Files.delete must NOT be attempted after VS detach reports NotFound.
    assert mock.files.delete_calls == []


def test_delete_section_swallows_files_notfound(monkeypatch):
    """VS detach succeeded but the File was already gone — that's the
    same end-state as a clean two-step, so we MUST NOT raise."""
    NotFoundError = _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    mock.files.delete_raises["file-abc"] = NotFoundError("gone")
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    c.delete_section("file-abc")  # should not raise

    assert mock.vector_stores.files.delete_calls == [
        {"vector_store_id": "vs_xxx", "file_id": "file-abc"}
    ]


def test_delete_section_other_errors_propagate(monkeypatch):
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    mock.vector_stores.files.delete_raises["file-abc"] = RuntimeError("server error")
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    with pytest.raises(RuntimeError, match="server error"):
        c.delete_section("file-abc")


# ─── list_sections / pagination ──────────────────────────────────────────────


def test_list_sections_single_page():
    mock = MockOpenAI()
    mock.vector_stores.files.list_pages = [
        _MockPage(data=[
            _MockVSFile(id="file-1", attributes={"section_id": "a"}),
            _MockVSFile(id="file-2", attributes={"section_id": "b"}),
        ], has_more=False),
    ]
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    result = c.list_sections()
    assert result == [
        VectorStoreSection(file_id="file-1", attributes={"section_id": "a"}),
        VectorStoreSection(file_id="file-2", attributes={"section_id": "b"}),
    ]
    # Single .list() call with no `after`.
    assert mock.vector_stores.files.list_calls == [
        {"vector_store_id": "vs_xxx", "limit": 100}
    ]


def test_list_sections_paginates_via_after_cursor():
    mock = MockOpenAI()
    mock.vector_stores.files.list_pages = [
        _MockPage(data=[_MockVSFile(id="file-1", attributes={"x": "1"})], has_more=True),
        _MockPage(data=[_MockVSFile(id="file-2", attributes={"x": "2"})], has_more=True),
        _MockPage(data=[_MockVSFile(id="file-3", attributes={"x": "3"})], has_more=False),
    ]
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)

    result = c.list_sections()
    assert [vs.file_id for vs in result] == ["file-1", "file-2", "file-3"]

    # First call no `after`, subsequent calls use the prior page's last id.
    assert mock.vector_stores.files.list_calls == [
        {"vector_store_id": "vs_xxx", "limit": 100},
        {"vector_store_id": "vs_xxx", "limit": 100, "after": "file-1"},
        {"vector_store_id": "vs_xxx", "limit": 100, "after": "file-2"},
    ]


def test_list_sections_handles_missing_attributes():
    mock = MockOpenAI()
    mock.vector_stores.files.list_pages = [
        _MockPage(data=[_MockVSFile(id="file-1", attributes=None)], has_more=False),
    ]
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    result = c.list_sections()
    assert result == [VectorStoreSection(file_id="file-1", attributes={})]


def test_list_sections_empty_vs():
    mock = MockOpenAI()
    # No pages preloaded → MockVSFiles.list returns an empty terminal page.
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    assert c.list_sections() == []


def test_list_sections_terminates_on_has_more_true_with_empty_data():
    """Defensive: if the server ever sends `has_more=True` with an empty
    page, we still terminate instead of looping forever."""
    mock = MockOpenAI()
    mock.vector_stores.files.list_pages = [
        _MockPage(data=[], has_more=True),
    ]
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    result = c.list_sections()
    assert result == []
    assert len(mock.vector_stores.files.list_calls) == 1


def test_list_sections_handles_none_data():
    """Defensive: `list(page.data or [])` must not raise if data is None."""
    mock = MockOpenAI()
    mock.vector_stores.files.list_pages = [_MockPage(data=None, has_more=False)]  # type: ignore[arg-type]
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    assert c.list_sections() == []


# ─── attribute fidelity ──────────────────────────────────────────────────────


def test_upload_section_passes_attributes_through_verbatim():
    mock = MockOpenAI()
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    attrs = {
        "section_id": "AGGR::syntax",
        "slug": "AGGR",
        "sourceType": "language",
        "section_name": "Syntax",
        "heading_path": "AGGR > Syntax",
        "source_url": "https://docs/AGGR",
        "section_payload_hash": "sha256:abc",
        "source_file": "AGGR.md",
        "file_hash": "sha256:def",
    }
    c.upload_section(content="x", filename="x.md", attributes=attrs)
    assert mock.vector_stores.files.create_and_poll_calls[0]["attributes"] == attrs


def test_upload_section_encodes_content_as_utf8():
    mock = MockOpenAI()
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    c.upload_section(content="héllo «foo»", filename="x.md", attributes={"section_id": "s"})
    body = mock.files.create_calls[0]["body"]
    assert isinstance(body, bytes)
    assert body.decode("utf-8") == "héllo «foo»"


# ─── status handling (round 1 review) ────────────────────────────────────────


def test_upload_section_in_progress_status_preserves_orphan():
    """`in_progress` is non-terminal — file might still be indexing.
    Must NOT delete; operator inspects manually."""
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_response = _MockVSFile(
        id="file-000", status="in_progress", last_error=None
    )
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    with pytest.raises(RuntimeError, match="orphan File preserved"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    # Critical: orphan NOT deleted.
    assert mock.files.delete_calls == []


def test_upload_section_unknown_status_preserves_orphan():
    """Defensive: unknown status (e.g. SDK change) → don't delete."""
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_response = _MockVSFile(
        id="file-000", status="ALIEN_STATUS", last_error=None
    )
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    with pytest.raises(RuntimeError, match="ALIEN_STATUS"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert mock.files.delete_calls == []


def test_upload_section_cancelled_status_deletes_orphan():
    """`cancelled` is terminal-failure — definitely clean up."""
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_response = _MockVSFile(
        id="file-000", status="cancelled", last_error=None
    )
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    with pytest.raises(RuntimeError, match="status='cancelled'"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert mock.files.delete_calls == ["file-000"]


def test_upload_section_missing_status_preserves_orphan():
    """SDK response without a `status` attr → treated as unknown → preserve."""
    mock = MockOpenAI()
    # Bare object that lacks `status` (and lacks `last_error`).
    class _Bare:
        id = "file-000"
    mock.vector_stores.files.create_and_poll_response = _Bare()  # type: ignore[assignment]
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    with pytest.raises(RuntimeError, match="orphan File preserved"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert mock.files.delete_calls == []


# ─── construction options ────────────────────────────────────────────────────


def test_custom_file_purpose():
    mock = MockOpenAI()
    c = OpenAIVectorStoreClient("vs_xxx", client=mock, file_purpose="user_data")
    c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert mock.files.create_calls[0]["purpose"] == "user_data"


def test_vector_store_id_property():
    c = OpenAIVectorStoreClient("vs_xxx", client=MockOpenAI())
    assert c.vector_store_id == "vs_xxx"


# ─── orphan cleanup observability ────────────────────────────────────────────


def test_orphan_cleanup_failure_is_logged(caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="fill.real_openai_client")
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_raises = RuntimeError("attach exploded")
    mock.files.delete_raises["file-000"] = RuntimeError("cleanup also exploded")
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    with pytest.raises(RuntimeError, match="attach exploded"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert any(
        "orphan File file-000" in record.getMessage() and "cleanup also exploded" in record.getMessage()
        for record in caplog.records
    ), f"expected orphan-cleanup warning in caplog, got: {[r.getMessage() for r in caplog.records]}"


# ─── delete_section step-2 propagates non-NotFoundError ──────────────────────


def test_delete_section_step2_non_notfound_propagates(monkeypatch):
    """If `files.delete` raises something OTHER than NotFoundError after VS
    detach succeeded, the error propagates (caller decides to retry)."""
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    mock.files.delete_raises["file-abc"] = RuntimeError("files API down")
    c = OpenAIVectorStoreClient("vs_xxx", client=mock)
    with pytest.raises(RuntimeError, match="files API down"):
        c.delete_section("file-abc")
    # VS detach DID happen first.
    assert mock.vector_stores.files.delete_calls == [
        {"vector_store_id": "vs_xxx", "file_id": "file-abc"}
    ]


# ─── attach 404 retry (eventual consistency recovery) ────────────────────────


def _build_attach_404(file_id: str, vs_id: str = "vs_xxx") -> Exception:
    """Construct an OpenAI NotFoundError that matches the production error
    surface (`Error code: 404 - {'error': {'message': 'No file found with
    id X in vector store Y.', ...}}`). The `_is_retryable_attach_not_found`
    pattern matcher keys off `file_id`, "No file found with id", and
    "vector store"."""
    from openai import NotFoundError  # injected by _install_openai_notfound_shim
    return NotFoundError(
        f"Error code: 404 - No file found with id '{file_id}' in vector store {vs_id}."
    )


def test_upload_section_retries_transient_attach_404(monkeypatch):
    """Transient 404 on first create_and_poll → wait → retry → success.
    No orphan File cleanup, no propagated error."""
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    err = _build_attach_404("file-000")
    mock.vector_stores.files.create_and_poll_side_effects = [
        err,
        _MockVSFile(id="file-000", status="completed"),
    ]
    sleeps: list[float] = []
    c = OpenAIVectorStoreClient(
        "vs_xxx",
        client=mock,
        attach_retry_delays_seconds=(0.5, 1.0),
        sleep=sleeps.append,
    )
    fid = c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert fid == "file-000"
    # Two create_and_poll attempts (first 404, second success).
    assert len(mock.vector_stores.files.create_and_poll_calls) == 2
    # Slept once between attempts.
    assert sleeps == [0.5]
    # No orphan deleted.
    assert mock.files.delete_calls == []


def test_upload_section_retrieve_short_circuits_retry(monkeypatch):
    """After the first backoff sleep, if retrieve sees the file as
    `completed`, return immediately without another create_and_poll call.
    Saves a redundant attach for the eventual-consistency win case."""
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    err = _build_attach_404("file-000")
    mock.vector_stores.files.create_and_poll_side_effects = [err]
    mock.vector_stores.files.retrieve_response = _MockVSFile(
        id="file-000", status="completed"
    )
    sleeps: list[float] = []
    c = OpenAIVectorStoreClient(
        "vs_xxx",
        client=mock,
        attach_retry_delays_seconds=(0.5, 1.0),
        sleep=sleeps.append,
    )
    fid = c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert fid == "file-000"
    # Single create_and_poll attempt, retrieve resolved after the first sleep.
    assert len(mock.vector_stores.files.create_and_poll_calls) == 1
    assert mock.vector_stores.files.retrieve_calls == [
        {"vector_store_id": "vs_xxx", "file_id": "file-000"}
    ]
    # Slept once (after 404), then retrieve won.
    assert sleeps == [0.5]
    assert mock.files.delete_calls == []


def test_upload_section_terminal_failed_via_retrieve_aborts_retry(monkeypatch):
    """If retrieve during a retry observes status='failed', stop retrying
    immediately — re-attaching wouldn't recover. The orphan File still
    gets cleaned up via upload_section's except branch."""
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    err = _build_attach_404("file-000")
    mock.vector_stores.files.create_and_poll_side_effects = [err, err]
    mock.vector_stores.files.retrieve_response = _MockVSFile(
        id="file-000", status="failed", last_error={"code": "indexing_error"}
    )
    sleeps: list[float] = []
    c = OpenAIVectorStoreClient(
        "vs_xxx",
        client=mock,
        attach_retry_delays_seconds=(0.1, 0.2),
        sleep=sleeps.append,
    )
    with pytest.raises(RuntimeError, match="status='failed'"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    # Only ONE attach attempt — retrieve aborted the loop after first sleep.
    assert len(mock.vector_stores.files.create_and_poll_calls) == 1
    assert sleeps == [0.1]
    # Orphan cleaned.
    assert mock.files.delete_calls == ["file-000"]


def test_upload_section_exhausts_retries_then_cleans_up(monkeypatch):
    """All retries return 404 AND retrieve never sees the file → final
    exception propagates, orphan File is deleted."""
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    err = _build_attach_404("file-000")
    mock.vector_stores.files.create_and_poll_side_effects = [err, err, err]
    mock.vector_stores.files.retrieve_raises = err
    sleeps: list[float] = []
    c = OpenAIVectorStoreClient(
        "vs_xxx",
        client=mock,
        attach_retry_delays_seconds=(0.1, 0.2),  # 2 retries → 3 attempts total
        sleep=sleeps.append,
    )
    with pytest.raises(Exception, match="No file found"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert len(mock.vector_stores.files.create_and_poll_calls) == 3
    assert sleeps == [0.1, 0.2]
    # Orphan File cleaned up.
    assert mock.files.delete_calls == ["file-000"]


def test_upload_section_does_not_retry_non_matching_404(monkeypatch):
    """A NotFoundError whose message doesn't match the eventual-consistency
    pattern (e.g. "Vector store not found" — wrong vs_id) must NOT be
    retried — that's a real configuration bug."""
    NotFoundError = _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_side_effects = [
        NotFoundError("Error code: 404 - Vector store not found")
    ]
    sleeps: list[float] = []
    c = OpenAIVectorStoreClient(
        "vs_xxx",
        client=mock,
        attach_retry_delays_seconds=(0.1, 0.2),
        sleep=sleeps.append,
    )
    with pytest.raises(Exception, match="Vector store not found"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert len(mock.vector_stores.files.create_and_poll_calls) == 1
    assert sleeps == []
    # Orphan cleaned (current behavior — attach raised, we delete the File).
    assert mock.files.delete_calls == ["file-000"]


def test_upload_section_does_not_retry_non_notfound(monkeypatch):
    """Random RuntimeError from attach is NOT a 404 → no retry, immediate
    cleanup + propagate."""
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    mock.vector_stores.files.create_and_poll_side_effects = [
        RuntimeError("backend exploded")
    ]
    sleeps: list[float] = []
    c = OpenAIVectorStoreClient(
        "vs_xxx",
        client=mock,
        attach_retry_delays_seconds=(0.1, 0.2),
        sleep=sleeps.append,
    )
    with pytest.raises(RuntimeError, match="backend exploded"):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    assert sleeps == []
    assert mock.files.delete_calls == ["file-000"]


def test_upload_section_retry_uses_exponential_backoff_schedule(monkeypatch):
    """The wait between retries follows the configured delay tuple in
    order — not constant, not randomized."""
    _install_openai_notfound_shim(monkeypatch)
    mock = MockOpenAI()
    err = _build_attach_404("file-000")
    mock.vector_stores.files.create_and_poll_side_effects = [err, err, err, err]
    mock.vector_stores.files.retrieve_raises = err
    sleeps: list[float] = []
    c = OpenAIVectorStoreClient(
        "vs_xxx",
        client=mock,
        attach_retry_delays_seconds=(1.0, 2.0, 4.0),  # 3 retries → 4 attempts
        sleep=sleeps.append,
    )
    with pytest.raises(Exception):
        c.upload_section(content="x", filename="x.md", attributes={"section_id": "s"})
    # 4 attempts (initial + 3 retries), 3 sleeps in the exact configured order.
    assert len(mock.vector_stores.files.create_and_poll_calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]
