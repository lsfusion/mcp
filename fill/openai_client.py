"""fill.openai_client — minimal protocol that `fill.ingest` and `fill.reconcile`
use to talk to an OpenAI Vector Store, plus an in-memory fake for tests.

The protocol intentionally hides the OpenAI SDK's two-step Files-API +
vector_stores.files.create_and_poll dance behind one `upload_section` call,
and the symmetric Files-API + vector_stores.files.delete dance behind
`delete_section`. Production code wraps the real SDK; tests use the fake.

Attribute values are constrained to JSON-scalar (`str | int | bool`) to
match OpenAI's Vector Store file-attribute API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


AttrValue = str | int | bool


@dataclass(frozen=True)
class VectorStoreSection:
    """One file's identity in a Vector Store, as observed by the client.
    Used by `fill.reconcile` to flatten what OpenAI has and compare to state.
    """

    file_id: str
    attributes: dict[str, AttrValue]


class VectorStoreClient(Protocol):
    """The operations `fill.ingest` and `fill.reconcile` need.

    Implementations are responsible for retry/backoff, polling for
    indexing completion, and surfacing fatal errors as exceptions.
    """

    def upload_section(
        self,
        *,
        content: str,
        filename: str,
        attributes: dict[str, AttrValue],
    ) -> str:
        """Upload a section's payload as one file + attach to the VS.
        Returns the OpenAI File id (`file-...`)."""

    def delete_section(self, file_id: str) -> None:
        """Detach the file from the VS and delete the File object.
        Production wrappers translate the SDK's 404-on-unknown-id into an
        exception so the caller observes the same behavior as the fake."""

    def list_sections(self) -> list[VectorStoreSection]:
        """Return all files currently in the Vector Store + their attributes.
        Used by `ragRebuildIndex --mode reconcile`."""


# ─── fake for tests ─────────────────────────────────────────────────────────


class FakeVectorStoreClient:
    """In-memory implementation. Records all calls for test assertions.

    Fault injection:
      - `fail_upload_for_section_id`: section_ids whose upload should raise.
      - `fail_delete_for_file_id`: file_ids whose delete should raise.
    """

    def __init__(self) -> None:
        self._files: dict[str, tuple[str, dict[str, AttrValue]]] = {}
        self._next_id = 0
        self.fail_upload_for_section_id: set[str] = set()
        self.fail_delete_for_file_id: set[str] = set()
        self.upload_calls: list[dict[str, object]] = []
        self.delete_calls: list[str] = []

    def upload_section(
        self,
        *,
        content: str,
        filename: str,
        attributes: dict[str, AttrValue],
    ) -> str:
        sid = str(attributes.get("section_id", ""))
        if sid and sid in self.fail_upload_for_section_id:
            raise RuntimeError(f"simulated upload failure for section_id={sid}")
        file_id = f"fake-file-{self._next_id}"
        self._next_id += 1
        self._files[file_id] = (content, dict(attributes))
        self.upload_calls.append(
            {"file_id": file_id, "filename": filename, "attributes": dict(attributes)}
        )
        return file_id

    def delete_section(self, file_id: str) -> None:
        if file_id in self.fail_delete_for_file_id:
            raise RuntimeError(f"simulated delete failure for file_id={file_id}")
        if file_id not in self._files:
            raise KeyError(f"unknown file_id {file_id!r}")
        del self._files[file_id]
        self.delete_calls.append(file_id)

    def list_sections(self) -> list[VectorStoreSection]:
        return [
            VectorStoreSection(file_id=fid, attributes=dict(attrs))
            for fid, (_, attrs) in self._files.items()
        ]

    # ─── test helpers ──────────────────────────────────────────────────────

    def stored_content(self, file_id: str) -> str:
        return self._files[file_id][0]

    def stored_attributes(self, file_id: str) -> dict[str, AttrValue]:
        return dict(self._files[file_id][1])

    @property
    def file_count(self) -> int:
        return len(self._files)

    def preload(self, file_id: str, attributes: dict[str, AttrValue], content: str = "") -> None:
        """Inject a VS entry with a chosen file_id, bypassing the upload log.
        For tests that reason about VS state (e.g. fill.reconcile) rather
        than how it got there."""
        self._files[file_id] = (content, dict(attributes))
