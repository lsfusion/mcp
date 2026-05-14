"""fill.real_openai_client — production wrapper around the OpenAI SDK that
implements `fill.openai_client.VectorStoreClient`.

The SDK handles transient-error retry and timeout (configured at client
construction). This wrapper adds:

  • Two-step upload (Files.create + VS attach-and-poll) with orphan cleanup:
    if the attach step fails after the file was uploaded, we delete the
    orphan File before re-raising so `state.json` doesn't drift relative
    to the VS.

  • `list_sections` pages explicitly through `after` cursors so a partial
    listing can never corrupt reconcile.

  • `NotFoundError → KeyError` translation on `delete_section`, matching
    the `FakeVectorStoreClient` contract that tests assert against.

The `openai` package is imported lazily so unit tests that pass a mock
`client=` argument don't require it to be installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fill.openai_client import AttrValue, VectorStoreSection

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI


log = logging.getLogger(__name__)

# OpenAI VS file statuses observed from `create_and_poll`:
#   "completed"   → indexed and queryable. The only success status.
#   "failed"      → indexing rejected (e.g. bad MIME, content too large).
#   "cancelled"   → cancelled by an out-of-band operation.
#   "in_progress" → still indexing. Should be impossible from a fully-polled
#                   `create_and_poll`, but observed in practice when the SDK's
#                   internal poll budget is exhausted.
_TERMINAL_OK = "completed"
_TERMINAL_FAILED = frozenset({"failed", "cancelled"})


class OpenAIVectorStoreClient:
    """Production `VectorStoreClient` against the OpenAI SDK.

    Args:
        vector_store_id: target VS id ("vs_...").
        client: a constructed `openai.OpenAI` instance. If omitted, one is
            constructed lazily via `OpenAI()`, which reads `OPENAI_API_KEY`
            (and friends) from the environment. Pass an explicit client to
            customize `timeout` / `max_retries` / `base_url`.
        poll_interval_ms: how often `create_and_poll` checks indexing
            status. Lives on this wrapper (not on the OpenAI client) because
            it's specific to VS file ingestion, not to general HTTP behavior.
        file_purpose: OpenAI Files API `purpose` for uploaded payloads.
            Default `"assistants"` matches today's Vector Store ingestion
            contract; expose as a knob so an operator can switch to
            `"user_data"` if OpenAI changes the accepted purpose.
    """

    def __init__(
        self,
        vector_store_id: str,
        *,
        client: Any | None = None,
        poll_interval_ms: int = 1000,
        file_purpose: str = "assistants",
    ) -> None:
        if not vector_store_id:
            raise ValueError("vector_store_id must be non-empty")
        if client is None:
            from openai import OpenAI  # lazy: tests bypass this branch
            client = OpenAI()
        self._client = client
        self._vs_id = vector_store_id
        self._poll_interval_ms = poll_interval_ms
        self._file_purpose = file_purpose

    @property
    def vector_store_id(self) -> str:
        return self._vs_id

    # ─── VectorStoreClient protocol ────────────────────────────────────────

    def upload_section(
        self,
        *,
        content: str,
        filename: str,
        attributes: dict[str, AttrValue],
    ) -> str:
        file_obj = self._client.files.create(
            file=(filename, content.encode("utf-8"), "text/markdown"),
            purpose=self._file_purpose,
        )
        file_id = file_obj.id
        try:
            vsf = self._client.vector_stores.files.create_and_poll(
                vector_store_id=self._vs_id,
                file_id=file_id,
                attributes=attributes,
                poll_interval_ms=self._poll_interval_ms,
            )
        except Exception:
            # Attach raised — the file may or may not be partially indexed.
            # Cleanup is best-effort; a cleanup failure is logged, the
            # ORIGINAL exception still propagates.
            # `except Exception` (not BaseException): on Ctrl-C, we prefer
            # to surface the interrupt immediately rather than issue another
            # API call. A leftover orphan is recoverable via reconcile.
            self._best_effort_files_delete(file_id)
            raise

        status = getattr(vsf, "status", None)
        if status == _TERMINAL_OK:
            return file_id

        # Surface the SDK's own error payload so operator triage doesn't
        # need a second roundtrip.
        last_error = getattr(vsf, "last_error", None)
        msg = (
            f"VS attach for file_id={file_id} ended with status={status!r}, "
            f"last_error={last_error!r}"
        )
        if status in _TERMINAL_FAILED:
            # Definite failure: delete the orphan and raise.
            self._best_effort_files_delete(file_id)
            raise RuntimeError(msg)
        # Unknown non-terminal (e.g. "in_progress" from a poll-budget timeout)
        # or an unfamiliar status: DON'T delete — the file might still be
        # indexing or transitioning. The operator inspects manually.
        raise RuntimeError(
            msg + " — orphan File preserved for operator inspection; "
            "use ragRebuildIndex --mode reconcile to absorb if it later completes."
        )

    def delete_section(self, file_id: str) -> None:
        # Step 1: detach from VS. NotFoundError → KeyError so callers can
        # treat it the same as `FakeVectorStoreClient.delete_section`.
        from openai import NotFoundError  # lazy import: see module docstring
        try:
            self._client.vector_stores.files.delete(
                vector_store_id=self._vs_id,
                file_id=file_id,
            )
        except NotFoundError as e:
            raise KeyError(f"unknown file_id {file_id!r} in vector store {self._vs_id!r}") from e

        # Step 2: delete the File object too. If it's already gone (e.g. a
        # prior detach succeeded but Files.delete failed midway) we accept
        # that as success — the end-state is the same as a clean two-step.
        try:
            self._client.files.delete(file_id)
        except NotFoundError:
            pass

    def list_sections(self) -> list[VectorStoreSection]:
        sections: list[VectorStoreSection] = []
        after: str | None = None
        while True:
            kwargs: dict[str, Any] = {"vector_store_id": self._vs_id, "limit": 100}
            if after is not None:
                kwargs["after"] = after
            page = self._client.vector_stores.files.list(**kwargs)
            page_data = list(page.data or [])
            for vsf in page_data:
                attrs = getattr(vsf, "attributes", None) or {}
                sections.append(VectorStoreSection(
                    file_id=vsf.id,
                    attributes=dict(attrs),
                ))
            if not getattr(page, "has_more", False):
                break
            if not page_data:
                # Defensive: has_more=True with empty data would loop forever.
                # Trust has_more=False as the only termination signal otherwise.
                break
            after = page_data[-1].id
        return sections

    # ─── helpers ──────────────────────────────────────────────────────────

    def _best_effort_files_delete(self, file_id: str) -> None:
        try:
            self._client.files.delete(file_id)
        except Exception as e:  # noqa: BLE001 — best-effort cleanup
            log.warning(
                "orphan File %s could not be cleaned up (manual delete needed): %s",
                file_id,
                e,
            )
