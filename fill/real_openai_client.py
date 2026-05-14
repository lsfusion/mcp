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
import time
from collections.abc import Callable
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

# Backoff schedule for retrying transient `create_and_poll` 404s caused by
# Files-API / Vector-Store read-after-write inconsistency on OpenAI's side.
# Observed in production (build #3, 2026-05-14): 478 consecutive attach
# requests returned 404 "No file found with id" over ~2 seconds, while the
# Files API had succeeded — most of those files later appeared in the VS
# as `completed` according to the direct VS query. Total worst-case wait
# per section: 1+2+4+8+16 = 31s.
_DEFAULT_ATTACH_RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)


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
        attach_retry_delays_seconds: tuple[float, ...] = _DEFAULT_ATTACH_RETRY_DELAYS_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
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
        # Retry policy is per-instance, not module-global, so a Jenkins
        # operator can override via the driver if OpenAI tightens or relaxes
        # the consistency window. `sleep` is injected so tests don't wait.
        self._attach_retry_delays_seconds = attach_retry_delays_seconds
        self._sleep = sleep

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
            vsf = self._create_and_poll_with_retry(file_id=file_id, attributes=attributes)
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

    def _create_and_poll_with_retry(
        self,
        *,
        file_id: str,
        attributes: dict[str, AttrValue],
    ) -> Any:
        """Call `vector_stores.files.create_and_poll`, with retry only for
        the OpenAI-side eventual-consistency 404 pattern observed in
        production. All other exceptions propagate immediately to the
        upload_section orphan-cleanup path."""
        delays = self._attach_retry_delays_seconds
        attempts = len(delays) + 1

        for attempt in range(attempts):
            try:
                return self._client.vector_stores.files.create_and_poll(
                    vector_store_id=self._vs_id,
                    file_id=file_id,
                    attributes=attributes,
                    poll_interval_ms=self._poll_interval_ms,
                )
            except Exception as e:
                if not self._is_retryable_attach_not_found(e, file_id):
                    raise

                if attempt == attempts - 1:
                    raise

                delay = delays[attempt]
                log.warning(
                    "VS attach/poll saw transient 404 for file_id=%s "
                    "(attempt %d/%d); retrying in %.1fs",
                    file_id, attempt + 1, attempts, delay,
                )
                self._sleep(delay)

                # After the wait, try retrieve once before the next
                # create_and_poll — saves an attach call if VS already has
                # the file. If the file is terminally-failed in the VS,
                # this raises (no point retrying) and `upload_section`'s
                # except path will run orphan cleanup.
                recovered = self._try_retrieve_vs_file(file_id)
                if recovered is not None:
                    return recovered

        # Unreachable: the loop always either returns or raises on the
        # final attempt above.
        raise AssertionError("unreachable: retry loop exited without resolution")

    def _is_retryable_attach_not_found(self, e: Exception, file_id: str) -> bool:
        """Match OpenAI's "No file found with id X in vector store" 404 —
        the read-after-write inconsistency between Files.create and the VS
        attach endpoint. Non-matching 404s (e.g. wrong VS id, deleted file)
        propagate so we don't paper over real bugs.

        If OpenAI reworded the message and the substring check stops
        matching, we'd silently lose retry coverage. The `log.warning`
        below fires the first time an otherwise-valid NotFoundError fails
        the message-match — so an operator sees the drift before the next
        full cascade outage hits."""
        try:
            from openai import NotFoundError  # lazy: see module docstring
        except ImportError:
            # In test environments without the openai SDK installed (and
            # without the test-only shim), any exception here is, by
            # definition, NOT a real OpenAI NotFoundError → not retryable.
            return False
        if not isinstance(e, NotFoundError):
            return False
        message = str(e)
        matched = (
            file_id in message
            and "No file found with id" in message
            and "vector store" in message
        )
        if not matched:
            log.warning(
                "VS attach 404 did NOT match expected eventual-consistency pattern "
                "(file_id=%s, message=%r). If OpenAI reworded the 404 surface, "
                "update _is_retryable_attach_not_found.",
                file_id, message,
            )
        return matched

    def _try_retrieve_vs_file(self, file_id: str) -> Any | None:
        """Direct `retrieve` on the VS file. Returns the file object if it
        is already `completed` in the VS (attach DID succeed despite the
        create_and_poll exception). Returns None if the file is unknown
        or still in-progress — caller continues the retry loop.

        Raises `RuntimeError` if the file is terminally-failed (status in
        `{failed, cancelled}`) — retrying create_and_poll wouldn't help,
        and we want the orphan-cleanup path in `upload_section` to run
        instead of accumulating dead VS rows."""
        try:
            vsf = self._client.vector_stores.files.retrieve(
                vector_store_id=self._vs_id,
                file_id=file_id,
            )
        except Exception:  # noqa: BLE001 — best-effort probe
            return None
        status = getattr(vsf, "status", None)
        if status == _TERMINAL_OK:
            return vsf
        if status in _TERMINAL_FAILED:
            last_error = getattr(vsf, "last_error", None)
            # Surface via RuntimeError so the upload_section's except path
            # picks it up (orphan File cleanup + propagation). Not a
            # retryable-404, so the next iteration would not have helped.
            raise RuntimeError(
                f"VS attach for file_id={file_id} ended with status={status!r}, "
                f"last_error={last_error!r} (observed via retry-time retrieve)"
            )
        return None

    def _best_effort_files_delete(self, file_id: str) -> None:
        try:
            self._client.files.delete(file_id)
        except Exception as e:  # noqa: BLE001 — best-effort cleanup
            log.warning(
                "orphan File %s could not be cleaned up (manual delete needed): %s",
                file_id,
                e,
            )
