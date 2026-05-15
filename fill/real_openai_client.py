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

# OpenAI `last_error.code` values that indicate a transient server-side
# fault — not content rejection. In production we've seen `server_error`
# arrive as `status='failed'`, only for OpenAI to silently mark the same
# file_id `completed` minutes later in the VS dashboard. Retry-via-probe
# is the right response. Content-related codes (e.g. file_too_large,
# invalid_content_type) must NOT retry.
_TRANSIENT_FAILURE_CODES = frozenset({"server_error", "internal_error"})

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

        # Transient `failed` with `server_error` — observed in production
        # cascades (build #8/#9). The file is already attached; only a
        # retrieve-only probe can recover. `cancelled` is intentional and
        # out-of-band — never retry.
        if (
            getattr(vsf, "status", None) == "failed"
            and self._is_retryable_status_failed(vsf)
        ):
            vsf = self._wait_for_completed_after_transient_failure(file_id, vsf)

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
        """Call `vector_stores.files.create_and_poll`, retrying ONLY the
        transient 404 "No file found with id X in vector store Y" error
        — the read-after-write inconsistency between Files.create and
        the VS attach endpoint.

        Other exceptions propagate. Returns whatever vsf the call
        ultimately produced (caller inspects status).

        `status='failed' server_error` (the second cascade mode) is
        handled by a DIFFERENT loop, `_wait_for_completed_after_transient_failure`,
        because re-issuing `create_and_poll` on a file_id that's already
        attached to the VS would either be a no-op or fail with a
        duplicate-attach error. A probe via `retrieve` is the only safe
        recovery mechanism once the file is in the VS.
        """
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
                    "VS attach for file_id=%s saw transient 404 (attempt %d/%d); "
                    "sleeping %.1fs",
                    file_id, attempt + 1, attempts, delay,
                )
                self._sleep(delay)
                # If the file already made it in despite the 404, surface it
                # immediately and skip the next attach.
                recovered = self._try_retrieve_vs_file(file_id)
                if recovered is not None:
                    return recovered

        raise AssertionError("unreachable: retry loop exited without resolution")

    def _wait_for_completed_after_transient_failure(
        self, file_id: str, initial_vsf: Any,
    ) -> Any:
        """Poll `vector_stores.files.retrieve` until either the file
        reaches `status='completed'` (recovery) or the budget is
        exhausted (return the last observed vsf, caller raises).

        Does NOT re-issue `create_and_poll`: by the time we get here, the
        file is already attached to the VS (even though the first poll
        reported failed). Issuing attach again would either be a no-op
        or fail as a duplicate attach.

        Production observation (build #9): 30 sections reported
        `status='failed' server_error` from create_and_poll; the VS
        dashboard showed them all `completed` minutes later. The
        retrieve-only loop is the right shape for that pattern.
        """
        delays = self._attach_retry_delays_seconds
        last_vsf = initial_vsf
        for i, delay in enumerate(delays):
            log.warning(
                "VS attach for file_id=%s saw transient status='failed'/server_error "
                "(probe %d/%d); sleeping %.1fs before retrieve",
                file_id, i + 1, len(delays), delay,
            )
            self._sleep(delay)
            recovered = self._try_retrieve_vs_file(file_id)
            if recovered is not None:
                return recovered
        return last_vsf

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
        """Soft probe of the VS for `file_id`. Returns the vsf only when
        `status == 'completed'` — the only signal that meaningfully
        short-circuits a retry loop. All other statuses (in_progress,
        failed, cancelled, missing) and errors return None so the caller
        keeps its retry budget intact.

        Probe errors are logged at DEBUG so a repeating auth/permission/
        rate-limit failure leaves a trail without contaminating WARN
        output of the surrounding retry logic.
        """
        try:
            vsf = self._client.vector_stores.files.retrieve(
                vector_store_id=self._vs_id,
                file_id=file_id,
            )
        except Exception as e:  # noqa: BLE001 — best-effort probe
            log.debug("retrieve probe for file_id=%s failed (swallowed): %s", file_id, e)
            return None
        if getattr(vsf, "status", None) == _TERMINAL_OK:
            return vsf
        return None

    def _is_retryable_status_failed(self, vsf: Any) -> bool:
        """Match a transient `status='failed'` (e.g. OpenAI internal
        indexing error) vs a permanent content-rejection failure.

        We've seen `last_error.code='server_error'` revert to `completed`
        on a later `retrieve`. Content-related codes never recover and
        must NOT retry."""
        last_error = getattr(vsf, "last_error", None)
        if last_error is None:
            return False
        # `last_error` may be a SDK pydantic model OR a plain dict
        # depending on the SDK version. Probe both shapes.
        code = (
            getattr(last_error, "code", None)
            if not isinstance(last_error, dict)
            else last_error.get("code")
        )
        return code in _TRANSIENT_FAILURE_CODES

    def _best_effort_files_delete(self, file_id: str) -> None:
        try:
            self._client.files.delete(file_id)
        except Exception as e:  # noqa: BLE001 — best-effort cleanup
            log.warning(
                "orphan File %s could not be cleaned up (manual delete needed): %s",
                file_id,
                e,
            )
