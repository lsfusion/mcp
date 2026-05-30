"""Structured JSONL event logging for the MCP server (feedback loop, Phase A).

One JSON object per line. Common envelope merged with event-specific fields:
``{schema_version, event, ts, server_version, ok, ...}``.

Sinks:
* **stderr** — always (A1). Chosen over stdout so the STDIO transport stays
  intact (stdout is the MCP protocol channel in ``server.py stdio``; Docker's
  ``json-file`` driver captures stderr just the same).
* **dated file** — only when ``LOG_DIR`` is set (A2): appends to
  ``<LOG_DIR>/<stream>-YYYYMMDD.jsonl``.

Best-effort by contract: a logging failure NEVER propagates to the caller, so
it cannot break ``retrieve_docs``. See MCP-FEEDBACK-PLAN.md.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from settings import LOG_DIR, LOG_SCHEMA_VERSION, SERVER_VERSION


def _utc_now_iso() -> str:
    """ISO-8601 UTC with millisecond precision, e.g. ``2026-05-29T12:34:56.789Z``."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def emit(event: str, fields: dict, *, stream: str, ok: bool = True) -> None:
    """Emit one structured event line. Never raises.

    ``event`` is the logical event name (e.g. ``retrieve_docs``); ``stream`` is
    the dated-file prefix (e.g. ``retrieval``); ``fields`` are merged after the
    common envelope.
    """
    try:
        record = {
            "schema_version": LOG_SCHEMA_VERSION,
            "event": event,
            "ts": _utc_now_iso(),
            "server_version": SERVER_VERSION,
            "ok": ok,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    except Exception:  # noqa: BLE001 — never let logging break the caller
        return

    # A1: stderr, best-effort (ignore write failure).
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass

    # A2: optional dated file, append-mode; warn on failure but do not raise.
    if LOG_DIR:
        try:
            day = datetime.now(timezone.utc).strftime("%Y%m%d")
            with open(os.path.join(LOG_DIR, f"{stream}-{day}.jsonl"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:  # noqa: BLE001
            # Keep stderr JSONL-clean: emit the file-sink failure as a JSON line too.
            try:
                print(
                    json.dumps(
                        {
                            "schema_version": LOG_SCHEMA_VERSION,
                            "event": "event_log_error",
                            "ts": _utc_now_iso(),
                            "server_version": SERVER_VERSION,
                            "ok": False,
                            "error_class": type(e).__name__,
                        },
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:  # noqa: BLE001
                pass
