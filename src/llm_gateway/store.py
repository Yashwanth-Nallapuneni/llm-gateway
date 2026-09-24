"""Durable, resumable storage for eval sweeps.

`RunStore` sits in front of the gateway and answers one question per
request: has this exact prompt already been paid for? If yes, it hands back
the stored answer and never touches the network. If no, it records that a
call is about to be made, then records what came back -- so killing the
process partway through a large run loses at most the one call that was in
flight, not everything already done.

Everything here is plain synchronous sqlite3 wrapped in `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .types import LLMRequest, LLMResponse

# --------------------------------------------------------------------------
# Idempotency key
# --------------------------------------------------------------------------
#
# The key is a hash over the fields that determine the answer a provider
# would give: model, prompt, max_tokens, and the two capability flags.
# `priority` and `metadata` are left out on purpose, since they never reach
# the provider and don't affect the answer.
#
# One consequence worth knowing: two requests identical except for priority
# or metadata share one cache entry. Submitting the same prompt twice under
# different metadata tags, expecting two separate calls, gets one call and
# one answer copied to both. Give the prompt itself a distinguishing detail
# if that sharing is unwanted.


def idempotency_key(request: LLMRequest) -> str:
    """Stable hash over the fields that determine the answer."""
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "needs_logprobs": request.needs_logprobs,
        "needs_strict_json": request.needs_strict_json,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'in_flight', 'done', 'failed')),
    response_json TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    provider TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests (status);
CREATE INDEX IF NOT EXISTS idx_requests_run_id ON requests (run_id);
"""


class RunStore:
    """One SQLite file backing one resumable sweep.

    All public methods are `async` and offload the actual sqlite call to a
    worker thread via `asyncio.to_thread`. A single `threading.Lock` guards
    every access to the connection, since sqlite3 connections aren't safe
    to share across threads and `to_thread` can run calls on different
    threads. This serializes every store operation, which is fine for the
    write rate a provider-bound eval sweep produces, but not meant for a
    workload writing thousands of rows per second.
    """

    def __init__(self, path: str | Path, *, run_id: str | None = None) -> None:
        self.path = str(path)
        self.run_id = run_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            # WAL mode lets a reader (e.g. a status query) proceed while a
            # writer is mid-commit, and survives a killed process cleanly:
            # a commit is one append to the log file, so a crash lands
            # before or after it with no half-written database to repair.
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL still fsyncs at checkpoints, enough to survive this
            # process dying (a kill -9, an unhandled exception), which is
            # the scenario this module cares about. FULL would also survive
            # an OS crash or power loss, at the cost of an fsync per commit.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            # Any row still `in_flight` belongs to a previous process that
            # died before finishing it. Reclaim those rows as `pending` so
            # the dispatch loop retries them instead of waiting forever.
            self._conn.execute(
                "UPDATE requests SET status = 'pending', updated_at = ? "
                "WHERE status = 'in_flight'",
                (time.time(),),
            )

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------

    async def get_response(self, key: str) -> LLMResponse | None:
        """Return the stored answer for `key` if it is already `done`.

        This is the resume fast path: a hit means no queueing, no batching,
        no provider call, just the answer already paid for.

        This always returns the first answer this store ever received for
        `key`, never a fresh sample, even though a provider running at
        temperature > 0 could return different text each time. That is
        correct for resuming a sweep, since resuming must not change graded
        outputs; a caller that wants a new sample each time should vary the
        key itself instead of using a store.
        """
        return await asyncio.to_thread(self._sync_get_response, key)

    def _sync_get_response(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM requests WHERE key = ? AND status = 'done'",
                (key,),
            ).fetchone()
        if row is None or row["response_json"] is None:
            return None
        return _response_from_json(row["response_json"])

    # ------------------------------------------------------------------
    # the crash-safe write path
    # ------------------------------------------------------------------
    #
    # A row moves to `done` in exactly one place (`complete`, below), and
    # only as part of the same statement that writes the response, so a
    # crash can never leave a row marked done with no response saved.
    #
    # A row can still be lost between the provider actually answering and
    # `complete()` persisting that answer: a crash there leaves the row
    # `in_flight`, it gets reclaimed as `pending` on the next open, and
    # resuming calls the provider again. That means one duplicate call, not
    # a lost result. This is a deliberate trade: an occasional duplicate
    # call is preferable to silently losing a result, and getting exactly
    # one call per result is not possible without the provider itself
    # participating in the same transaction.

    async def reserve(self, request: LLMRequest, key: str) -> LLMResponse | None:
        """Claim `key` for a call about to be made.

        Returns the stored response if another attempt already finished it
        since the caller's own `get_response` check -- a benign race, and
        the caller should just use that response and skip the call. Returns
        `None` otherwise: the row is now `in_flight`, and the caller should
        call the provider and report back via `complete()` or `fail()`.

        A row that is `pending`, `failed`, or new all reserve the same way;
        `failed` is included so a resumed run retries prompts that actually
        failed, not ones that already succeeded. A row already `in_flight`
        (a concurrent duplicate submission of the same key) is claimed
        again rather than made to wait, which can cause two concurrent
        calls for one key -- a duplicate like any other here, not
        corruption, since whichever call finishes last simply wins the row.
        """
        return await asyncio.to_thread(self._sync_reserve, request, key)

    def _sync_reserve(self, request: LLMRequest, key: str) -> LLMResponse | None:
        now = time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT status, response_json FROM requests WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO requests "
                    "(key, run_id, prompt, status, attempts, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'in_flight', 0, ?, ?)",
                    (key, self.run_id, request.prompt, now, now),
                )
                return None
            if row["status"] == "done":
                return _response_from_json(row["response_json"])
            self._conn.execute(
                "UPDATE requests SET status = 'in_flight', run_id = ?, updated_at = ? "
                "WHERE key = ?",
                (self.run_id, now, key),
            )
            return None

    async def complete(
        self,
        key: str,
        response: LLMResponse,
        *,
        cost: float = 0.0,
    ) -> None:
        """Durably record a successful answer. See the note above this
        section for why `status = 'done'` is set in the same statement
        that writes `response_json`, never after it."""
        await asyncio.to_thread(self._sync_complete, key, response, cost)

    def _sync_complete(self, key: str, response: LLMResponse, cost: float) -> None:
        payload = json.dumps(dataclasses.asdict(response))
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE requests SET "
                "status = 'done', response_json = ?, error = NULL, "
                "provider = ?, input_tokens = ?, output_tokens = ?, cost = ?, "
                "attempts = attempts + 1, updated_at = ? "
                "WHERE key = ?",
                (
                    payload,
                    response.provider,
                    response.input_tokens,
                    response.output_tokens,
                    cost,
                    now,
                    key,
                ),
            )

    async def fail(self, key: str, error: str) -> None:
        """Record a failed attempt. The row goes to `failed`, not `done` --
        a resumed run treats it the same as `pending` and retries it (see
        `reserve`)."""
        await asyncio.to_thread(self._sync_fail, key, error)

    def _sync_fail(self, key: str, error: str) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE requests SET "
                "status = 'failed', error = ?, attempts = attempts + 1, updated_at = ? "
                "WHERE key = ?",
                (error, now, key),
            )

    # ------------------------------------------------------------------
    # reporting / lifecycle
    # ------------------------------------------------------------------

    async def counts(self) -> dict[str, int]:
        """Row count per status, for a run summary."""
        return await asyncio.to_thread(self._sync_counts)

    def _sync_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM requests GROUP BY status"
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    async def aclose(self) -> None:
        await asyncio.to_thread(self._conn.close)

    def close(self) -> None:
        # Synchronous variant for callers that never entered an event loop,
        # e.g. cleanup in a `finally` around a CLI invocation.
        self._conn.close()


def _response_from_json(payload: str) -> LLMResponse:
    return LLMResponse(**json.loads(payload))
