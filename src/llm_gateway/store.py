"""Durable, resumable storage for eval sweeps.

`RunStore` sits in front of the gateway and answers one question per
request: has this exact prompt already been paid for? If yes, hand back the
stored answer and never touch the network. If no, remember that a call is
about to be made, then remember what came back -- so that killing the
process at prompt 14,000 of 20,000 loses at most the one call that was in
flight at the moment of the kill, not the other 13,999.

Everything here is synchronous sqlite3 wrapped in `asyncio.to_thread`. See
the "Blocking I/O" section near the bottom of this module for why that is
the right shape rather than an async sqlite driver.
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
# The key is a hash over exactly the fields that determine the answer a
# provider would give: model, prompt, max_tokens, and the two capability
# flags (they change what the provider is asked to produce -- logprobs,
# strict JSON -- so they are part of "the question", not bookkeeping about
# it).
#
# `priority` and `metadata` are deliberately excluded. Priority only affects
# queue ordering inside this process; metadata is caller bookkeeping (a
# dataset row id, a tag for later filtering) that never reaches the
# provider. Neither changes what answer comes back.
#
# The direct consequence, worth stating plainly: two `LLMRequest`s that are
# identical in every scored field but differ in priority or metadata will
# share one cache entry. If a sweep submits the same prompt twice under two
# different metadata tags expecting two independent provider calls, it gets
# one call and one answer copied to both -- by design. Give the prompt (or
# the model/max_tokens/flags) a distinguishing detail if that sharing is not
# wanted.


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
    every access to the connection: sqlite3 connections are not safe to use
    concurrently from multiple threads, and `to_thread` can and does run
    overlapping calls on different threads from the default executor. The
    lock turns every store operation into a short, serialized transaction --
    correct, and fine for the hundreds-of-milliseconds-apart write rate a
    provider-bound eval sweep actually produces; it is not meant for a
    workload writing thousands of rows per second.
    """

    def __init__(self, path: str | Path, *, run_id: str | None = None) -> None:
        self.path = str(path)
        self.run_id = run_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            # WAL lets a reader (e.g. a status query) proceed while a writer
            # is mid-commit, instead of blocking behind it the way the
            # default rollback journal would -- useful once a CLI wants to
            # report progress while the sweep is still writing. It also
            # survives a killed process better than the rollback journal:
            # a WAL commit is a single append to the log file, so a crash
            # either lands before or after that append and there is no
            # half-written main database file to repair on next open.
            self._conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL still fsyncs at WAL checkpoints, which is what makes a
            # commit durable against *this process* dying (a `kill -9`, an
            # unhandled exception) -- the crash scenario this module exists
            # for. It does not guarantee durability against the OS itself
            # crashing or the machine losing power between the write and
            # the checkpoint; FULL would, at the cost of an fsync per
            # commit. An eval sweep resuming after a killed process is the
            # scenario in scope here, so NORMAL is the right trade.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            # Any row still `in_flight` belongs to a previous process. This
            # process just started, so nothing of its is actually in
            # flight -- those rows are not "being worked on", they are
            # abandoned, and must be reclaimed as `pending` so the dispatch
            # loop below picks them back up instead of waiting forever for
            # a call that already died with the last process.
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

        This is the resume fast path: callers check this before doing
        anything else, and a hit means no queueing, no batching, no
        provider call -- just the answer that was already paid for.

        Replay caveat: at temperature > 0 a provider can return different
        text for the same prompt on different calls. This method always
        returns the FIRST answer this store ever received for `key`, never
        a fresh sample. That is exactly what an eval sweep wants --
        resuming must not silently change graded outputs -- and exactly
        wrong for a caller that wants a new sample each time it asks; that
        caller should not be using a store, or should vary the key (e.g.
        via the prompt text) itself.
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
    # Reasoning for the ordering below, spelled out once here rather than
    # at each call site:
    #
    # A row is moved to `done` in exactly one place (`complete`, below),
    # and only after the response has already been handed to sqlite as
    # part of that same UPDATE statement's committed transaction. There is
    # no separate "write the response" step followed later by a "flip the
    # status" step -- if there were, a crash between them would leave a row
    # that looks unfinished (good) but has already silently thrown away the
    # answer it received (bad, and pointless: the whole point of writing it
    # down was to not have to ask again).
    #
    # The window that remains is upstream of this module entirely: between
    # the provider actually answering and this process calling `complete()`
    # to persist that answer, the row is still `in_flight`. A crash in that
    # window means the answer that was on its way here is lost, the row
    # gets reclaimed as `pending` on the next open, and resume calls the
    # provider again for it. That is one duplicate call, not a lost result.
    #
    # This is deliberate and is the core trade the whole module makes:
    # losing a result silently is worse than an eval sweep occasionally
    # paying for one prompt twice, so the ordering is chosen to make
    # duplication the failure mode instead of loss. This is at-least-once
    # delivery. Exactly-once is not achievable here (or anywhere a local
    # commit and a remote, non-transactional API call cannot be made part
    # of one atomic operation) -- doing so would require the provider
    # itself to participate in a two-phase commit, which no LLM API offers.

    async def reserve(self, request: LLMRequest, key: str) -> LLMResponse | None:
        """Claim `key` for a call about to be made.

        Returns the stored response if another attempt already finished
        it since the caller's own `get_response` check (a benign race, not
        an error -- the caller should use this response and skip the call).
        Returns `None` otherwise, meaning: no finished answer exists, the
        row is now `in_flight`, and the caller should proceed to call the
        provider and report back via `complete()` or `fail()`.

        A row that was `pending`, `failed`, or freshly created all reserve
        the same way: `failed` is included so a resumed run retries only
        the prompts that actually failed, not the ones that already
        succeeded. A row already `in_flight` (a same-process concurrent
        duplicate of this exact key, submitted before the first attempt
        finished) is also claimed again rather than made to wait -- this
        can cause two concurrent provider calls for one key in that narrow
        case, which is a duplicate call like any other in this at-least-once
        design, not corruption: whichever `complete()`/`fail()` call lands
        last simply wins the row.
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
        # Synchronous variant: closing a sqlite connection is cheap (no I/O
        # beyond releasing the file handle) and callers that never entered
        # an event loop -- e.g. cleanup in a `finally` around a whole CLI
        # invocation -- need a way to release the file without `await`.
        self._conn.close()


def _response_from_json(payload: str) -> LLMResponse:
    return LLMResponse(**json.loads(payload))
