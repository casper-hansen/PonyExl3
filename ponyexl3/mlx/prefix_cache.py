"""Prefix (prompt) cache: reuse target state across requests that share a prefix.

Long-horizon agent loops re-send the whole conversation every turn. At the
~70 tok/s prefill a 27B reaches on an M2 Pro, a 4k-token context costs ~1 min
per turn just to re-read what the model already saw. This module snapshots
the target's cache (KV for full-attention layers, conv + recurrent state for
DeltaNet layers) at token boundaries and restores the longest snapshot that
is an exact prefix of the next prompt, so only the suffix is prefilled.

DeltaNet state cannot be trimmed, so a snapshot is only reusable at exactly
its own token position — hence snapshots at regular boundaries (``every``)
plus one at the end of each prefill and each generation. Chat templates
re-render history (Qwen3 drops earlier <think> blocks, the empty no-think
block after ``<|im_start|>assistant`` is not in history), so the common
prefix usually ends a few tokens *before* the previous prompt's end; the
periodic snapshots are what make that case still hit.

Cost: a Qwen3.8-27B snapshot is ~150 MB (dominated by 48 fp32 DeltaNet states)
+ 64 KB/token of KV — ``max_entries`` bounds it (LRU).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ponyexl3.types import MlxLmModel


@dataclass
class _Entry:
    tokens: tuple[int, ...]
    states: list[Any]  # per-layer cache.state
    logits: mx.array | None  # next-token logits at this position (if known)
    bytes: int
    aux: mx.array | None = None  # drafter aux features window (DFlash2), (1, W, 5*H)


def _state_bytes(states: list[Any]) -> int:
    n = 0
    for st in states:
        if isinstance(st, (list, tuple)):
            n += sum(a.nbytes for a in st if isinstance(a, mx.array))
        elif isinstance(st, mx.array):
            n += st.nbytes
    return n


def _common_prefix(a: tuple[int, ...], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCache:
    """LRU of cache snapshots keyed by their exact token prefix."""

    def __init__(self, model: MlxLmModel, *, every: int = 2048, max_entries: int = 3):
        self.lm = getattr(model, "language_model", model)
        self.every = every
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], _Entry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.tokens_reused = 0

    # -- storage ----------------------------------------------------------------------------
    @property
    def bytes(self) -> int:
        return sum(e.bytes for e in self._entries.values())

    def snapshot(
        self,
        tokens: list[int],
        cache: list[Any],
        logits: mx.array | None = None,
        *,
        aux: mx.array | None = None,
    ) -> None:
        """Record ``cache``'s current state as the state after ``tokens``.

        ``aux`` (optional) is the DFlash2 drafter's context-feature window at
        this position, so the speculative path can resume without recomputing
        target hidden states for the reused prefix."""
        key = tuple(tokens)
        states = [c.state for c in cache]
        if logits is not None:
            logits = mx.array(logits)
        mx.eval(*[a for st in states for a in (st if isinstance(st, (list, tuple)) else [st]) if isinstance(a, mx.array)])
        if logits is not None:
            mx.eval(logits)
        if aux is not None:
            mx.eval(aux)
        if key in self._entries:
            self._entries.move_to_end(key)
        nbytes = _state_bytes(states) + (aux.nbytes if aux is not None else 0)
        self._entries[key] = _Entry(key, states, logits, nbytes, aux)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def lookup_entry(self, tokens: list[int], *, max_len: int | None = None) -> _Entry | None:
        """Longest snapshot that is an exact prefix of ``tokens`` (len <= max_len)."""
        best: _Entry | None = None
        for key, e in self._entries.items():
            if len(key) <= len(tokens) and (max_len is None or len(key) <= max_len):
                if _common_prefix(key, tokens) == len(key) and (best is None or len(key) > len(best.tokens)):
                    best = e
        if best is not None:
            self._entries.move_to_end(best.tokens)
        return best

    def restore_into(self, cache: list[Any], entry: _Entry) -> None:
        for c, st in zip(cache, entry.states):
            c.state = st

    def lookup(self, tokens: list[int]) -> tuple[list[Any], int, mx.array | None] | None:
        """Longest snapshot that is an exact prefix of ``tokens`` (may equal it).

        Returns ``(fresh caches restored to that state, n_prefix, logits)``.
        """
        best: _Entry | None = None
        for key, e in self._entries.items():
            if len(key) <= len(tokens) and _common_prefix(key, tokens) == len(key):
                if best is None or len(key) > len(best.tokens):
                    best = e
        if best is None:
            self.misses += 1
            return None
        self._entries.move_to_end(best.tokens)
        self.hits += 1
        self.tokens_reused += len(best.tokens)
        cache = self.lm.make_cache()
        for c, st in zip(cache, best.states):
            c.state = st
        return cache, len(best.tokens), best.logits

    # -- prefill ----------------------------------------------------------------------------
    def prefill(
        self,
        prompt_ids: list[int],
        *,
        chunk: int = 512,
        snapshot_end: bool = True,
    ) -> tuple[list[Any], mx.array, int]:
        """Prefill ``prompt_ids`` reusing the best snapshot.

        Returns ``(cache, logits_last, n_reused)`` where ``logits_last`` are the
        next-token logits after the full prompt (shape (1, 1, V)).
        """
        lm = self.lm
        hit = self.lookup(prompt_ids)
        if hit is not None:
            cache, n0, logits = hit
            if n0 == len(prompt_ids) and logits is not None:
                return cache, logits, n0
            if n0 == len(prompt_ids):
                # exact-length hit without logits: recompute from one token back
                # is impossible (state already includes it) -> treat as miss on
                # the last token by re-running from the previous boundary
                prev = self._nearest_boundary_below(prompt_ids, n0)
                if prev is None:
                    cache, n0 = lm.make_cache(), 0
                else:
                    cache, n0, _ = prev
        else:
            cache, n0 = lm.make_cache(), 0

        toks = mx.array([prompt_ids])
        S = len(prompt_ids)
        h = None
        pos = n0
        while pos < S:
            end = min(pos + chunk, S)
            # stop at a snapshot boundary so we can record it
            nb = ((pos // self.every) + 1) * self.every
            if nb < end and nb > pos:
                end = nb
            h = lm.model(toks[:, pos:end], cache=cache)
            mx.eval(h)
            pos = end
            if pos % self.every == 0 and pos < S:
                self.snapshot(prompt_ids[:pos], cache)
        assert h is not None
        logits = lm.lm_head(h[:, -1:, :])
        mx.eval(logits)
        if snapshot_end:
            self.snapshot(prompt_ids, cache, logits)
        return cache, logits, n0

    def _nearest_boundary_below(self, tokens: list[int], n: int):
        best = None
        for key, e in self._entries.items():
            if len(key) < n and _common_prefix(key, tokens) == len(key):
                if best is None or len(key) > len(best.tokens):
                    best = e
        if best is None:
            return None
        cache = self.lm.make_cache()
        for c, st in zip(cache, best.states):
            c.state = st
        return cache, len(best.tokens), best.logits

    def stats(self) -> str:
        return (
            f"prefix-cache: {len(self._entries)} entries, {self.bytes / 1e9:.2f} GB, "
            f"hits={self.hits} misses={self.misses} tokens_reused={self.tokens_reused}"
        )
