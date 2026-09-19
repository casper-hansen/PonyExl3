"""DFlash2 drafter glue for PonyExl3 targets.

Wraps the vendored z-lab reference (``dflash2_ref``) so it drives an EXL3
target from :func:`ponyexl3.mlx.model.load_model`:

- loads the drafter from a local directory (no hub round-trip),
- optionally requantizes the drafter body (``nn.Linear`` >= 1M params) to
  4/8-bit affine with a safetensors sidecar cache, so the 3.85 GB bf16
  checkpoint costs ~1 GB resident on 16 GB Macs and never has to be fully
  materialized in bf16 after the first run,
- exposes a ``GenStats``-style generator over the reference's verify-gated
  loop (greedy output is token-identical to plain decoding).

The reference binds to the target's ``embed_tokens`` / ``lm_head`` and hooks
the target layers at ``target_layer_ids`` for aux features, which all works on
the mlx_lm skeleton PonyExl3 builds (EXL3Linear / MmapEmbedding are drop-ins).
It also swaps ``GatedDeltaNet.__call__`` for a capturing version during
generation (state rollback for rejected drafts); PonyExl3's compiled-glue
patch is restored afterwards by the reference's own ``close()``.
"""

from __future__ import annotations

import json
import os
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ponyexl3.mlx import dflash2_ref as ref
from ponyexl3.mlx.generate import GenStats
from ponyexl3.types import MlxLmModel, Tokenizer

QUANT_MIN_PARAMS = 1_000_000


def _config_from_dir(path: Path) -> ref.DFlashConfig:
    cfg = json.loads((path / "config.json").read_text())
    dflash = cfg.get("dflash_config", {})
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling")
    layer_types = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])
    return ref.DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg.get("rope_theta", (rope or {}).get("rope_theta", 10000.0)),
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=int(dflash.get("block_size", cfg.get("block_size", 16))),
        target_layer_ids=tuple(dflash["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=dflash["mask_token_id"],
        rope_scaling=rope,
        layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=dflash.get("final_logit_softcapping", cfg.get("final_logit_softcapping")),
        input_embedding_scale=float(dflash.get("input_embedding_scale", 1.0)),
        output_multiplier=float(dflash.get("output_multiplier", 1.0)),
        conv_kernel_size=int(dflash.get("conv_kernel_size", 0)),
        conv_group_size=int(dflash.get("conv_group_size", 0)),
        selector_rank=int(dflash.get("selector_rank", 0)),
        selector_top_k=int(dflash.get("selector_top_k", 0)),
        is_causal=cfg.get("is_causal"),
    )


def _is_dflash2(path: Path) -> bool:
    cfg = json.loads((path / "config.json").read_text())
    return "DFlash2DraftModel" in (cfg.get("architectures") or [])


def _quant_predicate(min_params: int):
    def pred(_path: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) and module.weight.size >= min_params

    return pred


def load_dflash2(
    draft_dir: str | os.PathLike[str],
    *,
    bits: int | None = 4,
    group_size: int = 64,
    verbose: bool = False,
) -> ref.DFlashDraftModel:
    """Load a DFlash / DFlash2 drafter from ``draft_dir``.

    ``bits=None`` keeps the bf16 body. Otherwise the body linears are affine
    requantized and cached at ``<draft_dir>/.pony_cache/body_w{bits}g{gs}.safetensors``;
    on later loads only the quantized tensors are read (the bf16 shard is not
    touched), which is what keeps the load transient small.
    """
    path = Path(draft_dir)
    config = _config_from_dir(path)
    model_cls = ref.DFlash2DraftModel if _is_dflash2(path) else ref.DFlashDraftModel
    model = model_cls(config)
    model.eval()
    model.bind = types.MethodType(_bind_to_pony_target, model)  # type: ignore[method-assign]

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors under {path}")

    if bits is None:
        weights: dict[str, mx.array] = {}
        for f in shards:
            weights.update(mx.load(str(f)))
        _fix_selector_keys(weights, model_cls)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        return model

    cache_file = path / ".pony_cache" / f"body_w{bits}g{group_size}.safetensors"
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_quant_predicate(QUANT_MIN_PARAMS))
    if cache_file.is_file():
        model.load_weights(str(cache_file))
        mx.eval(model.parameters())
        if verbose:
            print(f"  dflash2: loaded w{bits} body from {cache_file.name}")
        return model

    # First run: build from bf16, quantizing linear-by-linear so the peak is
    # one bf16 tensor plus the quantized model rather than the whole shard.
    t0 = time.perf_counter()
    weights = {}
    for f in shards:
        weights.update(mx.load(str(f)))  # lazy: nothing is read until used
    _fix_selector_keys(weights, model_cls)
    quantized: dict[str, mx.array] = {}
    qmods = {p: m for p, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)}
    plain: list[tuple[str, mx.array]] = []
    for key, val in weights.items():
        owner = key[: -len(".weight")] if key.endswith(".weight") else None
        if owner in qmods:
            wq, sc, bi = mx.quantize(val, group_size=group_size, bits=bits)
            mx.eval(wq, sc, bi)
            quantized[f"{owner}.weight"] = wq
            quantized[f"{owner}.scales"] = sc
            quantized[f"{owner}.biases"] = bi
            del val
        else:
            plain.append((key, val))
    model.load_weights(list(quantized.items()) + plain)
    mx.eval(model.parameters())
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(cache_file), flat)
    if verbose:
        print(f"  dflash2: quantized body to w{bits} in {time.perf_counter() - t0:.1f}s -> {cache_file}")
    mx.clear_cache()
    return model


def _bind_to_pony_target(draft: ref.DFlashDraftModel, target_model: Any) -> ref.DFlashDraftModel:
    """``DFlashDraftModel.bind`` picks ``lm_head`` with ``getattr(...) or ...``;
    an ``nn.Module`` is a dict, and EXL3Linear keeps its state in private
    attributes, so it is an *empty* dict -> falsy -> the reference falls back
    to ``embed_tokens.as_linear``. Bind explicitly instead."""
    ref.DFlashDraftModel.bind(draft, target_model)
    lm = getattr(target_model, "language_model", target_model)
    head = getattr(target_model, "lm_head", None)
    if head is None:
        head = getattr(lm, "lm_head", None)
    if head is None:
        raise AttributeError("target has no lm_head")
    draft.lm_head = head
    return draft


def _fix_selector_keys(weights: dict[str, mx.array], model_cls: type) -> None:
    if model_cls is ref.DFlash2DraftModel:
        for name in ("predecessor_codebook", "successor_codebook"):
            key = f"candidate_selector.{name}"
            if key in weights:
                weights[f"{key}.weight"] = weights.pop(key)


# Prefix cache consulted by the prefill override below (set per call by
# dflash2_stream_generate; the reference loop's signature has no slot for it).
_ACTIVE_PREFIX_CACHE: Any | None = None
_LAST_PREFILL_REUSED = 0


def _pony_prefill_target(model: Any, prompt: mx.array, cache: Any, hidden_limit: int | None, step_size: int):
    """Same contract as the reference ``_prefill_target`` but:

    - runs the target's decoder only and applies ``lm_head`` to the final
      position (the reference calls the full model per chunk, i.e. lm_head over
      512 rows — on an EXL3 target that is the striped path with a 2.5 GB cache);
    - with an active :class:`PrefixCache`, restores the longest exact-prefix
      snapshot (target state + drafter aux-feature window) into ``cache`` and
      prefills only the suffix, snapshotting at chunk boundaries.
    """
    global _LAST_PREFILL_REUSED
    if step_size <= 0:
        raise ValueError("prefill_step_size must be positive.")
    inner = model.model
    pc = _ACTIVE_PREFIX_CACHE
    ids = [int(t) for t in prompt.tolist()]
    hidden_chunks: list[mx.array] = []
    start = 0
    _LAST_PREFILL_REUSED = 0
    if pc is not None:
        # keep at least the last token to prefill (we need fresh logits + h)
        e = pc.lookup_entry(ids, max_len=len(ids) - 1)
        if e is not None and e.aux is not None:
            pc.restore_into(cache, e)
            hidden_chunks = [e.aux]
            start = len(e.tokens)
            _LAST_PREFILL_REUSED = start
            pc.hits += 1
            pc.tokens_reused += start
        else:
            pc.misses += 1
    h = None
    while start < prompt.size:
        remaining = prompt.size - start
        end = start + (1 if remaining == 1 else min(step_size, remaining - 1))
        h = inner(prompt[None, start:end], cache)
        hidden = mx.concatenate(model._hidden_states, axis=-1)
        if hidden_limit is None:
            hidden_chunks.append(hidden)
        else:
            if hidden_chunks:
                hidden = mx.concatenate((hidden_chunks[0], hidden), axis=1)
            hidden_chunks = [hidden[:, -hidden_limit:]]
        if end < prompt.size:
            mx.eval([c.state for c in cache], hidden_chunks[-1], h)
            mx.clear_cache()
            if pc is not None and end % pc.every == 0:
                pc.snapshot(ids[:end], cache, aux=mx.concatenate(hidden_chunks, axis=1))
        start = end
    hidden = hidden_chunks[0] if len(hidden_chunks) == 1 else mx.concatenate(hidden_chunks, axis=1)
    logits = model.lm_head(h[:, -1:])  # type: ignore[index]
    if pc is not None:
        mx.eval(logits, hidden, [c.state for c in cache])
        pc.snapshot(ids, cache, logits, aux=hidden)
    return logits, hidden, prompt.size - hidden.shape[1]


ref._prefill_target = _pony_prefill_target  # pyright: ignore[reportPrivateUsage]


@dataclass
class DFlash2Stats(GenStats):
    accepted_hist: dict[int, int] | None = None


def dflash2_stream_generate(
    model: MlxLmModel,
    draft: ref.DFlashDraftModel,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    *,
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    prefill_chunk: int = 512,
    block_size: int | None = None,
    stats: GenStats | None = None,
    prefix_cache: Any | None = None,
) -> Iterator[int]:
    """Yield token ids from the reference DFlash2 loop, filling ``stats``.

    The target is the wrapped PonyExl3 model (``model.language_model`` is what
    the reference sees as an mlx_lm model with ``.model.layers`` + ``lm_head``).
    """
    global _ACTIVE_PREFIX_CACHE
    stats = stats if stats is not None else GenStats()
    stats.prompt_tokens = len(prompt_ids)
    lm = getattr(model, "language_model", model)
    hist: dict[int, int] = {}
    tic = time.perf_counter()
    first = True
    _ACTIVE_PREFIX_CACHE = prefix_cache
    for resp in ref._stream_generate(  # pyright: ignore[reportPrivateUsage]
        lm,
        draft,
        tokenizer,
        mx.array(prompt_ids),
        block_size=block_size,
        max_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        prefill_step_size=prefill_chunk,
    ):
        if first:
            stats.prefill_s = resp.prompt_tokens / resp.prompt_tps if resp.prompt_tps else 0.0
            stats.prompt_reused = _LAST_PREFILL_REUSED
            _ACTIVE_PREFIX_CACHE = None
            tic = time.perf_counter()
            first = False
        if resp.accepted is not None:
            stats.spec_cycles += 1
            stats.spec_accepted += max(resp.accepted - 1, 0)  # accepted drafts (excl. bonus)
            stats.spec_drafted += (block_size or int(draft.config.block_size)) - 1
            hist[resp.accepted] = hist.get(resp.accepted, 0) + 1
        eos_ids = getattr(tokenizer, "eos_token_ids", None) or set()
        for t in resp.tokens:
            if t in eos_ids:
                # the reference includes the terminal EOS in ``tokens``; keep
                # it out of the yielded text like the other generate paths
                stats.finish_reason = "stop"
                break
            stats.gen_tokens += 1
            yield t
        if resp.finish_reason:
            stats.finish_reason = resp.finish_reason
    stats.decode_s = time.perf_counter() - tic
    if isinstance(stats, DFlash2Stats):
        stats.accepted_hist = hist
