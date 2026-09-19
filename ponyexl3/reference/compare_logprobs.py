#!/usr/bin/env python3
"""Logprob parity: PonyExl3 (MLX/Metal) vs native exllamav3 (CUDA) on every position.

Replays the ``input_ids`` stored by ``export_logprobs.py`` through the MLX
loader and reports, over all S-1 predicted positions:

- |Δ nll|      per-position |log p_mlx(target) - log p_cuda(target)|
- KL           KL(p_cuda || p_mlx) per position (full vocab)
- top-1        argmax agreement rate
- top-K        overlap of the CUDA top-K set with the MLX top-K set
- logits       max|Δ| / rms on the last R rows whose full logits were exported

``--noise-floor`` additionally replays MLX against itself with a different
prefill chunk (the schedule changes fp16 accumulation order), so the CUDA↔MLX
numbers can be read against the implementation's own nondeterminism.

lm_head is evaluated in ≤64-row blocks (fused GEMM path) so the whole-vocab
logits for a 512-token sequence never need the striped 2.5 GB fp16 lm_head
cache — this runs on a 16 GB Mac.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ponyexl3.types import MlxLmModel

LM_HEAD_ROWS = 64


def mlx_hidden(model: MlxLmModel, input_ids: np.ndarray, *, chunk: int) -> Any:
    """Final hidden states (1, S, H) fp16 for the whole sequence, chunked prefill."""
    import mlx.core as mx

    lm = model.language_model
    cache = lm.make_cache()
    ids = mx.array(input_ids.astype(np.int64))
    S = ids.shape[1]
    parts = []
    for s0 in range(0, S, chunk):
        h = lm.model(ids[:, s0 : s0 + chunk], cache=cache)
        mx.eval(h)
        parts.append(h)
    return mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def mlx_position_stats(
    model: MlxLmModel,
    h: Any,
    input_ids: np.ndarray,
    *,
    topk: int,
    logit_rows: int,
) -> dict[str, np.ndarray]:
    """Per-position nll / top1 / topk / full logp for the last rows, from hidden states."""
    import mlx.core as mx

    lm = model.language_model
    S = int(h.shape[1])
    V = None
    ids = input_ids[0]
    nll = np.empty(S - 1, dtype=np.float32)
    top1 = np.empty(S - 1, dtype=np.int64)
    topk_ids = np.empty((S - 1, topk), dtype=np.int64)
    topk_logp = np.empty((S - 1, topk), dtype=np.float32)
    logp_tail: list[np.ndarray] = []
    logits_tail: list[np.ndarray] = []
    tail_start = S - logit_rows
    for r0 in range(0, S, LM_HEAD_ROWS):
        r1 = min(S, r0 + LM_HEAD_ROWS)
        logits = lm.lm_head(h[:, r0:r1, :])[0].astype(mx.float32)  # (rows, V)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logits, logp)
        V = int(logp.shape[-1])
        lp = np.array(logp)
        lg = np.array(logits) if r1 > tail_start else None
        for i in range(r1 - r0):
            t = r0 + i
            if t >= tail_start:
                logp_tail.append(lp[i])
                logits_tail.append(lg[i])  # type: ignore[index]
            if t >= S - 1:
                continue
            row = lp[i]
            nll[t] = -row[ids[t + 1]]
            top1[t] = int(row.argmax())
            part = np.argpartition(-row, topk)[:topk]
            order = part[np.argsort(-row[part])]
            topk_ids[t] = order
            topk_logp[t] = row[order]
        del logits, logp
        mx.clear_cache()
    assert V is not None
    return {
        "nll": nll,
        "top1": top1,
        "topk_ids": topk_ids,
        "topk_logp": topk_logp,
        "logp_tail": np.stack(logp_tail),
        "logits_tail": np.stack(logits_tail),
    }


def summarize(
    ref: dict[str, Any],
    cand: dict[str, np.ndarray],
    *,
    label: str,
) -> dict[str, Any]:
    ref_nll = np.asarray(ref["nll"], dtype=np.float32)
    ref_top1 = np.asarray(ref["top1"], dtype=np.int64)
    ref_topk = np.asarray(ref["topk_ids"], dtype=np.int64)
    ref_logits = np.asarray(ref["logits"], dtype=np.float32)  # (R, V)
    n = ref_nll.shape[0]
    d_nll = np.abs(cand["nll"] - ref_nll)
    top1_match = cand["top1"] == ref_top1
    k = ref_topk.shape[1]
    overlap = np.array(
        [len(set(ref_topk[t].tolist()) & set(cand["topk_ids"][t].tolist())) / k for t in range(n)]
    )
    # tail rows: full-distribution comparison
    R = ref_logits.shape[0]
    ref_logp = ref_logits - np.logaddexp.reduce(ref_logits, axis=-1, keepdims=True)
    cand_logp = cand["logp_tail"][-R:]
    cand_logits = cand["logits_tail"][-R:]
    # KL(ref || cand) = sum p_ref (logp_ref - logp_cand)
    p_ref = np.exp(ref_logp)
    kl = (p_ref * (ref_logp - cand_logp)).sum(axis=-1)
    d_logp = cand_logp - ref_logp
    d_raw = cand_logits - ref_logits
    out = {
        "label": label,
        "positions": int(n),
        "nll_ref_mean": float(ref_nll.mean()),
        "nll_cand_mean": float(cand["nll"].mean()),
        "d_nll_mean": float(d_nll.mean()),
        "d_nll_median": float(np.median(d_nll)),
        "d_nll_p95": float(np.percentile(d_nll, 95)),
        "d_nll_max": float(d_nll.max()),
        "top1_agreement": float(top1_match.mean()),
        "top1_mismatches": int((~top1_match).sum()),
        f"top{k}_overlap_mean": float(overlap.mean()),
        "tail_rows": int(R),
        "tail_kl_mean": float(kl.mean()),
        "tail_kl_max": float(kl.max()),
        "tail_dlogp_max_abs": float(np.abs(d_logp).max()),
        "tail_dlogp_rms": float(np.sqrt((d_logp**2).mean())),
        "tail_dlogit_max_abs": float(np.abs(d_raw).max()),
        "tail_dlogit_rms": float(np.sqrt((d_raw**2).mean())),
        "tail_logit_rms": float(np.sqrt((ref_logits**2).mean())),
    }
    return out


def print_summary(s: dict[str, Any]) -> None:
    print(f"\n== {s['label']} ({s['positions']} positions)")
    print(f"  nll mean         ref={s['nll_ref_mean']:.4f}  cand={s['nll_cand_mean']:.4f}")
    print(
        f"  |Δ logp(target)|  mean={s['d_nll_mean']:.4g}  median={s['d_nll_median']:.4g}  "
        f"p95={s['d_nll_p95']:.4g}  max={s['d_nll_max']:.4g}"
    )
    print(f"  top-1 agreement  {100 * s['top1_agreement']:.2f}%  ({s['top1_mismatches']} mismatches)")
    kkey = next(k for k in s if k.startswith("top") and k.endswith("_overlap_mean"))
    print(f"  {kkey[:-13]} overlap    {100 * s[kkey]:.2f}%")
    print(
        f"  tail ({s['tail_rows']} rows, full vocab): KL(ref||cand) mean={s['tail_kl_mean']:.3g} "
        f"max={s['tail_kl_max']:.3g}"
    )
    print(
        f"    Δlogp  max|Δ|={s['tail_dlogp_max_abs']:.4g} rms={s['tail_dlogp_rms']:.4g} | "
        f"Δlogit max|Δ|={s['tail_dlogit_max_abs']:.4g} rms={s['tail_dlogit_rms']:.4g} "
        f"(rel {100 * s['tail_dlogit_rms'] / max(s['tail_logit_rms'], 1e-9):.2f}% of logit rms)"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path, help=".npz from export_logprobs.py (CUDA)")
    ap.add_argument("-m", "--model-dir", type=str, required=True)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--noise-floor", type=int, default=0, metavar="CHUNK",
                    help="also replay MLX vs MLX with this prefill chunk (0=skip)")
    ap.add_argument("--json", type=Path, default=None, help="write summaries as JSON")
    ap.add_argument("--save", type=Path, default=None, help="write the MLX per-position stats .npz")
    args = ap.parse_args()

    ref = dict(np.load(args.reference, allow_pickle=True))
    input_ids = np.asarray(ref["input_ids"]).astype(np.int64)
    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    topk = int(np.asarray(ref["topk_ids"]).shape[1])
    logit_rows = int(np.asarray(ref["logits"]).shape[0])

    from ponyexl3.mlx.model import describe, load_model

    print(f"reference: {args.reference}")
    print(f"           device={ref.get('device', '?')} exllamav3={ref.get('exllamav3_version', '?')} attn={ref.get('attn_mode', '?')}")
    print(f"model:     {args.model_dir}")
    print(f"seq_len:   {input_ids.shape[1]}  topk={topk}  tail_rows={logit_rows}")
    model, _ = load_model(args.model_dir, engine="exl3", warm=False)
    print(f"loaded:    {describe(model)}")

    import mlx.core as mx

    h = mlx_hidden(model, input_ids, chunk=args.prefill_chunk)
    cand = mlx_position_stats(model, h, input_ids, topk=topk, logit_rows=logit_rows)
    del h
    mx.clear_cache()
    summaries = [summarize(ref, cand, label=f"CUDA exllamav3 vs MLX PonyExl3 (chunk {args.prefill_chunk})")]
    print_summary(summaries[0])

    if args.noise_floor:
        h2 = mlx_hidden(model, input_ids, chunk=args.noise_floor)
        cand2 = mlx_position_stats(model, h2, input_ids, topk=topk, logit_rows=logit_rows)
        del h2
        mx.clear_cache()
        # treat the first MLX run as the "reference" for the self-comparison
        pseudo_ref = {
            "nll": cand["nll"],
            "top1": cand["top1"],
            "topk_ids": cand["topk_ids"],
            "logits": cand["logits_tail"],
        }
        s2 = summarize(pseudo_ref, cand2, label=f"MLX noise floor: chunk {args.prefill_chunk} vs chunk {args.noise_floor}")
        summaries.append(s2)
        print_summary(s2)

    if args.save:
        np.savez(args.save, input_ids=input_ids, **cand)
        print(f"\nwrote MLX stats to {args.save}")
    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
