#!/usr/bin/env python3
"""Export teacher-forced logprobs (every position) + full logits (last rows) from exllamav3.

Complements ``export_reference.py`` (last-position logits only) with the
quantities a logprob-parity audit needs on every position of a sequence:

- ``nll``        (S-1,)      -log p(ids[t+1] | ids[:t+1])   float32
- ``top1``       (S-1,)      argmax token id per position   int64
- ``topk_ids``   (S-1, K)    top-K token ids                int64
- ``topk_logp``  (S-1, K)    their log-probs                float32
- ``logits``     (R, V)      full float32 logits of the last R positions
- ``input_ids``  (1, S)

``input_ids`` come from ``--from-npz`` (tokenized elsewhere, e.g. on the Mac
with the HF tokenizer so both sides see identical ids) or from the standard
``torch.manual_seed(seed)`` random contract.

Run on the CUDA host with exllamav3 importable; only ``_cuda_common`` from
this directory is needed alongside.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

if __package__ in (None, ""):
    from _cuda_common import (
        forward_params,
        input_ids_to_torch,
        load_exllama_model,
        load_input_ids,
        make_input_ids,
        mask_logits,
        save_npz,
        standard_metadata,
    )
else:
    from ponyexl3.reference._cuda_common import (
        forward_params,
        input_ids_to_torch,
        load_exllama_model,
        load_input_ids,
        make_input_ids,
        mask_logits,
        save_npz,
        standard_metadata,
    )


def main(args: argparse.Namespace) -> int:
    if args.from_npz:
        ids_np = load_input_ids(args.from_npz)
        input_ids_t = input_ids_to_torch(ids_np)
        seq_len = int(ids_np.shape[1])
    else:
        seq_len = args.seq_len
        input_ids_t = None

    model, config = load_exllama_model(args.model_dir, seq_len=seq_len)
    vocab_size = int(config.vocab_size)
    if input_ids_t is None:
        input_ids_t = make_input_ids(vocab_size, seq_len, args.seed)

    params = forward_params(attn_mode=args.attn_mode)
    with torch.inference_mode():
        output = model.forward(input_ids_t, params)
        mask_logits(output, vocab_size)
        logits = output[0, :, :vocab_size].float()  # (S, V) on device
        logp = torch.log_softmax(logits, dim=-1)
        ids_dev = input_ids_t.to(logits.device)[0]
        nll = -logp[:-1].gather(1, ids_dev[1:, None])[:, 0]
        top1 = logits[:-1].argmax(dim=-1)
        topk_logp, topk_ids = logp[:-1].topk(args.topk, dim=-1)
        logits_rows = logits[-args.logit_rows :].cpu().numpy()

    payload = standard_metadata(
        model_dir=args.model_dir,
        input_ids=input_ids_t.cpu().numpy(),
        seed=args.seed,
        seq_len=seq_len,
        attn_mode=args.attn_mode,
        vocab_size=np.int64(vocab_size),
        hidden_size=np.int64(config.hidden_size),
        logits=logits_rows,
        nll=nll.cpu().numpy().astype(np.float32),
        top1=top1.cpu().numpy().astype(np.int64),
        topk_ids=topk_ids.cpu().numpy().astype(np.int64),
        topk_logp=topk_logp.cpu().numpy().astype(np.float32),
        device=np.array(torch.cuda.get_device_name(0)),
        exllamav3_version=np.array(_exllamav3_version()),
    )
    save_npz(args.output, payload, compressed=False)
    print(f" -- input_ids shape: {tuple(input_ids_t.shape)}")
    print(f" -- logits shape: {tuple(logits_rows.shape)}  nll mean: {float(nll.mean()):.4f}")
    print(f" -- attn_mode: {args.attn_mode}  device: {torch.cuda.get_device_name(0)}")
    return 0


def _exllamav3_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("exllamav3")
    except Exception:  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--from-npz", type=str, default=None, help=".npz with input_ids (1, S)")
    parser.add_argument("-s", "--seq_len", type=int, default=512, help="random-ids length (no --from-npz)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-r", "--logit_rows", type=int, default=8, help="full-logit rows kept (last R)")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--attn-mode", default="flash_attn_nc")
    raise SystemExit(main(parser.parse_args()))
