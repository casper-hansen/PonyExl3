#!/usr/bin/env python3
"""Tokenize a text file with the checkpoint's HF tokenizer into an ``input_ids`` .npz.

Both the CUDA export (``export_logprobs.py --from-npz``) and the MLX compare
replay these exact ids, so tokenizer differences can never leak into the
parity numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model-dir", required=True)
    ap.add_argument("-t", "--text", type=Path, required=True, help="UTF-8 text file")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-s", "--seq-len", type=int, default=512)
    ap.add_argument("--chat", action="store_true", help="wrap as a single user turn via the chat template")
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(args.model_dir)
    text = args.text.read_text(encoding="utf-8")
    if args.chat:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True
        )
    else:
        ids = tok.encode(text)
    ids = list(ids)[: args.seq_len]
    if len(ids) < args.seq_len:
        raise SystemExit(f"text tokenizes to {len(ids)} < --seq-len {args.seq_len}; use a longer text")
    arr = np.asarray(ids, dtype=np.int64)[None, :]
    np.savez(args.output, input_ids=arr, source=np.array(str(args.text)), model_dir=np.array(args.model_dir))
    print(f"wrote {args.output}: input_ids {arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
