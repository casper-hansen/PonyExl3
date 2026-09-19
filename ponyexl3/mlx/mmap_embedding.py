"""Host-resident, memory-mapped token embedding.

On a 27B-class checkpoint the fp16 ``embed_tokens`` table is ~2.5 GB
(248k x 5120) — a third of the whole 2-bpw model — yet a forward pass only
ever *gathers* a handful of rows from it. Keeping the table wired on the GPU
is pure waste on memory-constrained Macs (16 GB), so this module leaves the
table in the safetensors shard on disk, ``mmap``s it, and gathers the needed
rows on the host per call. The rows are converted to fp16 on device, so the
output is bit-identical to ``nn.Embedding`` over a fp16 cast of the table.

Cost: one host sync per call (the token ids must be concrete) plus a tiny
copy — negligible next to the decoder stack, and the OS page cache keeps the
hot rows resident.
"""

from __future__ import annotations

import json
import os
import struct
from glob import glob

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_ST_DTYPES: dict[str, tuple[np.dtype | None, mx.Dtype]] = {
    "BF16": (None, mx.bfloat16),  # no numpy bf16 — read as uint16, view on device
    "F16": (np.dtype(np.float16), mx.float16),
    "F32": (np.dtype(np.float32), mx.float32),
}


def find_safetensors_tensor(model_dir: str, key: str) -> tuple[str, str, list[int], int] | None:
    """Locate ``key`` across the shards: ``(path, dtype, shape, byte_offset)``."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    candidates: list[str]
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            wm = json.load(f).get("weight_map", {})
        candidates = [os.path.join(model_dir, wm[key])] if key in wm else []
    else:
        candidates = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    for path in candidates:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        if key in header:
            ent = header[key]
            start, _end = ent["data_offsets"]
            return path, ent["dtype"], list(ent["shape"]), 8 + n + start
    return None


class MmapEmbedding(nn.Module):
    """Drop-in for ``nn.Embedding`` whose table lives in the checkpoint on disk."""

    def __init__(self, path: str, dtype: str, shape: list[int], offset: int, *, out_dtype: mx.Dtype = mx.float16):
        super().__init__()
        if dtype not in _ST_DTYPES:
            raise ValueError(f"unsupported embedding dtype {dtype!r}")
        np_dtype, mx_dtype = _ST_DTYPES[dtype]
        self._path = path
        self._src_dtype = mx_dtype
        self._out_dtype = out_dtype
        self.num_embeddings, self.dims = int(shape[0]), int(shape[1])
        self._mm = np.memmap(
            path,
            dtype=np_dtype if np_dtype is not None else np.uint16,
            mode="r",
            offset=offset,
            shape=(self.num_embeddings, self.dims),
        )

    def _extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.dims}, mmap={os.path.basename(self._path)}"

    def __call__(self, x: mx.array) -> mx.array:
        ids = np.asarray(x, dtype=np.int64)
        flat = ids.reshape(-1)
        rows = np.ascontiguousarray(self._mm[flat])
        out = mx.array(rows)
        if self._src_dtype == mx.bfloat16:
            out = out.view(mx.bfloat16)
        out = out.astype(self._out_dtype)
        return out.reshape(*ids.shape, self.dims)

    def as_linear(self, x: mx.array) -> mx.array:  # pragma: no cover
        raise NotImplementedError("MmapEmbedding does not support tied lm_head (as_linear)")


def mmap_embedding_enabled() -> bool:
    """``PONYEXL3_EMBED_MMAP`` — default on; ``0``/``false`` restores the
    device-resident ``nn.Embedding``."""
    env = os.environ.get("PONYEXL3_EMBED_MMAP", "").strip().lower()
    return env not in ("0", "false", "no", "off")
