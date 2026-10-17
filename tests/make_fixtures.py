"""Genera GGUF sintetici per i test.

I file sono SPARSI: header e tabella tensori sono reali, l'area dati e' creata
con truncate() e quindi non occupa spazio su disco ma si legge come zeri. Serve
per validare parser, planner ed expert cache (che legge davvero da disco) senza
scaricare modelli da 18 GB.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argine.gguf import GGML_TYPES  # noqa: E402

T_UINT32, T_FLOAT32, T_STRING, T_ARRAY, T_UINT64 = 4, 6, 8, 9, 10

Q4_K = 12
Q6_K = 14
F32 = 0


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _str(key) + _u32(vtype) + payload


def _kv_u32(key: str, v: int) -> bytes:
    return _kv(key, T_UINT32, _u32(v))


def _kv_f32(key: str, v: float) -> bytes:
    return _kv(key, T_FLOAT32, struct.pack("<f", v))


def _kv_str(key: str, v: str) -> bytes:
    return _kv(key, T_STRING, _str(v))


def _kv_str_array(key: str, n: int) -> bytes:
    # array di stringhe corte (vocabolario finto)
    payload = _u32(T_STRING) + _u64(n)
    for i in range(n):
        payload += _str(f"t{i}")
    return _kv(key, T_ARRAY, payload)


def tensor_bytes(dims: List[int], ttype: int) -> int:
    ne = 1
    for d in dims:
        ne *= d
    _n, blk, tsize = GGML_TYPES[ttype]
    return ((ne + blk - 1) // blk) * tsize


def build_gguf(path: Path, metadata: bytes, kv_count: int,
               tensors: List[Tuple[str, List[int], int]], alignment: int = 32) -> None:
    header = b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(kv_count)
    table = b""
    offset = 0
    for name, dims, ttype in tensors:
        table += _str(name) + _u32(len(dims))
        for d in dims:
            table += _u64(d)
        table += _u32(ttype) + _u64(offset)
        nb = tensor_bytes(dims, ttype)
        pad = (alignment - (nb % alignment)) % alignment
        offset += nb + pad
    body = header + metadata + table
    pad = (alignment - (len(body) % alignment)) % alignment
    body += b"\x00" * pad
    data_size = offset

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(body)
        fh.truncate(len(body) + data_size)  # sparse


# ---------------------------------------------------------------- modelli ----

def make_dense_8b(path: Path) -> None:
    """Modello dense tipo Qwen3-8B Q4_K_M (~4.7 GB)."""
    arch = "qwen3"
    n_layers, n_embd, n_head, n_head_kv, head_dim, n_ff = 36, 4096, 32, 8, 128, 12288
    vocab = 151936

    kvs: List[bytes] = [
        _kv_str("general.architecture", arch),
        _kv_str("general.name", "Qwen3-8B-fixture"),
        _kv_u32("general.file_type", 15),
        _kv_u32(f"{arch}.block_count", n_layers),
        _kv_u32(f"{arch}.embedding_length", n_embd),
        _kv_u32(f"{arch}.feed_forward_length", n_ff),
        _kv_u32(f"{arch}.attention.head_count", n_head),
        _kv_u32(f"{arch}.attention.head_count_kv", n_head_kv),
        _kv_u32(f"{arch}.attention.key_length", head_dim),
        _kv_u32(f"{arch}.attention.value_length", head_dim),
        _kv_u32(f"{arch}.context_length", 32768),
        _kv_f32(f"{arch}.rope.freq_base", 1000000.0),
        _kv_u32(f"{arch}.vocab_size", vocab),
        _kv_str("tokenizer.ggml.model", "gpt2"),
    ]
    tensors: List[Tuple[str, List[int], int]] = [
        ("token_embd.weight", [n_embd, vocab], Q4_K),
        ("output_norm.weight", [n_embd], F32),
        ("output.weight", [n_embd, vocab], Q6_K),
    ]
    for i in range(n_layers):
        p = f"blk.{i}."
        tensors += [
            (p + "attn_norm.weight", [n_embd], F32),
            (p + "attn_q.weight", [n_embd, n_head * head_dim], Q4_K),
            (p + "attn_k.weight", [n_embd, n_head_kv * head_dim], Q4_K),
            (p + "attn_v.weight", [n_embd, n_head_kv * head_dim], Q6_K),
            (p + "attn_output.weight", [n_head * head_dim, n_embd], Q4_K),
            (p + "ffn_norm.weight", [n_embd], F32),
            (p + "ffn_gate.weight", [n_embd, n_ff], Q4_K),
            (p + "ffn_up.weight", [n_embd, n_ff], Q4_K),
            (p + "ffn_down.weight", [n_ff, n_embd], Q6_K),
        ]
    build_gguf(path, b"".join(kvs), len(kvs), tensors)


def make_moe_30b(path: Path, quant: int = Q4_K, n_ff_exp: int = 768) -> None:
    """Modello MoE tipo Qwen3-30B-A3B (128 expert, 8 attivi)."""
    arch = "qwen3moe"
    n_layers, n_embd, n_head, n_head_kv, head_dim = 48, 2048, 32, 4, 128
    n_expert, n_used = 128, 8
    vocab = 151936

    kvs: List[bytes] = [
        _kv_str("general.architecture", arch),
        _kv_str("general.name", "Qwen3-30B-A3B-fixture"),
        _kv_u32("general.file_type", 15),
        _kv_u32(f"{arch}.block_count", n_layers),
        _kv_u32(f"{arch}.embedding_length", n_embd),
        _kv_u32(f"{arch}.expert_count", n_expert),
        _kv_u32(f"{arch}.expert_used_count", n_used),
        _kv_u32(f"{arch}.expert_feed_forward_length", n_ff_exp),
        _kv_u32(f"{arch}.attention.head_count", n_head),
        _kv_u32(f"{arch}.attention.head_count_kv", n_head_kv),
        _kv_u32(f"{arch}.attention.key_length", head_dim),
        _kv_u32(f"{arch}.attention.value_length", head_dim),
        _kv_u32(f"{arch}.context_length", 32768),
        _kv_f32(f"{arch}.rope.freq_base", 1000000.0),
        _kv_u32(f"{arch}.vocab_size", vocab),
        _kv_str("tokenizer.ggml.model", "gpt2"),
    ]
    tensors: List[Tuple[str, List[int], int]] = [
        ("token_embd.weight", [n_embd, vocab], Q4_K),
        ("output_norm.weight", [n_embd], F32),
        ("output.weight", [n_embd, vocab], Q6_K),
    ]
    for i in range(n_layers):
        p = f"blk.{i}."
        tensors += [
            (p + "attn_norm.weight", [n_embd], F32),
            (p + "attn_q.weight", [n_embd, n_head * head_dim], Q4_K),
            (p + "attn_k.weight", [n_embd, n_head_kv * head_dim], Q4_K),
            (p + "attn_v.weight", [n_embd, n_head_kv * head_dim], Q6_K),
            (p + "attn_output.weight", [n_head * head_dim, n_embd], Q4_K),
            (p + "attn_q_norm.weight", [head_dim], F32),
            (p + "attn_k_norm.weight", [head_dim], F32),
            (p + "ffn_norm.weight", [n_embd], F32),
            (p + "ffn_gate_inp.weight", [n_embd, n_expert], F32),
            (p + "ffn_gate_exps.weight", [n_embd, n_ff_exp, n_expert], quant),
            (p + "ffn_up_exps.weight", [n_embd, n_ff_exp, n_expert], quant),
            (p + "ffn_down_exps.weight", [n_ff_exp, n_embd, n_expert], quant),
        ]
    build_gguf(path, b"".join(kvs), len(kvs), tensors)



# ------------------------------------------------- helper per i test split ---

def _moe_metadata(split_count: int = 0) -> List[bytes]:
    """Metadata minimi di un MoE tipo Qwen3-30B-A3B."""
    arch = "qwen3moe"
    kvs = [
        _kv_str("general.architecture", arch),
        _kv_str("general.name", "moe-split-fixture"),
        _kv_u32(f"{arch}.block_count", 48),
        _kv_u32(f"{arch}.embedding_length", 2048),
        _kv_u32(f"{arch}.expert_count", 128),
        _kv_u32(f"{arch}.expert_used_count", 8),
        _kv_u32(f"{arch}.attention.head_count", 32),
        _kv_u32(f"{arch}.attention.head_count_kv", 4),
        _kv_u32(f"{arch}.attention.key_length", 128),
        _kv_u32(f"{arch}.attention.value_length", 128),
        _kv_u32(f"{arch}.context_length", 32768),
        _kv_u32(f"{arch}.vocab_size", 151936),
    ]
    if split_count:
        kvs.append(_kv_u32("split.count", split_count))
    return kvs


def _moe_layer_tensors(layers) -> List[Tuple[str, List[int], int]]:
    """Solo i tensori expert dei layer indicati: bastano per i test sugli shard."""
    n_embd, n_ff_exp, n_expert = 2048, 768, 128
    out: List[Tuple[str, List[int], int]] = []
    for i in layers:
        p = f"blk.{i}."
        out += [
            (p + "ffn_gate_exps.weight", [n_embd, n_ff_exp, n_expert], Q4_K),
            (p + "ffn_up_exps.weight", [n_embd, n_ff_exp, n_expert], Q4_K),
            (p + "ffn_down_exps.weight", [n_ff_exp, n_embd, n_expert], Q4_K),
        ]
    return out


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
    out.mkdir(parents=True, exist_ok=True)
    dense = out / "fixture-dense-8B-Q4_K_M.gguf"
    moe = out / "fixture-moe-30B-A3B-Q4_K_M.gguf"
    moe_big = out / "fixture-moe-30B-A3B-Q6_K.gguf"
    make_dense_8b(dense)
    make_moe_30b(moe)
    make_moe_30b(moe_big, quant=Q6_K)
    for p in (dense, moe, moe_big):
        st = p.stat()
        du = st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
        print(f"{p.name}: apparente {st.st_size/2**30:.2f} GiB, su disco {du/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
