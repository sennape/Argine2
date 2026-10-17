"""Genera un GGUF llama minuscolo ma REALE, caricabile da llama.cpp.

Serve per il collaudo end-to-end di Argine (run, API, benchmark) senza
scaricare un modello vero. Le risposte sono ovviamente senza senso: conta solo
che il runtime carichi, generi token e risponda sull'API.
"""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path
from typing import List, Tuple

T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY = 4, 5, 6, 7, 8, 9
F32 = 0

N_VOCAB_EXTRA = 32
N_EMBD = 64
N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 4
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 128
CTX = 4096
ALIGN = 32

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _u32(v): return struct.pack("<I", v)
def _i32(v): return struct.pack("<i", v)
def _u64(v): return struct.pack("<Q", v)
def _f32(v): return struct.pack("<f", v)


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _kv(key, vtype, payload): return _str(key) + _u32(vtype) + payload
def _kv_u32(k, v): return _kv(k, T_UINT32, _u32(v))
def _kv_f32(k, v): return _kv(k, T_FLOAT32, _f32(v))
def _kv_str(k, v): return _kv(k, T_STRING, _str(v))
def _kv_bool(k, v): return _kv(k, T_BOOL, struct.pack("<?", v))


def _kv_arr_str(k, items: List[str]) -> bytes:
    p = _u32(T_STRING) + _u64(len(items))
    for s in items:
        p += _str(s)
    return _kv(k, T_ARRAY, p)


def _kv_arr_f32(k, items: List[float]) -> bytes:
    return _kv(k, T_ARRAY, _u32(T_FLOAT32) + _u64(len(items)) + b"".join(_f32(x) for x in items))


def _kv_arr_i32(k, items: List[int]) -> bytes:
    return _kv(k, T_ARRAY, _u32(T_INT32) + _u64(len(items)) + b"".join(_i32(x) for x in items))


def build_vocab() -> Tuple[List[str], List[float], List[int]]:
    tokens: List[str] = ["<unk>", "<s>", "</s>"]
    types: List[int] = [2, 3, 3]          # UNKNOWN, CONTROL, CONTROL
    scores: List[float] = [0.0, 0.0, 0.0]
    # I token BYTE servono al tokenizer per accettare testo arbitrario.
    # Per evitare che il modello li *generi* (byte non-UTF8 che llama-server
    # rifiuta di serializzare) azzeriamo le loro righe in output.weight.
    for i in range(256):
        tokens.append(f"<0x{i:02X}>")
        types.append(6)                    # BYTE
        scores.append(-20.0)
    pieces = ["▁", "▁il", "▁la", "▁di", "▁che", "▁e",
              "▁un", "▁per", "▁con", "▁non", "▁si",
              "▁memoria", "▁modello", "▁RAM", "▁token",
              "a", "e", "i", "o", "u", "n", "r", "t", "s", "l", "c", "d", "m", "p", "v",
              ".", ","]
    for j, piece in enumerate(pieces[:N_VOCAB_EXTRA]):
        tokens.append(piece)
        types.append(1)                    # NORMAL
        scores.append(float(-j))
    return tokens, scores, types


def tensors_spec() -> List[Tuple[str, List[int]]]:
    t: List[Tuple[str, List[int]]] = []
    n_vocab = 3 + 256 + N_VOCAB_EXTRA
    t.append(("token_embd.weight", [N_EMBD, n_vocab]))
    for i in range(N_LAYER):
        p = f"blk.{i}."
        t += [
            (p + "attn_norm.weight", [N_EMBD]),
            (p + "attn_q.weight", [N_EMBD, N_HEAD * HEAD_DIM]),
            (p + "attn_k.weight", [N_EMBD, N_HEAD_KV * HEAD_DIM]),
            (p + "attn_v.weight", [N_EMBD, N_HEAD_KV * HEAD_DIM]),
            (p + "attn_output.weight", [N_HEAD * HEAD_DIM, N_EMBD]),
            (p + "ffn_norm.weight", [N_EMBD]),
            (p + "ffn_gate.weight", [N_EMBD, N_FF]),
            (p + "ffn_up.weight", [N_EMBD, N_FF]),
            (p + "ffn_down.weight", [N_FF, N_EMBD]),
        ]
    t.append(("output_norm.weight", [N_EMBD]))
    t.append(("output.weight", [N_EMBD, n_vocab]))
    return t


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tiny-llama-test-F32.gguf")
    rng = random.Random(7)
    tokens, scores, types = build_vocab()
    n_vocab = len(tokens)

    kvs = [
        _kv_str("general.architecture", "llama"),
        _kv_str("general.name", "Argine-tiny-test"),
        _kv_u32("general.file_type", 0),
        _kv_u32("llama.block_count", N_LAYER),
        _kv_u32("llama.context_length", CTX),
        _kv_u32("llama.embedding_length", N_EMBD),
        _kv_u32("llama.feed_forward_length", N_FF),
        _kv_u32("llama.attention.head_count", N_HEAD),
        _kv_u32("llama.attention.head_count_kv", N_HEAD_KV),
        _kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        _kv_u32("llama.rope.dimension_count", HEAD_DIM),
        _kv_f32("llama.rope.freq_base", 10000.0),
        _kv_u32("llama.vocab_size", n_vocab),
        _kv_str("tokenizer.ggml.model", "llama"),
        _kv_str("tokenizer.ggml.pre", "default"),
        _kv_arr_str("tokenizer.ggml.tokens", tokens),
        _kv_arr_f32("tokenizer.ggml.scores", scores),
        _kv_arr_i32("tokenizer.ggml.token_type", types),
        _kv_u32("tokenizer.ggml.bos_token_id", 1),
        _kv_u32("tokenizer.ggml.eos_token_id", 2),
        _kv_u32("tokenizer.ggml.unknown_token_id", 0),
        _kv_bool("tokenizer.ggml.add_bos_token", True),
        _kv_bool("tokenizer.ggml.add_eos_token", False),
        # template ChatML standard: e' quello che llama-server riconosce meglio
        _kv_str("tokenizer.chat_template", CHATML_TEMPLATE),
    ]
    metadata = b"".join(kvs)

    specs = tensors_spec()
    table = b""
    offset = 0
    for name, dims in specs:
        table += _str(name) + _u32(len(dims))
        for d in dims:
            table += _u64(d)
        table += _u32(F32) + _u64(offset)
        nb = 4
        for d in dims:
            nb *= d
        pad = (ALIGN - (nb % ALIGN)) % ALIGN
        offset += nb + pad

    head = b"GGUF" + _u32(3) + _u64(len(specs)) + _u64(len(kvs)) + metadata + table
    pad = (ALIGN - (len(head) % ALIGN)) % ALIGN
    head += b"\x00" * pad

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(head)
        for name, dims in specs:
            ne = 1
            for d in dims:
                ne *= d
            if "norm" in name:
                data = struct.pack(f"<{ne}f", *([1.0] * ne))
            elif name == "output.weight":
                # righe per token: [vocab][n_embd]. Azzeriamo i token BYTE (id 3..258)
                # cosi' il loro logit resta 0 e i token normali (varianza alta) vincono.
                vals: List[float] = []
                for tok_id in range(ne // N_EMBD):
                    if 3 <= tok_id < 3 + 256:
                        vals.extend([0.0] * N_EMBD)
                    else:
                        vals.extend(rng.gauss(0.0, 0.5) for _ in range(N_EMBD))
                data = struct.pack(f"<{ne}f", *vals)
            else:
                data = struct.pack(f"<{ne}f", *[rng.gauss(0.0, 0.02) for _ in range(ne)])
            fh.write(data)
            nb = len(data)
            fh.write(b"\x00" * ((ALIGN - (nb % ALIGN)) % ALIGN))
    print(f"{out}  {out.stat().st_size/1024:.1f} KiB, vocab {n_vocab}, "
          f"{len(specs)} tensori")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- MoE reale ---

def make_tiny_moe(out: Path, n_expert: int = 8, n_used: int = 2,
                  n_layer: int = 2, n_embd: int = 64, n_head: int = 4,
                  n_head_kv: int = 4, n_ff_exp: int = 32) -> None:
    """GGUF qwen3moe minuscolo ma realmente caricabile.

    Serve a collaudare il log del routing e il riordino degli expert senza
    scaricare 13 GiB.
    """
    rng = random.Random(11)
    head_dim = n_embd // n_head
    tokens, scores, types = build_vocab()
    n_vocab = len(tokens)
    arch = "qwen3moe"

    kvs = [
        _kv_str("general.architecture", arch),
        _kv_str("general.name", "Argine-tiny-moe"),
        _kv_u32("general.file_type", 0),
        _kv_u32(f"{arch}.block_count", n_layer),
        _kv_u32(f"{arch}.context_length", CTX),
        _kv_u32(f"{arch}.embedding_length", n_embd),
        _kv_u32(f"{arch}.feed_forward_length", n_ff_exp),
        _kv_u32(f"{arch}.expert_feed_forward_length", n_ff_exp),
        _kv_u32(f"{arch}.expert_count", n_expert),
        _kv_u32(f"{arch}.expert_used_count", n_used),
        _kv_u32(f"{arch}.attention.head_count", n_head),
        _kv_u32(f"{arch}.attention.head_count_kv", n_head_kv),
        _kv_u32(f"{arch}.attention.key_length", head_dim),
        _kv_u32(f"{arch}.attention.value_length", head_dim),
        _kv_f32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6),
        _kv_u32(f"{arch}.rope.dimension_count", head_dim),
        _kv_f32(f"{arch}.rope.freq_base", 10000.0),
        _kv_u32(f"{arch}.vocab_size", n_vocab),
        _kv_str("tokenizer.ggml.model", "llama"),
        _kv_str("tokenizer.ggml.pre", "default"),
        _kv_arr_str("tokenizer.ggml.tokens", tokens),
        _kv_arr_f32("tokenizer.ggml.scores", scores),
        _kv_arr_i32("tokenizer.ggml.token_type", types),
        _kv_u32("tokenizer.ggml.bos_token_id", 1),
        _kv_u32("tokenizer.ggml.eos_token_id", 2),
        _kv_u32("tokenizer.ggml.unknown_token_id", 0),
        _kv_bool("tokenizer.ggml.add_bos_token", True),
        _kv_bool("tokenizer.ggml.add_eos_token", False),
        _kv_str("tokenizer.chat_template", CHATML_TEMPLATE),
    ]

    specs: List[Tuple[str, List[int]]] = [
        ("token_embd.weight", [n_embd, n_vocab]),
    ]
    for i in range(n_layer):
        p = f"blk.{i}."
        specs += [
            (p + "attn_norm.weight", [n_embd]),
            (p + "attn_q.weight", [n_embd, n_head * head_dim]),
            (p + "attn_k.weight", [n_embd, n_head_kv * head_dim]),
            (p + "attn_v.weight", [n_embd, n_head_kv * head_dim]),
            (p + "attn_output.weight", [n_head * head_dim, n_embd]),
            (p + "attn_q_norm.weight", [head_dim]),
            (p + "attn_k_norm.weight", [head_dim]),
            (p + "ffn_norm.weight", [n_embd]),
            (p + "ffn_gate_inp.weight", [n_embd, n_expert]),
            (p + "ffn_gate_exps.weight", [n_embd, n_ff_exp, n_expert]),
            (p + "ffn_up_exps.weight", [n_embd, n_ff_exp, n_expert]),
            (p + "ffn_down_exps.weight", [n_ff_exp, n_embd, n_expert]),
        ]
    specs += [("output_norm.weight", [n_embd]), ("output.weight", [n_embd, n_vocab])]

    table = b""
    offset = 0
    for name, dims in specs:
        table += _str(name) + _u32(len(dims))
        for d in dims:
            table += _u64(d)
        table += _u32(F32) + _u64(offset)
        nb = 4
        for d in dims:
            nb *= d
        offset += nb + ((ALIGN - (nb % ALIGN)) % ALIGN)

    head = b"GGUF" + _u32(3) + _u64(len(specs)) + _u64(len(kvs)) + b"".join(kvs) + table
    head += b"\x00" * ((ALIGN - (len(head) % ALIGN)) % ALIGN)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(head)
        for name, dims in specs:
            ne = 1
            for d in dims:
                ne *= d
            if "norm" in name:
                data = struct.pack(f"<{ne}f", *([1.0] * ne))
            elif name == "output.weight":
                vals: List[float] = []
                for tok in range(ne // n_embd):
                    vals.extend([0.0] * n_embd if 3 <= tok < 3 + 256
                                else [rng.gauss(0.0, 0.5) for _ in range(n_embd)])
                data = struct.pack(f"<{ne}f", *vals)
            elif "gate_inp" in name:
                # router: pesi ben separati, cosi' le scelte variano fra i token
                data = struct.pack(f"<{ne}f", *[rng.gauss(0.0, 1.0) for _ in range(ne)])
            else:
                data = struct.pack(f"<{ne}f", *[rng.gauss(0.0, 0.05) for _ in range(ne)])
            fh.write(data)
            fh.write(b"\x00" * ((ALIGN - (len(data) % ALIGN)) % ALIGN))
    print(f"{out}  {out.stat().st_size/1024:.1f} KiB, {n_expert} expert ({n_used} attivi), "
          f"{n_layer} layer, vocab {n_vocab}")
