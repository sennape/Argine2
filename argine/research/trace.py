"""Router trace format ("argine-router-trace" v1) and converters.

A router trace records, for every forward pass of the model, which experts the
router selected in every MoE layer. It is independent of residency: the same
trace can be replayed through any cache or allocation policy without running
the model again (counterfactual evaluation).

Data model
----------
A trace is a header (``TraceMeta``) plus a list of ``Step`` objects. One step
is one forward pass (one llama.cpp graph = one ubatch):

* during generation a step has ``n_tokens == 1``;
* during prompt processing a step may carry many tokens, and the runtime needs
  the *union* of their experts resident at the same time (this is why the
  replay engine is event-atomic, see ``replay.py``).

``Step.routes[layer]`` is a list with one tuple of selected expert ids per
token, in router order, exactly as llama.cpp produced them.

On disk
-------
JSON Lines, optionally gzip-compressed (``.jsonl.gz``)::

    {"format": "argine-router-trace", "version": 1, "meta": {...}}
    {"g": 0, "n": 77, "ph": "prefill", "seg": "coding", "t": 123, "r": {"0": [[..], ..], ..}}
    {"g": 1, "n": 1,  "ph": "decode",  ...}

The first line is the header. Every other line is one step. Layer keys are
strings because JSON object keys are strings.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

FORMAT = "argine-router-trace"
VERSION = 1

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_UNKNOWN = "unknown"


class TraceError(ValueError):
    pass


@dataclass
class TraceMeta:
    model: str = ""
    n_experts: int = 0
    top_k: int = 0
    layers: List[int] = field(default_factory=list)
    # bytes needed to make one expert of a layer resident (gate+up+down);
    # keys are layer ids as strings in JSON, ints in memory
    expert_bytes: Dict[int, int] = field(default_factory=dict)
    source: str = ""
    created: str = ""
    workload: str = ""
    split: str = ""              # "dev" | "eval" | "" (unknown / legacy)
    provenance: str = ""         # free text: how the trace was obtained
    fingerprint: Dict[str, object] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        d = asdict(self)
        d["expert_bytes"] = {str(k): int(v) for k, v in self.expert_bytes.items()}
        return d

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "TraceMeta":
        d = dict(d)
        eb = d.pop("expert_bytes", {}) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        m = cls(**{k: v for k, v in d.items() if k in known})
        m.expert_bytes = {int(k): int(v) for k, v in dict(eb).items()}  # type: ignore[arg-type]
        m.layers = [int(x) for x in m.layers]
        return m

    def bytes_of(self, layer: int) -> int:
        """Bytes of one resident expert in `layer` (1 if unknown: counts)."""
        return int(self.expert_bytes.get(layer, 1)) or 1


@dataclass
class Step:
    graph: int
    n_tokens: int
    routes: Dict[int, List[Tuple[int, ...]]]
    phase: str = PHASE_UNKNOWN
    segment: Optional[str] = None
    t_ns: Optional[int] = None

    def experts(self, layer: int) -> Tuple[int, ...]:
        """Union of experts needed by this step in `layer`, sorted (event set)."""
        sel = self.routes.get(layer)
        if not sel:
            return ()
        if len(sel) == 1:
            return tuple(sorted(set(sel[0])))
        u = set()
        for tok in sel:
            u.update(tok)
        return tuple(sorted(u))

    def to_json(self) -> Dict[str, object]:
        d: Dict[str, object] = {"g": self.graph, "n": self.n_tokens, "ph": self.phase,
                                "r": {str(l): [list(t) for t in sel]
                                      for l, sel in sorted(self.routes.items())}}
        if self.segment is not None:
            d["seg"] = self.segment
        if self.t_ns is not None:
            d["t"] = self.t_ns
        return d

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "Step":
        routes = {int(l): [tuple(int(x) for x in tok) for tok in sel]  # type: ignore[union-attr]
                  for l, sel in dict(d["r"]).items()}  # type: ignore[arg-type]
        return cls(graph=int(d["g"]), n_tokens=int(d["n"]), routes=routes,  # type: ignore[arg-type]
                   phase=str(d.get("ph", PHASE_UNKNOWN)),
                   segment=d.get("seg"),  # type: ignore[arg-type]
                   t_ns=d.get("t"))  # type: ignore[arg-type]


@dataclass
class Trace:
    meta: TraceMeta
    steps: List[Step]

    # ------------------------------------------------------------ views ---
    @property
    def layers(self) -> List[int]:
        if self.meta.layers:
            return list(self.meta.layers)
        s = set()
        for st in self.steps:
            s.update(st.routes)
        return sorted(s)

    def n_tokens(self, phase: Optional[str] = None) -> int:
        return sum(st.n_tokens for st in self.steps if phase is None or st.phase == phase)

    def select(self, *, phase: Optional[str] = None, segment: Optional[str] = None,
               graphs: Optional[Tuple[int, int]] = None) -> "Trace":
        """A sub-trace. `graphs` is a half-open [first, last) range of step indices."""
        steps = self.steps
        if graphs is not None:
            steps = steps[graphs[0]:graphs[1]]
        if phase is not None:
            steps = [s for s in steps if s.phase == phase]
        if segment is not None:
            steps = [s for s in steps if s.segment == segment]
        return Trace(meta=self.meta, steps=list(steps))

    def segments(self) -> List[str]:
        out: List[str] = []
        for st in self.steps:
            if st.segment is not None and st.segment not in out:
                out.append(st.segment)
        return out

    def token_selections(self, layer: int, phase: Optional[str] = None
                         ) -> Iterator[Tuple[int, Tuple[int, ...]]]:
        """(token_index, selected experts) for every token, in order.

        Token index counts tokens across steps (a prefill step of 77 tokens
        advances it by 77). Used by the routing statistics, which are about
        what the router chose per token, not about residency events.
        """
        t = 0
        for st in self.steps:
            sel = st.routes.get(layer)
            if phase is not None and st.phase != phase:
                t += st.n_tokens
                continue
            if sel:
                for i, tok in enumerate(sel):
                    yield t + i, tok
            t += st.n_tokens

    def events(self) -> Iterator[Tuple[int, int, Tuple[int, ...]]]:
        """Residency events in execution order: (step_index, layer, expert set).

        Within a step layers are visited in ascending order, which is the order
        in which llama.cpp executes the MoE nodes.
        """
        layers = self.layers
        for i, st in enumerate(self.steps):
            for l in layers:
                ex = st.experts(l)
                if ex:
                    yield i, l, ex

    # ------------------------------------------------------------- I/O ----
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "wt", encoding="utf-8") as fh:  # type: ignore[operator]
            fh.write(json.dumps({"format": FORMAT, "version": VERSION,
                                 "meta": self.meta.to_json()}, sort_keys=True) + "\n")
            for st in self.steps:
                fh.write(json.dumps(st.to_json(), separators=(",", ":")) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        p = Path(path)
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
            first = fh.readline()
            try:
                head = json.loads(first)
            except json.JSONDecodeError as e:
                raise TraceError(f"{p}: first line is not JSON") from e
            if head.get("format") != FORMAT:
                raise TraceError(f"{p}: not an {FORMAT} file")
            if int(head.get("version", 0)) != VERSION:
                raise TraceError(f"{p}: unsupported version {head.get('version')}")
            meta = TraceMeta.from_json(head.get("meta", {}))
            steps = [Step.from_json(json.loads(line)) for line in fh if line.strip()]
        return cls(meta=meta, steps=steps)


# ------------------------------------------------------------ converters ---

def _layer_kind(name: str) -> Tuple[Optional[int], Optional[str]]:
    # "blk.12.ffn_gate_exps.weight" -> (12, "gate")
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "blk":
        return None, None
    try:
        layer = int(parts[1])
    except ValueError:
        return None, None
    for kind in ("gate", "up", "down"):
        if f"ffn_{kind}_exps" in parts[2]:
            return layer, kind
    return None, None


def from_route_log(path: str | Path, *, model: str = "", n_experts: int = 0,
                   provenance: str = "") -> Tuple[Trace, List[str]]:
    """Convert Argine's legacy ``ARGINE_ROUTE_LOG`` output into a trace.

    Format of the legacy log, one line per MUL_MAT_ID call (gate, up and down of
    every layer, with identical ids)::

        <tensor_name> <n_tokens> <n_used> <id> <id> ...

    Forward-pass boundaries are not written by the legacy patch: a new step is
    inferred whenever the layer index decreases. Only the gate rows are used
    (up/down repeat the same ids); a mismatch between them is reported.

    Returns the trace and a list of human-readable warnings (truncated last
    line, incomplete final pass, gate/up/down disagreement).
    """
    warnings: List[str] = []
    rows: List[Tuple[int, str, int, int, List[int]]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            f = line.split()
            if not f:
                continue
            layer, kind = _layer_kind(f[0])
            if layer is None or len(f) < 3:
                warnings.append(f"line {ln}: unreadable record ({line.strip()[:40]!r}), skipped")
                continue
            try:
                n_tok, n_used = int(f[1]), int(f[2])
                ids = [int(x) for x in f[3:]]
            except ValueError:
                warnings.append(f"line {ln}: non-numeric field, skipped")
                continue
            if len(ids) != n_tok * n_used:
                warnings.append(f"line {ln}: {len(ids)} ids, expected {n_tok * n_used} "
                                "(truncated write?), skipped")
                continue
            rows.append((layer, kind or "", n_tok, n_used, ids))

    steps: List[Step] = []
    cur: Dict[int, List[Tuple[int, ...]]] = {}
    cur_n = 0
    prev_layer = -1
    other: Dict[Tuple[int, int, str], List[int]] = {}
    top_k = 0
    for layer, kind, n_tok, n_used, ids in rows:
        if kind != "gate":
            other[(len(steps), layer, kind)] = ids
            continue
        if layer <= prev_layer and cur:
            steps.append(Step(graph=len(steps), n_tokens=cur_n, routes=cur))
            cur = {}
            cur_n = 0
        prev_layer = layer
        # max over layers, not the last layer's count: llama.cpp computes only
        # the tokens whose output is needed in the LAST layer (one, for a
        # prompt), so a 77-token prompt step has 77 tokens in layers 0..46
        # and 1 token in layer 47.
        cur_n = n_tok if not cur else max(cur_n, n_tok)
        top_k = max(top_k, n_used)
        cur[layer] = [tuple(ids[i * n_used:(i + 1) * n_used]) for i in range(n_tok)]
    if cur:
        steps.append(Step(graph=len(steps), n_tokens=cur_n, routes=cur))

    # gate/up/down must agree: they are the same router decision
    mismatch = 0
    for (gi, layer, kind), ids in other.items():
        if gi < len(steps) and layer in steps[gi].routes:
            flat = [x for tok in steps[gi].routes[layer] for x in tok]
            if flat != ids:
                mismatch += 1
    if mismatch:
        warnings.append(f"{mismatch} up/down rows disagree with their gate row")

    layers = sorted({l for st in steps for l in st.routes})
    if steps and len(steps[-1].routes) < len(layers):
        warnings.append(f"last forward pass incomplete ({len(steps[-1].routes)} of "
                        f"{len(layers)} layers): dropped")
        steps.pop()
    for st in steps:
        st.phase = PHASE_DECODE if st.n_tokens == 1 else PHASE_PREFILL

    if not n_experts:
        mx = -1
        for st in steps:
            for sel in st.routes.values():
                for tok in sel:
                    mx = max(mx, max(tok) if tok else -1)
        n_experts = mx + 1
        warnings.append(f"n_experts not given: inferred {n_experts} from the largest id "
                        "(a lower bound)")
    meta = TraceMeta(model=model, n_experts=n_experts, top_k=top_k, layers=layers,
                     source=str(Path(path).name), provenance=provenance or
                     "legacy ARGINE_ROUTE_LOG (patch 0001); pass boundaries inferred")
    return Trace(meta=meta, steps=steps), warnings


def split_requests(trace: Trace) -> List[Tuple[int, int]]:
    """Index ranges [first, last) of requests: a request starts at a prefill step.

    Generation steps that precede the first prefill (warm-up) form their own
    range. Useful to separate repeated benchmark requests in legacy logs.
    """
    out: List[Tuple[int, int]] = []
    start = 0
    for i, st in enumerate(trace.steps):
        if st.phase == PHASE_PREFILL and i > start:
            # consecutive prefill steps (multi-ubatch prompt) stay together
            if trace.steps[i - 1].phase != PHASE_PREFILL:
                out.append((start, i))
                start = i
    if start < len(trace.steps):
        out.append((start, len(trace.steps)))
    return out
