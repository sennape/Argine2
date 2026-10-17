"""Interpretable I/O stall model used to turn replayed misses into time.

Replay counts misses and bytes exactly; those numbers are model-free and are
always reported first. Converting them into stall time needs a model of the
device and of the runtime's read path. This one is deliberately simple and
fully exposed, so that it can be checked against the runtime's own I/O
timeline (iotimeline.py) and replaced when it is wrong.

Model (per MoE node, i.e. per (step, layer) event)
---------------------------------------------------
With the emit-and-continue read path every thread issues the reads of its
misses and waits for them at the barrier, so the reads of one node are in
flight together. If a node misses m experts, it issues r = m * slices_per_expert
reads of total size B bytes. The device serves them at the aggregate bandwidth
it reaches at queue depth qd = min(r, max_queue_depth):

    stall(node) = B / BW(qd) + r * per_read_overhead

BW(qd) is interpolated from a measured table. Nothing of the node's I/O is
assumed to overlap with matrix compute: in Argine's runtime the barrier after
the reads precedes the kernel, so it cannot (see docs/ARCHITECTURE.md).

Default parameters
------------------
[MEASURED] on the reference machine by `argine diskbench`, pattern `moe`,
896 KiB slices (Argine README, "Il margine vero"): aggregate MB/s by number of
concurrent readers 1:1338, 2:2335, 3:2777, 4:3069, 6:3398, 8:3563.
[DERIVED] The store reached 84% of that curve in generation (barrier between
threads); `efficiency` defaults to 0.84 to reflect it.
`per_read_overhead_us` defaults to 0: the commit/decommit/lock bookkeeping was
measured as a share of I/O time on two models (10% on the 30B, 56% on the 80B),
not per read; treat the default model as optimistic on the 80B and calibrate
`per_read_overhead_us` from a runtime trace (`argine research iotimeline`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

MB = 1_000_000  # diskbench reports decimal MB/s

DEFAULT_BW_BY_QD = {1: 1338.0, 2: 2335.0, 3: 2777.0, 4: 3069.0, 6: 3398.0, 8: 3563.0}


@dataclass
class CostModel:
    bw_by_qd_mb_s: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_BW_BY_QD))
    efficiency: float = 0.84
    per_read_overhead_us: float = 0.0
    slices_per_expert: int = 3
    max_queue_depth: int = 8
    source: str = ("argine diskbench (pattern moe, 896 KiB, i5-1340P NVMe) x 0.84 store "
                   "efficiency; per-read overhead not calibrated")

    def bandwidth(self, qd: float) -> float:
        """Aggregate bytes/s at queue depth qd (linear interpolation, clamped)."""
        pts = sorted((int(k), float(v)) for k, v in self.bw_by_qd_mb_s.items())
        if not pts:
            raise ValueError("empty bandwidth table")
        qd = max(pts[0][0], min(qd, pts[-1][0]))
        for (q0, b0), (q1, b1) in zip(pts, pts[1:]):
            if q0 <= qd <= q1:
                f = (qd - q0) / (q1 - q0) if q1 != q0 else 0.0
                return (b0 + f * (b1 - b0)) * MB * self.efficiency
        return pts[-1][1] * MB * self.efficiency

    def node_stall_s(self, n_miss_experts: int, miss_bytes: int) -> float:
        if n_miss_experts <= 0:
            return 0.0
        reads = n_miss_experts * self.slices_per_expert
        qd = min(reads, self.max_queue_depth)
        return miss_bytes / self.bandwidth(qd) + reads * self.per_read_overhead_us * 1e-6

    def to_json(self) -> Dict[str, object]:
        d = asdict(self)
        d["bw_by_qd_mb_s"] = {str(k): v for k, v in self.bw_by_qd_mb_s.items()}
        return d

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "CostModel":
        d = dict(d)
        bw = d.pop("bw_by_qd_mb_s", None)
        m = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})  # type: ignore[arg-type]
        if bw:
            m.bw_by_qd_mb_s = {int(k): float(v) for k, v in dict(bw).items()}  # type: ignore[arg-type]
        return m

    @classmethod
    def load(cls, path: Optional[str | Path]) -> "CostModel":
        if not path:
            return cls()
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
