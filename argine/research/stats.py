"""Statistics for repeated end-to-end measurements (section 27).

Minimum 3 runs, preferably 5. Every summary reports mean, median, standard
deviation, coefficient of variation and a 95% confidence interval of the mean
(Student t). Configurations are interleaved (A B C B A C ...) so that thermal
or temporal drift hits all of them instead of favouring one: in Argine, two
runs of the same curve six hours apart differed by up to 5% while repetitions
inside one session differed by 1-4%.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

# two-sided 95% Student t quantiles, df = 1..30
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    return _T95[df - 1] if df <= 30 else 1.96


def summarize(xs: Sequence[float]) -> Dict[str, float]:
    v = [float(x) for x in xs if x is not None]
    n = len(v)
    if n == 0:
        return {"n": 0}
    mean = sum(v) / n
    s = sorted(v)
    median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    half = t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "median": median, "sd": sd,
            "cv": sd / mean if mean else float("nan"),
            "ci95_low": mean - half if n > 1 else float("nan"),
            "ci95_high": mean + half if n > 1 else float("nan"),
            "min": s[0], "max": s[-1],
            "enough_repetitions": n >= 3}


def interleaved_schedule(configs: Sequence[str], repeats: int) -> List[str]:
    """Order of runs such that every configuration appears once per round and
    the order changes between rounds (rotation, reversed on odd rounds).
    For A,B,C and 3 rounds: A B C | A C B -> see tests for the exact order."""
    cfg = list(configs)
    out: List[str] = []
    n = len(cfg)
    for r in range(repeats):
        rot = cfg[r % n:] + cfg[:r % n] if n else []
        if r % 2 == 1:
            rot = list(reversed(rot))
        out.extend(rot)
    return out


def relative_difference(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, object]:
    """b relative to a, and whether their 95% CIs overlap (overlap = the
    difference is not established by these measurements)."""
    if not a.get("n") or not b.get("n"):
        return {"rel": None, "ci_overlap": None}
    rel = (b["mean"] - a["mean"]) / a["mean"] if a["mean"] else None
    overlap = None
    if a["n"] > 1 and b["n"] > 1:
        overlap = not (a["ci95_high"] < b["ci95_low"] or b["ci95_high"] < a["ci95_low"])
    return {"rel": rel, "ci_overlap": overlap}
