"""Deterministic writers for research outputs (CSV / JSON).

Outputs are meant to be diffed and re-derived: keys are sorted, floats are
rounded to a fixed number of significant digits, and no timestamp is written
unless the caller puts one in the data on purpose.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _clean(x: Any, digits: int = 8) -> Any:
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
        if x == 0:
            return 0.0
        return float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {str(k): _clean(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v, digits) for v in x]
    return x


def write_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            json.dumps(_clean(obj), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )
    return p


def write_csv(rows: Sequence[Dict[str, Any]], path: str | Path,
              columns: Sequence[str] | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = list(columns) if columns else []
    if not cols:
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
    fh,
    fieldnames=cols,
    extrasaction="ignore",
    lineterminator="\n",
)
        w.writeheader()
        for r in rows:
            w.writerow({k: _clean(r.get(k)) for k in cols})
    return p


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
