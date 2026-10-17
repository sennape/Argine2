"""Read-only audit of generated-text provenance in recorded research runs.

    python tools/audit_text_provenance.py ROOT [--out CSV]

Walks ROOT for run directories (containing responses.jsonl) and classifies
every record with argine.research.textprov.status_of. Nothing in the run
directories is modified: historical hashes stay as recorded, and runs from
before the 2026-10-04 fix are reported as provenance-limited.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argine.research import textprov  # noqa: E402
from argine.research.io import write_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.root).rglob("responses.jsonl")):
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        sm = textprov.summarize(recs)
        rows.append({"run_dir": str(f.parent), "requests": sm["requests"], "verdict": sm["verdict"],
                     "counts": json.dumps(sm["counts"], sort_keys=True),
                     "generated_tokens": sum(int(r.get("predicted_n") or 0) for r in recs)})
        print(f"{f.parent}: {sm['verdict']} {sm['counts']}")
    if a.out:
        write_csv(rows, Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
