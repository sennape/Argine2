"""Import Argine's original measurement logs into Argine2 as raw evidence.

    python tools/import_argine_logs.py %LOCALAPPDATA%\\Argine\\logs data/raw/argine-v1

Reads runs.jsonl and perplexity.jsonl, writes sanitised copies (host name and
Windows user name replaced; nothing else changed, line order preserved) and a
flat CSV of every speed measurement to data/processed/argine-v1-runs.csv.

The sanitisation is the ONLY transformation. It exists because the repository
is meant to be public; the originals stay where Argine wrote them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPLACE = [
    # Replace the recorded host value without embedding the original machine name.
    (re.compile(r'("host"\s*:\s*")[^"]+(")'), r"\1reference-machine\2"),
    # Replace any Windows user component in both JSON-escaped and ordinary paths.
    (re.compile(r"(?i)(C:\\\\+Users\\\\+)[^\\\"]+"), r"\1USER"),
    (re.compile(r"(?i)(C:\\+Users\\+)[^\\\"]+"), r"\1USER"),
]


def sanitise(line: str) -> str:
    for rx, rep in REPLACE:
        line = rx.sub(rep, line)
    return line


def flat_rows(records):
    for r in records:
        m = r.get("metrics") or {}
        if not isinstance(m, dict) or "runs" not in m:
            continue
        model = r.get("model") or {}
        plan = r.get("plan") or {}
        rt = r.get("runtime") or {}
        res = m.get("resources") or {}
        for i, run in enumerate(m.get("runs") or []):
            yield {
                "ts_iso": r.get("ts_iso"), "event": r.get("event"), "rep": i + 1,
                "model": model.get("name"), "architecture": model.get("architecture"),
                "quantization": model.get("quantization"), "bpw": model.get("bits_per_weight"),
                "expert_count": model.get("expert_count"), "expert_used": model.get("expert_used"),
                "build": rt.get("build"), "backend": rt.get("backend") or plan.get("backend"),
                "threads": plan.get("threads"), "ctx": plan.get("ctx"),
                "expert_store": plan.get("expert_store"),
                "expert_budget": r.get("expert_budget") or plan.get("expert_cache_budget"),
                "resident_fraction": r.get("frazione_residente"),
                "read_mode": r.get("modo_lettura"),
                "prompt_tokens": run.get("prompt_tokens"),
                "generated_tokens": run.get("generated_tokens"),
                "gen_tok_s": run.get("gen_tok_s"), "prompt_tok_s": run.get("prompt_tok_s"),
                "ttft_s": run.get("ttft_s"),
                "ram_peak": res.get("ram_peak"), "disk_read_mb_s": res.get("sys_disk_read_mb_s"),
            }


def main(src: str, dst: str) -> int:
    s, d = Path(src), Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ("runs.jsonl", "perplexity.jsonl"):
        p = s / name
        if not p.exists():
            continue
        raw = p.read_text(encoding="utf-8")
        clean = "".join(sanitise(l) for l in raw.splitlines(keepends=True))
        (d / name).write_text(clean, encoding="utf-8")
        manifest[name] = {"records": sum(1 for l in clean.splitlines() if l.strip()),
                          "sha256_sanitised": hashlib.sha256(clean.encode()).hexdigest(),
                          "sha256_original": hashlib.sha256(raw.encode()).hexdigest()}
    (d / "IMPORT.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    recs = [json.loads(l) for l in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = list(flat_rows(recs))
    out = d.parent.parent / "processed" / "argine-v1-runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(recs)} records, {len(rows)} speed measurements -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
