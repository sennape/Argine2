# Data

| path | what | provenance |
|---|---|---|
| `raw/argine-v1/runs.jsonl` | 163 measurement records of Argine v1 (2026-09-27 → 10-03) | `%LOCALAPPDATA%\Argine\logs\runs.jsonl`, imported by `tools/import_argine_logs.py`; host name and Windows user name replaced, nothing else changed; `IMPORT.json` holds original and sanitised SHA-256 |
| `raw/argine-v1/perplexity.jsonl` | 8 perplexity measurements | same |
| `raw/legacy/route.log` | router log of 2026-09-27 (Qwen3-30B-A3B, patch 0001, mmap era) | moved from the root of the Argine folder |
| `raw/legacy/buffers*.txt` | llama.cpp buffer printouts of 2026-09-27 | moved from the root of the Argine folder |
| `processed/argine-v1-runs.csv` | one row per speed repetition (353 rows) | `tools/import_argine_logs.py` |
| `processed/legacy-30b-route-request1.jsonl.gz` | request 1 of `route.log` in the Argine2 trace format; expert sizes derived (uniform) | `argine research convert` (command in EXP-007) |
| `processed/pilot-legacy-30b/` | full pilot analysis of that trace + REPORT.md + figures | `argine research analyze` (command in EXP-009) |

Everything under `processed/` is reproducible from `raw/` with the commands in the README's Reproduction section; `tests/test_research.py` checks that the historical experiment tables regenerate byte-identically.
