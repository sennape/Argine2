# Superseded workload files

Historical workload files kept for provenance. They are **not canonical**: the loader never lists this folder (`argine research workloads` only reads `v<N>/`), and no EXP result uses them.

| file | SHA-256 | used by | superseded by |
|---|---|---|---|
| `v2-multi-turn-n1024-ff89429e96c0.json` | `ff89429e96c0e8db9dca6b69bc85307b6f8f6206d25fde83b7f2cb7e6aacfe93` | the smoke run `experiments/EXP-007-router-characterization/raw/30b-multi-turn-v2-smoke-4G` only (all four DEV turns ended with `finish_reason` = `length`) | `v2/multi-turn.json` @ `e978bde0341f` (`n_predict` 1536), canonical for EXP-007 |

**How the file was obtained (2026-10-11).**
- It was reconstructed from `v2/multi-turn.json` (`e978bde0341f`) by changing the nine `n_predict` values from 1536 to 1024 and converting the line endings from CRLF back to LF.
- Its SHA-256 equals the `workload_sha256` recorded in the smoke run's `manifest.json`, so it is byte-identical to the file that run used.
- A copy of the original 2026-10-10 file held in the session workspace has the same hash.
- Do not edit it: any change breaks the hash match.

See [`../../docs/research/MULTITURN-V2-HISTORY.md`](../../docs/research/MULTITURN-V2-HISTORY.md).
