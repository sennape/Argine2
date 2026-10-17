# EXP-015 — PRETOUCH: page zeroing moved, not added

| field | value |
|---|---|
| ID | `EXP-015-pretouch` |
| Status | **SUPPORTED** |
| Period | 2026-10-02 → 2026-10-03 |
| Origin | Argine v1 (historical); summary measurements transcribed from the historical Argine v1 README; raw run linkage incomplete |
| Related questions | RQ-015 |
| Supersedes / superseded | — |

> **Title scope (2026-10-11 audit).** The title preserves the historical mechanism interpretation. The 2026-10-11 audit supports a weaker statement: the preserved store-counter measurements are quantitatively consistent with substantial displacement of page-initialization work, but the mechanism is not independently raw-reproducible.

## Research question

In the historical Qwen3-30B Q4_K_M expert-store measurements, how did PRETOUCH relate to reported read-per-miss cost and the fitted speed/residency slope?

*(Original wording, kept as history: "Does touching freshly committed pages before the read (PRETOUCH) change the cost per miss and the slope of the speed/residency curve?")*

## Hypothesis

[HYPOTHESIS] PRETOUCH moves Windows' page zeroing out of the DMA instead of adding work; it raises the intercept, not the slope.

*Audit reading (2026-10-11):*
- The displacement mechanism is supported only as provenance-limited consistency evidence. The preserved counters agree numerically with it, but the PRETOUCH flag and the per-miss counters are not in the archived raw logs, and no OS-level instrumentation isolates page-zeroing causality.
- "Not added" is too absolute. The explicit new-page work exceeds the read-path saving, and PRETOUCH was slower at both reported endpoints.
- "Raises the intercept, not the slope" means that the large historical slope-change claim from the short span was not reproduced on the wider span. It is not proof of a zero slope effect.

## Setup

30B Q4_K_M + store, budgets 4 and 1.5 GiB, with and without ARGINE_EXPERT_PRETOUCH; per-miss timings from the store's counters.

*Audited setup (2026-10-11):*
- **Model and budgets.** Qwen3-30B Q4_K_M with the expert store; reported endpoint budgets 4 GiB and 1.5 GiB; historical comparison with and without PRETOUCH.
- **PRETOUCH attribution** comes from the historical Argine v1 README, not from run flags.
- **Per-miss counters** (read, new pages) were printed to stderr historically and are not archived.
- **No experiment-local raw file.** There is no `raw/records.jsonl` for EXP-015.
- **Protocol era.** The experiment predates Argine2's fingerprinting/manifest protocol. Binary identity, run order and repetitions are not preserved for the PRETOUCH comparison and are not reconstructed here.

## Commands

```bat
argine benchmark MODEL --expert-store --sweep-expert-budget 4G,1.5G [--pretouch] --expert-stats
```

## Raw data

Speed records in `data/raw/argine-v1/runs.jsonl` (2026-10-02 23:21 → 2026-10-03 09:23), but the logs do NOT record whether PRETOUCH was on: the per-run attribution comes from Argine's README only. Per-miss timings were printed, not logged.

*Provenance (2026-10-11 audit):*
- The global Argine v1 raw run logs (`data/raw/argine-v1/runs.jsonl`) contain speed measurements from the relevant period.
- They do not encode PRETOUCH state, so they cannot independently map each run to PRETOUCH on or off.
- The per-miss timing counters were not logged.
- `results.csv` is a historical summary table sourced from the immutable Argine v1 README (`docs/history/ARGINE-V1-README.it.md`); its `source` column says so. It is **not** independently re-derived from raw run records.
- No `raw/records.jsonl` exists for EXP-015, and `tools/build_historical_data.py` does not provide an EXP-015 raw reconstruction. The audit creates no synthetic raw evidence.
- The Origin field above was corrected accordingly; it previously said "data re-derived from raw logs by Argine2".
- Context: EXP-018 later documents unrecorded historical configuration, including PRETOUCH and the read mode, as a reproducibility limitation. Its data are not used to supply PRETOUCH attribution here.

## Result

[MEASURED, Argine README] read per miss 1.233 vs 0.932 ms (4 GiB), recovered 73-81% of the explicit zeroing time; exponent at 2.67x span 0.54 +- 0.03 without vs 0.58 +- 0.06 with. *(Original wording; audited table below.)*

**Audited table (2026-10-11).** Every source value is [MEASURED, HISTORICAL ARGINE V1 README]; none is raw-log derived.

| quantity | 4 GiB without | 4 GiB with | 1.5 GiB without | 1.5 GiB with |
|---|---:|---:|---:|---:|
| read/miss | 1.233 ms | 0.932 ms | 1.063 ms | 0.773 ms |
| explicit new-pages/miss | — | 0.370 ms | — | 0.395 ms |
| generation | 6.14 tok/s | 5.90 tok/s | 3.61 tok/s | 3.32 tok/s |

Deterministic arithmetic on those values [DERIVED]:

| derived quantity | 4 GiB | 1.5 GiB |
|---|---:|---:|
| read-path reduction with PRETOUCH | 0.301 ms | 0.290 ms |
| reduction as a share of explicit new-page cost | ≈81% | ≈73% |
| explicit new-page cost relative to the read reduction | ≈23% higher | ≈36% higher |
| throughput contrast with PRETOUCH | ≈-3.9% | ≈-8.0% |

Speed/residency exponent fits [MEASURED, HISTORICAL ARGINE V1 README]:

| span | without PRETOUCH | with PRETOUCH |
|---|---:|---:|
| 2.67x (4 → 1.5 GiB) | 0.54 ± 0.03 | 0.58 ± 0.06 |
| 1.33x (earlier; inadequate: the tool itself declared the span insufficient for a verdict) | 0.70 ± 0.20 | 0.47 ± 0.15 |

The 1.33x-span values come from the historical README's prose and table; they are not in `results.csv`.

## Interpretation

*[Historical interpretation, narrowed by the 2026-10-11 audit: "Zeroing is displaced work (23-37% more expensive when done explicitly). The earlier 'PRETOUCH steepens the curve' was an artefact of a 1.33x span." The historical source also called the mechanism "confirmed". Both are superseded by the wording below; recomputed from the tabulated values the overhead is ≈23–36%.]*

[DERIVED, HISTORICAL / PROVENANCE-LIMITED] The historical counter decomposition is quantitatively consistent with PRETOUCH displacing a substantial fraction of page-initialization work from the recorded read path.
- The historical source first called the mechanism "plausible" and predicted that read-per-miss without PRETOUCH would be higher by about the new-page cost paid with it. The reported counters match that prediction approximately: approximately 73–81% of the explicit new-page cost corresponds numerically to reduced read-per-miss time.
- This supports the historical mechanism interpretation, but does not independently prove an OS-level causal decomposition, because the PRETOUCH flag and the per-miss counters are absent from the archived raw logs. Other unrecorded historical-state differences cannot be independently excluded.
- The picture is substantial displacement plus additional explicit-touch overhead. The work appears mostly displaced rather than purely added, with measurable extra overhead: the explicit new-page cost is ≈23–36% larger than the read-path time saved.

[SUPPORTED, HISTORICAL / PROVENANCE-LIMITED] **Audit closure (2026-10-11).**
- SUPPORTED is retained, with provenance-limited mechanism scope.
- The historical counter decomposition (read-path reduction 0.301 / 0.290 ms vs explicit new-page cost 0.370 / 0.395 ms) is quantitatively consistent with substantial displacement of page-initialization work from the recorded read path. Explicit PRETOUCH still carries additional overhead.
- PRETOUCH was slower at both reported historical endpoints (≈−3.9% at 4 GiB, ≈−8.0% at 1.5 GiB). These two values are not a universal PRETOUCH penalty.
- On the reported 2.67x-span fits, the exponent estimates were similar (0.54 ± 0.03 without vs 0.58 ± 0.06 with PRETOUCH), so the earlier large slope-change interpretation from the 1.33x span was not reproduced. No formal equality or equivalence of slope is established.
- The historical source interpreted the wider-span fit primarily as an intercept/level effect rather than a large slope change. Its '~4%' intercept statement is a historical fit interpretation, not an independently raw-reconstructable invariant.
- OS-level page-zeroing causality is not independently raw-proven.
- No rerun is required for historical closure ([historical audit](../../docs/research/HISTORICAL-AUDIT-EXP011-018.md)).

## Limitations

[LIMITATION] The PRETOUCH flag is absent from the raw records, so this experiment cannot be re-derived from raw data — a reason why Argine2 records runtime flags in every manifest.

[LIMITATION, historical audit 2026-10-11]
1. No experiment-local `raw/records.jsonl` exists.
2. PRETOUCH state is not encoded in the archived Argine v1 run records.
3. Per-miss store counters were printed but not logged.
4. `results.csv` is sourced from the historical README rather than independently regenerated from raw measurements.
5. `tools/build_historical_data.py` does not provide an EXP-015 raw reconstruction.
6. Historical run ordering and condition rotation are not established by any preserved immutable evidence.
7. No modern Argine2 binary/runtime fingerprint.
8. One model, one machine, a Windows historical environment.
9. The mechanism is supported by quantitative consistency, not by direct OS-level causal instrumentation.
10. 0.54 ± 0.03 vs 0.58 ± 0.06 is not an equivalence test.
11. "~4% intercept" is a historical fitted interpretation, not a universal constant.
12. The throughput penalties at the two reported endpoints differ (≈3.9% and ≈8.0%).
13. EXP-015 predates the Argine2 reproducibility protocol.

## Status history

2026-10-03 SUPPORTED (mechanism); raw linkage incomplete. 2026-10-11 historical audit: SUPPORTED retained with provenance-limited mechanism scope. Historical counters are consistent with substantial read-path work displacement; 'moved, not added' and 'no slope change' are narrowed from causal/absolute statements. Raw PRETOUCH attribution and per-miss counters are not preserved; no new measurement. The Origin field was corrected (previously "Argine v1 (historical); data re-derived from raw logs by Argine2").
