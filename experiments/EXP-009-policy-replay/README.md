# EXP-009 — Counterfactual replay of cache and allocation policies

| field | value |
|---|---|
| ID | `EXP-009-policy-replay` |
| Status | **INCONCLUSIVE** |
| Closure | CLOSED 2026-10-11: the primary hypothesis has no pre-specified quantitative criterion (see Interpretation) |
| Period | 2026-10-03 → 2026-10-11 |
| Origin | Argine2 |
| Related questions | RQ-001, RQ-005, RQ-007, RQ-017 |
| Supersedes / superseded | pilot numbers v1-legacy SUPERSEDED by v2 semantics (EXP-021); `sharded-lru-16` renamed `sharded-lru-16-expert` (counterfactual, not a runtime model) |

## Research question

On identical routing traces, how do LRU, sharded LRU (Argine's policy at expert granularity), per-layer LRU, LFU, 2Q, Least-Stale, static hot sets, layer-aware static allocation and the Belady-style oracle compare?

## Hypothesis

[HYPOTHESIS] Layer-aware allocation closes a significant part of the gap between LRU and the offline oracle under a fixed budget.

## Setup

Event-atomic replay (`argine/research/replay.py`), expert granularity, semantics v2 (every recorded step served to the cache, decode steps counted, warm-up and flush sentinel excluded), categories kept apart: runtime reproduction (`store-replay`, validated in EXP-021), counterfactual, counterfactual IN-SAMPLE / UPPER BOUND, oracle; budgets 4 / 3 / 2 / 1.5 / 1 GiB / 768 / 512 MiB. PILOT on the legacy 30B trace, offline policies fitted in-sample.

## Commands

```bat
argine research analyze data/processed/legacy-30b-route-request1.jsonl.gz --budgets 4G,3G,2G,1.5G,1G,768M,512M --out data/processed/pilot-legacy-30b-v2
argine research analyze RUN00\trace.jsonl.gz --budgets 4G,2G,1.5G --slice-bytes-from RUN00 --runtime-runs RUNS --out experiments/EXP-007-router-characterization/analysis-v2/run-00-4G
```

## Raw data

`data/processed/pilot-legacy-30b-v2/` (current) and `data/processed/pilot-legacy-30b/` (v1-legacy, kept as history): `policy_comparison.csv` (all rows), `per_layer_misses.csv`, `heterogeneity.json`, `REPORT.md`, figures. `results.csv` here = pilot comparison.

## Result

### 28-cell replay (EXP-007 traces, 2026-10-11)

[DERIVED, exact deterministic replay, counterfactual: expert-unit, event-atomic, decode, run-00-4G trace of each cell]
- **Coverage.** All nine policy families named in the research question were replayed on all 28 EXP-007 cells (2 models × 7 workload families × DEV/EVAL) at 4 / 2 / 1.5 GiB:
  - global LRU; sharded LRU at expert granularity (`sharded-lru-16-expert`); per-layer LRU (uniform split); LFU; 2Q;
  - Least-Stale, both variants (`least-stale-lru`, `least-stale-mru`);
  - static hot sets (`static-frequency`, `static-hot-per-layer`); layer-aware static allocation;
  - the Belady-style reference.
- **Sources:** `../EXP-007-router-characterization/analysis-v3/<cell>/policy_comparison.csv` (column `gap_closed_vs_lru`) and `../EXP-007-router-characterization/analysis-v3/cross-cell-summary.csv`.

**Share of the LRU-to-Belady-style miss gap closed** (minimum to maximum over the 14 cells of each model):

| model | budget | per-layer LRU, uniform split (implementable) | layer-aware static (IN-SAMPLE, fitted on the same trace) |
|---|---|---|---|
| 30B | 1.5 GiB | 18–23% | 27–32% |
| 30B | 2 GiB | 9–13% | 17–22% |
| 30B | 4 GiB | −5 to +3% | 7–14% |
| 80B | 1.5 GiB | 8–11% | 12–16% |
| 80B | 2 GiB | 6–10% | 9–14% |
| 80B | 4 GiB | −4 to +5% | 3–9% |

[DERIVED] Other policies, same source:
- **2Q** is within −12 … +16% of LRU's gap closure everywhere. It matches or beats layer-aware policies only at some 80B high-memory points (EXP-007 C3).
- **LFU and static-frequency** are worse than LRU at almost every point (down to −190% of the gap on the 80B).
- **`sharded-lru-16-expert`** is within −5 … +4% of LRU.
- **`least-stale-lru` is identical to global LRU** (by construction), and **`least-stale-mru` is much worse** (EXP-007 C11 and RQ-017).

**Evidence types in this experiment:**
- **Measured runtime:** the `runtime (measured)` rows (30B coding DEV, run-00 analysis only) and EXP-007's `runtime-hit-rates.csv`.
- **Exact deterministic replay:** all counterfactual rows. They are comparable with each other, never with the measured runtime (EXP-021).
- **Runtime reproduction:** `store-replay` in run-00-4G; its fidelity is validated per campaign in EXP-021.
- **Fitted IN-SAMPLE upper bounds:** `static-frequency`, `static-hot-per-layer`, `layer-aware-static` and the optimal static split. They are not deployable competitors.
- **Future-informed reference:** the Belady-style row. It is a reference, not a proven bound for multi-key, mixed-size events.
- **Out-of-sample:** none. Agreement between DEV and EVAL is pattern replication, not DEV-trained → EVAL-tested generalisation.

### Earlier results (kept as recorded)

[SUPERSEDED, v1-legacy semantics + grid-rounded allocator] At 4 GiB: global LRU 81.7% expert hits, Belady-style 90.4% (LRU makes 1.9x the oracle's misses). At 1.5 GiB: LRU 50.5%, per-layer LRU uniform 54.9%, layer-aware static 57.2% (in-sample), Belady 73.0%. Uniform per-layer split makes 5-9% more misses than the in-sample optimal static split at 1.5-4 GiB.

[DERIVED, pilot, v2, in-sample where marked] At 4 GiB: global LRU 82.4%, Belady-style 93.5% (LRU makes 2.7x the oracle's misses). At 1.5 GiB: LRU 50.6%, per-layer LRU uniform 55.1%, layer-aware static 57.4% (IN-SAMPLE / UPPER BOUND), Belady 74.1%. Uniform makes 10.1 / 6.4 / 5.2 / 5.4% more misses than the exact in-sample optimal static split at 4 / 3 / 2 / 1.5 GiB.

[DERIVED, EXP-007 run-00-4G, 30B coding dev, 389 decode tokens, v2] Expert-event hit, decode: global LRU 75.7 / 53.3 / 44.1% at 4 / 2 / 1.5 GiB; per-layer uniform 75.7 / 56.1 / 49.1%; `sharded-lru-16-expert` 75.4 / 53.5 / 45.1%; Belady 89.9 / 75.5 / 68.8%; layer-aware static (IN-SAMPLE) 77.4 / 57.9 / 51.3%. Runtime reproduction (`store-replay`, per-pool pins) 73.8 / 51.7 / 43.1%, measured runtime 73.7 / 51.7 / 43.1% (mean of 3). `analysis-v3/run-00-4G/REPORT.md`; the analysis-v2 values 73.9 / 51.9 / 43.2% used the superseded single-pool pin model.

[DERIVED, pilot] Below one token's expert footprint (384 experts = 1.02 GiB) global LRU gets 0% hits at 1 GiB, 768 MiB and 512 MiB (cyclic layer sweep longer than the cache), while LFU keeps 29-42%, static frequency 31-45% (in-sample) and Belady 43-64% (v2: unchanged to within 1 pp). Strictly layer-partitioned policies are infeasible below 1.02 GiB.

[DERIVED, pilot] Least-Stale 'stale-first, LRU among stale' is identical to global LRU (proved by construction, verified in tests); the 'MRU among stale' variant is much worse (58% at 4 GiB v1-legacy; 53.9% v2 pilot; 46.9% on EXP-007 run-00 at 4 GiB).

## Interpretation

**Status: INCONCLUSIVE (methodological).**
- The primary hypothesis was: "Layer-aware allocation closes a **significant** part of the gap between LRU and the offline oracle under a fixed budget."
- "Significant" was never given a numerical threshold or closure criterion before the results were observed. Choosing one now would be post hoc.
- So the hypothesis can be neither supported nor falsified cleanly under the experiment's original definition.
- This is a methodological INCONCLUSIVE, not a lack of evidence: the replay evidence above is complete for the policies, cells and budgets in the question.

**Supported regime-specific findings** (recorded separately from the status):
- **[SUPPORTED]** Layer-aware allocation can recover a **non-trivial but minority** share of the LRU-to-Belady-style gap under memory pressure.
- **[SUPPORTED]** The effect is strongly model- and budget-dependent:
  - **30B at 1.5 GiB:** large enough to matter in practice. Per-layer LRU recovers 18–23% of the gap, the in-sample layer-aware static split 27–32%.
  - **4 GiB, especially on the 80B:** small. Per-layer LRU recovers −4 … +5%, layer-aware static 3–9%.
- **[SUPPORTED]** Even the in-sample fitted layer-aware static allocation, an upper bound on static layer splits on that trace, leaves at least roughly **two thirds** of the gap unresolved in the strongest case observed (30B, 1.5 GiB). Within the tested replay abstraction, the remaining headroom is substantially larger for expert retention/selection than for layer-budget reallocation (EXP-007).

**Not in EXP-009:**
- The adaptive controller and its ablations belong to **EXP-010**.
- So does the missing DEV-trained → EVAL-tested allocation test.
- Out-of-sample evaluation of a fitted static split (RQ-005) is also not part of this experiment's question ("on identical traces").

The pilot reading below, "the oracle gap dwarfs the layer-allocation effect", is kept as written; the 28-cell evidence is consistent with it.

[DERIVED] In this pilot the oracle gap dwarfs the layer-allocation effect: what is lost to LRU is mostly about WHICH experts are kept, not how memory is split across layers. This is a pilot reading, not a result.

## Limitations

[LIMITATION] Pilot: one prompt, 127 tokens, in-sample fitting, uniform expert sizes, stall modelled (not measured). EXP-007: one workload, one run analysed. [SUPERSEDED, 2026-10-11] The EXP-007 part: 28 cells are now replayed (one run-00-4G trace per cell; routing is identical across repetitions where checked).

[LIMITATION, 2026-10-11] All fitted static rows are in-sample, and no out-of-sample result exists. Only 4 / 2 / 1.5 GiB were replayed on the 28 cells; 3 GiB and sub-1 GiB were replayed on the pilot only, and are not needed for this closure. Counterfactual rows are expert-unit and event-atomic: compare them with each other and with `store-replay`'s counterpart `sharded-lru-16-expert`, never with the measured runtime number (EXP-021).

## Status history

2026-10-03 RUNNING (pilot done). 2026-10-03 pilot and EXP-007 run-00 regenerated with replay semantics v2 and the exact allocator (EXP-021); v1-legacy outputs kept, marked superseded. **2026-10-11 CLOSED, INCONCLUSIVE.** The 28-cell replay is complete for the policies in the question. The primary hypothesis's "significant" had no pre-specified quantitative criterion, so no post-hoc threshold is applied. Regime-specific findings are recorded as supported. No new replay or inference was needed.
