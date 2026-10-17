# EXP-020 — What the runtime's hit rate counts

| field | value |
|---|---|
| ID | `EXP-020-hit-rate-semantics` |
| Status | **SUPPORTED** |
| Period | 2026-10-03 |
| Origin | Argine2 |
| Related questions | RQ-010 |
| Supersedes / superseded | — |

> **Audit scope (2026-10-12).** Four evidence sources are kept apart:
> - (a) the original fixture mechanism evidence: the EXP-019 end-to-end tiny-MoE run, summarised in `results.csv` and `processed/hit_rate_semantics.json`;
> - (b) the 2026-10-03 30B Windows smoke run (`experiments/SMOKE-30B`), which is not publication-quality;
> - (c) EXP-021's record-for-record confirmation on real 30B runs;
> - (d) EXP-007's publication-quality cross-model confirmation (`experiments/EXP-007-router-characterization/runtime-hit-rates.csv`, 252 qualified runs).
>
> Final scope: **SUPPORTED, within the scope of the tested Argine2 expert store and its hit-rate semantics.** The fixture files are unchanged, and later evidence is cross-referenced rather than copied.

## Research question

Is Argine's reported cache hit rate the expert-level hit rate?

## Hypothesis

[HYPOTHESIS] No: for every layer, the gate, up and down MUL_MAT_ID nodes each request the slices of all three kinds (argine-experts.cpp loads the whole layer in the first node); the second and third requests are guaranteed hits, so the runtime counter over-states the hit rate.

## Setup

End-to-end fixture run of EXP-019; every slice access attributed to the node whose residency window contains it (`rawlog.node_attributed_outcomes`).

Later confirmation (cross-referenced):
- EXP-021 attributed the slice accesses of real 30B runs record for record.
- EXP-007 lists both workload-only rates from each run's `manifest.json` in `runtime-hit-rates.csv`, collected by `tools/exp007_summary.py` and not recomputed from traces.

## Commands

```bat
argine research iotimeline experiments/EXP-019-instrumentation-validation/raw/e2e-fixture
```

## Raw data

Same raw log as EXP-019.

Later evidence (not copied here): `experiments/EXP-007-router-characterization/runtime-hit-rates.csv`, which sits at the EXP-007 root, not in `analysis-v3/`. It has 256 rows, 252 of them qualified by the EXP-021 closure method.

## Result

[MEASURED, fixture] 71 misses, all in gate nodes; up and down nodes: 1023 accesses each, 0 misses. Runtime-style hit rate 97.7%; gate-node/gate-slice (`expert_level_hit_rate`) hit rate 93.1%.

[MEASURED, Windows smoke run, 30B Q4_K_M, 4 GiB, coding dev, not publication-quality] runtime-style hit rate 90.06% vs gate-slice (`expert_level_hit_rate`) 70.17% (all records); misses 9.94% vs 29.83%, ratio 3.00. Workload-only (warm-up excluded, from re-processing the raw log): 90.17% vs 70.51%.

*Later confirmation (2026-10-12 audit; cross-referenced, not copied):*

[SUPPORTED, EXP-021] Confirmed record-for-record on the nine 30B runs: all misses are in gate nodes; the runtime's `expert_level_hit_rate` is gate-node slice hits over prefill + decode, workload only (prefill gate-slice hit ≈ 9%). This is the metric any replay must reproduce (and the reason the old replay looked 3.5-4.8 pp too optimistic). *(Moved here from Limitations by the 2026-10-12 audit; wording unchanged.)*

[MEASURED, EXP-007, 252 qualified runs] Every qualified run's manifest records both workload-only rates.
- **Coverage:** 30B and 80B, seven workload families, DEV and EVAL, 4 / 2 / 1.5 GiB; 84 cells with 3 qualified runs each.
- **Miss ratio:** (1 − `expert_level_hit_rate`) / (1 − `runtime_style_hit_rate`) is **3.000 in every qualified run** (2.99999985–3.00000015, i.e. 3 within the 8 significant digits stored). The 4 runs that failed EXP-021 closure, which are not counted, also give 3.000.
- **Workload gate-slice hit rate** (`expert_level_hit_rate`), ranges of cell means:

| | 4 GiB | 2 GiB | 1.5 GiB |
|---|---|---|---|
| 30B | 66.5–79.8% | 44.5–57.7% | 35.8–48.5% |
| 80B | 61.4–78.1% | 46.6–67.4% | 40.7–62.1% |

- **Runtime-style hit rate** at 4 GiB: 87.1–93.3% across both models.
- **Repetition SD:** ≤ 0.13 pp (maximum 0.125 pp).
- **80B coding DEV at 4 GiB:** 65.97 / 65.91 / 65.92% (≈ 65.9%).

## Interpretation

[SUPPORTED — tested Argine2 expert-store / hit-rate-semantics scope] The runtime's own counter does not measure first-request hits.

**Mechanism.** In the tested Argine2 expert store, each layer's gate, up and down MUL_MAT_ID nodes each request the slices of all three kinds. The gate node pays every miss, and the up/down requests re-pin resident slices and are guaranteed hits. The miss count is shared while the denominator triples, so

`runtime_style_miss_rate = gate_slice_miss_rate / 3`, equivalently `gate_slice_hit_rate = 1 − 3 × (1 − runtime_style_hit_rate)`.

**Exact, but implementation-specific.** The ratio is exactly 3.000 on the fixture, in the 30B smoke run and in all 252 qualified EXP-007 runs. It is a property of this store implementation, the instrumented binaries and the tested models, workloads and budgets. It is not a universal property of MoE runtimes, other expert stores or future binaries, and it does not hold where misses can occur outside the first node. It must be re-checked (`misses_outside_gate_nodes = 0`) whenever any of those change.

**Three metrics, not interchangeable:**
- **Runtime-style slice hit rate** (`runtime_style_hit_rate`): hits over all slice accesses in the gate, up and down nodes.
- **Workload gate-slice hit rate** (stored as `expert_level_hit_rate`; the field name is kept for compatibility): slice hits over the slice accesses made in the gate (first) node of each layer, prefill + decode, workload only (warm-up and flush sentinel excluded). `rawlog.node_attributed_outcomes` computes it as `gate_node_hit_rate`, and `runner.py` assigns it to the field.
- **Expert-event hit rate** (`expert_event` in EXP-021 and the replay analyses): an expert counts as a hit only if all three of its slices hit in the gate node. It is usually stated per phase (decode). For example, on the 30B coding DEV runs at 4 GiB: gate-slice 70.6% over all phases, gate-slice 75.4% on decode, expert-event 73.7% on decode (EXP-021, RQ-010). Phase and definition move the value by up to 4.7 pp.

All EXP-020 values (fixture 93.1%, smoke 70.17%, the EXP-007 ranges) are gate-slice values.

**The historical ≈74% for the 80B.**
- It was derived from Argine v1's 91.2% runtime-style hit rate, on another binary and prompt; it was never a measurement.
- As a number, it is superseded by the direct EXP-007 measurements: the 80B gate-slice hit rate at 4 GiB spans 61.4–78.1% across workloads, and the coding DEV value is ≈ 65.9%.
- EXP-006's qualitative conclusion survives: with 11.7% of the experts resident, the 80B's hit rate is far above its resident fraction.
- EXP-007 now measures routing and cache behaviour directly.

[STATUS] **Audit closure (2026-10-12).** CLOSED / SUPPORTED within the scope above. No new measurement was performed. Independent review completed on 2026-10-12; the modern audit is CLOSED / SUPPORTED.

*[Historical interpretation (2026-10-03), kept as recorded; its ≈74% is superseded as a numeric estimate: "[DERIVED] runtime_miss_rate ≈ expert_miss_rate / 3. Applied to Argine's 80B figure (91.2% runtime hits): expert-level ≈ 74%. The qualitative conclusion of EXP-006 (an 11.7% cache serving far more than 11.7% of requests) survives; the number does not."]*

## Limitations

*[Historical limitation (2026-10-03), resolved: "[LIMITATION] Mechanism measured on the fixture; the factor 3 assumes misses only in the first node of each layer (true on the fixture, to be verified on real models in EXP-007 runs, whose summaries report both rates)." Verified on real models by EXP-021 (30B, record for record) and EXP-007 (252 qualified 30B/80B runs, ratio 3.000).]*

1. [LIMITATION, audit 2026-10-12] Implementation scope. The factor 3 holds for the tested Argine2 expert store (the hook loads all three slice kinds in the gate node, and up/down re-pin them) and the instrumented binaries. A change to the hook, the slice layout, the binary, or any path where a slice could be evicted or first requested outside the gate node needs a fresh check of `misses_outside_gate_nodes`.
2. [LIMITATION, audit 2026-10-12] Evidence scope. Two models (Qwen3-30B-A3B Q4_K_M, Qwen3-Next-80B-A3B), seven workload families, DEV/EVAL, budgets 4 / 2 / 1.5 GiB, one machine. Budgets below 1.5 GiB (EXP-022), other models and other binaries are untested.
3. [LIMITATION, audit 2026-10-12] Precision. The hit rates are stored to 8 significant digits; the ratio equals 3 within that precision.
4. [LIMITATION, audit 2026-10-12] Terminology. The field name `expert_level_hit_rate` is historical and means workload gate-slice. Any cross-experiment comparison must state the metric and the phase, and must not mix it with `expert_event`.
5. [LIMITATION, audit 2026-10-12] Smoke-run provenance. The smoke run's workload-only values (90.17 / 70.51%) came from re-processing the raw log and were not re-verified by this audit. Its all-records values (0.90057833 / 0.70174034) are in `experiments/SMOKE-30B/run-00-4G/summary.json`.
6. [LIMITATION, audit 2026-10-12] `results.csv` and `processed/hit_rate_semantics.json` describe the fixture only. The real-model evidence lives in EXP-021 and EXP-007.
7. [LIMITATION, audit 2026-10-12] The docstring of `rawlog.node_attributed_outcomes` says "expert-level hit rate" for what is the gate-node slice rate. It was not edited (documentary audit, no code change).

## Status history

2026-10-03 SUPPORTED (mechanism, fixture). 2026-10-03 (later) factor 3.00 observed on the 30B in a Windows smoke run; publication-quality confirmation pending *(superseded 2026-10-12: completed by EXP-021 and EXP-007, see below)*.

2026-10-12 modern audit: **CLOSED / SUPPORTED** — tested Argine2 expert-store / hit-rate-semantics scope. The publication-quality confirmation is complete: EXP-021 confirms the gate-node semantics record-for-record on real 30B runs, and EXP-007 gives a miss ratio of 3.000 in all 252 qualified 30B/80B runs. `expert_level_hit_rate` is defined as the workload-only gate-slice rate; `expert_event` remains a distinct metric. The ≈74% 80B figure is superseded as an estimate. No new measurement. Independent review completed; modern audit CLOSED / SUPPORTED.
