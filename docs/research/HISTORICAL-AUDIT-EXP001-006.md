# Historical audit of EXP-001 … EXP-006 (closure, 2026-10-11)

EXP-001 to EXP-006 come from Argine v1 (2026-09-27 → 2026-10-03). Their raw records were imported into Argine2 and re-derived (`python tools/build_historical_data.py`). This audit closes them against the current Argine2 evidence standard.

Ground rules of the audit:
- No experiment was rerun and no raw record or measurement was changed.
- Claims that were reasonable at the time but too broad are **narrowed** or marked **[SUPERSEDED]** in place. They are not rewritten.

**Historical closure is not the same as future follow-up.**
- A *closed* historical experiment keeps its registry status: what its own data can support.
- A related *open research question* (for example RQ-014) is pursued by a **new** experiment.
- An open question never moves a closed historical experiment back to RUNNING.

## Audit table

| EXP | registry status | audit verdict | evidence quality | main limitation | rerun required for closure? | future follow-up |
|---|---|---|---|---|---|---|
| [EXP-001](../../experiments/EXP-001-thread-sweep/README.md) thread sweep | INCONCLUSIVE | **closed as INCONCLUSIVE**: the optimal generation thread count is build- and/or model-dependent; the historical evidence does not establish "one thread per P-core" on the current runtime | moderate per sweep (2–3 reps), but confounded | model and build confounded (MoE on the OpenMP build, dense 8B on the official build); sequential order; never measured on the current research binary | no | small thread sweep on the current binary, both MoE models, interleaved (RQ-014) |
| [EXP-002](../../experiments/EXP-002-repack/README.md) repack buffer | SUPPORTED | **closed, SUPPORTED with historical/narrow scope**: on the AVX2-only i5-1340P with the tested build and model, repack increased memory and did not improve throughput | weak-to-moderate (3 reps per config, one model) | AAA-then-BBB order, not interleaved; within-run thermal drift larger than the first-repetition difference | no | none needed; re-test only if a different CPU, build or model becomes relevant |
| [EXP-003](../../experiments/EXP-003-direct-io/README.md) direct I/O vs mmap | SUPPORTED | **closed, SUPPORTED (strong historical result)**: on the reference machine, with the tested Windows/runtime configuration and access pattern, the mmap/page-fault path, not raw NVMe sequential bandwidth, was a major streaming bottleneck | strong: three diskbench sessions | specific to this NVMe, OS and access pattern; diskbench is a device benchmark, not the full runtime | no | none |
| [EXP-004](../../experiments/EXP-004-async-store/README.md) async store | SUPPORTED | **closed, SUPPORTED with limitation**: issue-and-continue reads help under strong miss pressure (+10.8% at 1.5 GiB) and give little or nothing at high residency (−0.6% at 4 GiB) | moderate (one binary, 3 reps per point) | fixed within-budget order sync → block-2 → async, **not rotated**; queue depth was a wait-derived proxy | no | rotated replication only if the finding becomes central to the final paper |
| [EXP-005](../../experiments/EXP-005-30b-budget-sweep/README.md) 30B budget sweep | SUPPORTED | **closed, SUPPORTED with narrowed scope**: (A) non-expert overhead constant across budgets (hard budget, strong); (B) throughput degradation sub-linear over 4 → 1.5 GiB; exponent values are descriptive of their sessions | A strong (20 store runs, ±0.001 GiB); B moderate | between-session spread up to 15% (EXP-018) exceeds the quoted exponent ±; two-endpoint slope | no | none for closure; the speed model belongs to EXP-025 |
| [EXP-006](../../experiments/EXP-006-80b-budget-sweep/README.md) 80B prediction | FALSIFIED | **closed, FALSIFIED (valid negative result)**: resident expert percentage alone is insufficient to predict throughput across the tested MoE models (predicted 4.0–5.0, measured 6.87 tok/s; 7.10 in the sweep) | moderate (prediction recorded before the run; 3 reps) | a later 4 GiB run in the same session gave 5.74 tok/s (−19%), which does not reach the predicted range; I/O decomposition from one run | no | causal explanation of the 30B/80B difference: EXP-026 (RQ-009) |

## Narrowed or superseded claims

- **EXP-001.** Argine v1's general rule "one thread per P-core is optimal (+40-46%)" was already downgraded on 2026-10-03 (FH-022). It is now formally closed as INCONCLUSIVE. The historical measurements stay valid as measurements of the binaries used at the time.
- **EXP-002.** "Repack is worse on this CPU" is scoped to the tested CPU, build and model. Repack is **not** claimed to be harmful in general or slower on every CPU, build or model.
- **EXP-003.** The claim is scoped to the reference machine, the tested Windows/runtime configuration and the tested access pattern. Documentation correction: the README quoted only the 2026-09-27 and 2026-10-03 single-reader values. The 2026-10-01 diskbench session (mmap 524.5 MB/s vs direct 1665.8 MB/s, 896 KiB, one reader) is in `results.csv` and is now cited. It replicates the direction, with a 3.2× ratio against 4.3× on 2026-09-27.
- **EXP-004.** Documentation correction: the experiment README ("modes alternated inside each budget") and the global README F-005 ("interleaved within each budget") contradicted the experiment's own limitation and `processed/notes.json` ("same order at every budget, not rotated"). Both now state the fixed within-budget order.
- **EXP-005.**
  - The hypothesis "with the store, model size has no memory ceiling" is narrowed. There may be no sharp capacity ceiling within the tested streaming regime, but there is a practical time/performance ceiling (FH-017, PARTLY FALSIFIED).
  - The exponents 0.50–0.58 (30B) are descriptive measurements of those sessions, not universal constants.
- **EXP-006.**
  - The falsification stands.
  - "Concentration inferred from the hit counter" is [SUPERSEDED] by EXP-007: the 80B router is measured and is generally more concentrated than the 30B's.
  - EXP-007 also shows that routing concentration is not a model-agnostic predictor of inter-layer allocation opportunity.
  - So the cross-model speed difference is only partially explained (RQ-009, EXP-026).
  - The derived "≈74% expert-level hit rate" is kept as the historical derivation. The measured 80B values are 61.4–78.1% at 4 GiB across EXP-007 workloads, on a different binary and prompts.

## Registry state

There were no status changes: EXP-001 INCONCLUSIVE, EXP-002–005 SUPPORTED, EXP-006 FALSIFIED. These already matched the intended verdicts. No new run is required for any of the six closures.
