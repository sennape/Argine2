# Failed hypotheses

Disproven ideas are kept. Each entry: the original hypothesis, why it looked plausible, the experiment, the evidence, and the conclusion. Entries from Argine v1 cite its README ([`docs/history/ARGINE-V1-README.it.md`](docs/history/ARGINE-V1-README.it.md), section "Ipotesi smentite") and, where they exist, the raw records. A hypothesis that was later rehabilitated says so instead of being deleted.

Status vocabulary: FALSIFIED, PARTLY FALSIFIED, REHABILITATED, DOWNGRADED (moved to INCONCLUSIVE).

---

### FH-001 — Disk bandwidth is the primary bottleneck — FALSIFIED

- **Hypothesis.** Streaming experts from NVMe is limited by the device's bandwidth.
- **Why plausible.** Experts are gigabytes; generation read hundreds of MB per token.
- **Experiment.** [EXP-003](experiments/EXP-003-direct-io/README.md) (diskbench, store bench).
- **Evidence.** [MEASURED] Device on the expert access pattern: 4260-4470 MB/s; production with mmap: 562 MB/s (one thread was reading), 1139 MB/s after the fix, 2133 MB/s with async reads. Page faults 310 MB/s vs direct reads 1254-1324 MB/s at one reader. On the 80B the aggregate was 1082 MB/s with ~2500 available.
- **Conclusion.** The access path and the number of reads in flight limited throughput; on the 80B the limit moved to residency bookkeeping.

### FH-002 — Larger reads beat more numerous reads — FALSIFIED

- **Hypothesis.** Reading the three slices of an expert as one 2.75 MiB read is better than three 896 KiB reads.
- **Why plausible.** At queue depth 1, 2.75 MiB reads give 2351 MB/s vs 1338 MB/s.
- **Experiment.** EXP-003 (diskbench by size × readers).
- **Evidence.** [MEASURED] At fixed bytes per node, larger reads mean fewer reads in flight; three 896 KiB reads in flight reached ~3522 MB/s vs ~2351 MB/s for one 2.75 MiB read.
- **Conclusion.** The two levers are coupled; queue depth wins in the regime the store operates in.

### FH-003 — n-gram speculative decoding helps — FALSIFIED

- **Hypothesis.** Prompt-lookup / n-gram drafting speeds up generation on this machine.
- **Why plausible.** It costs almost nothing and helps on repetitive text.
- **Experiment.** Argine v1 benchmark with `--spec` (no dedicated raw record survives).
- **Evidence.** [MEASURED, Argine README only] No gain.
- **Conclusion.** Not pursued. Re-testing with logged configuration would be cheap.

### FH-004 — A larger compressed model knows more than a smaller intact one — FALSIFIED

- **Hypothesis.** The project's founding assumption (as PicoStar16).
- **Why plausible.** Parameter count correlates with capability at equal precision.
- **Experiment.** [EXP-012](experiments/EXP-012-dense-14b-control/README.md).
- **Evidence.** [MEASURED] Perplexity 13.28 ± 0.32 (dense 14B Q4_K_M) vs 13.63 ± 0.33 (30B-A3B, 2.56-bit experts); speed 3.83 vs 12.97 tok/s.
- **Conclusion.** On the matched-text perplexity proxy, the larger compressed 30B did not show the predicted advantage.
- **Scope (historical audit, 2026-10-11).**
  - FALSIFIED is retained for the operationalized historical claim that the larger compressed model would show a clear perplexity advantage on the tested corpus/signature.
  - EXP-012 does not demonstrate equal knowledge, equal assistant quality, equal reasoning capability or model equivalence.
  - The historical 3.83 vs 12.97 tok/s comparison is an observed approximately 3.39x throughput contrast, not a clean sparse-vs-dense architecture effect: the runs used different runtime binaries and different sessions.
  - The model footprints were similar, not equal: approximately 8.38 GiB vs 9.65 GiB (9.58 GiB per the 30B benchmark record).

### FH-005 — Resident percentage predicts scaling — FALSIFIED

- **Hypothesis.** Speed of an unseen MoE follows from its resident expert fraction (and bytes per token).
- **Why plausible.** It held on one model across budgets (EXP-005); the 80B prediction was written down first.
- **Experiment.** [EXP-006](experiments/EXP-006-80b-budget-sweep/README.md).
- **Evidence.** [MEASURED] Predicted 4.0-5.0 tok/s, measured 6.87 (and 7.10 in the sweep).
- **Conclusion.** A second, model-dependent variable is missing — Argine proposed router concentration (RQ-002).

### FH-006 — Asynchronous I/O improves every memory regime equally — FALSIFIED

- **Hypothesis.** Overlapped reads give +7-11% across budgets.
- **Why plausible.** The first measurement said so.
- **Experiment.** [EXP-004](experiments/EXP-004-async-store/README.md).
- **Evidence.** [MEASURED] On one binary: −0.6% at 4 GiB, +10.8% at 1.5 GiB. The first measurement compared three binaries.
- **Conclusion.** Benefit grows with miss pressure. The uniform claim is SUPERSEDED.

### FH-007 — More experts mean less locality and more misses — FALSIFIED

- **Hypothesis.** With 512 experts each expert is used more rarely, so a cache does worse than with 128.
- **Why plausible.** Intuition about per-item frequency.
- **Experiment.** EXP-006.
- **Evidence.** [MEASURED] At equal residency the 512-expert model had fewer misses per token.
- **Conclusion.** What matters is the size of the hot set relative to the cache, not per-expert frequency.

### FH-008 — Reordering the GGUF by co-activation speeds up streaming — FALSIFIED

- **Experiment.** [EXP-016](experiments/EXP-016-coactivation-reorder/README.md).
- **Why plausible.** Contiguous co-activated experts turn random reads into sequential runs.
- **Evidence.** [MEASURED] 4.69 vs 5.17 tok/s (reordered slower).
- **Conclusion.** *[Historical wording, narrowed by the 2026-10-11 audit: "Under page-fault streaming, layout was not the limit."]*
- **Audited conclusion (historical audit, 2026-10-11).**
  - The specific historical co-activation reorder did not improve throughput in the contemporaneous comparison (4.69 vs 5.17 tok/s).
  - The condition attribution comes from the historical README rather than independently encoded raw benchmark metadata. 4.69 is one repetition of a group whose median is 4.57.
  - This falsifies the tested historical improvement prediction, not the general usefulness of physical layout or co-activation information.
  - Later page-fault-vs-explicit-read results provide context but do not causally explain this result.

### FH-009 — The binary does not matter, only the algorithm — FALSIFIED

- **Experiment.** [EXP-011](experiments/EXP-011-binary-rebuild/README.md).
- **Evidence.** [MEASURED] 7.45 vs 12.97 tok/s between two builds of nearly the same source (1.74×).
- **Conclusion.** Binaries are experimental variables and are fingerprinted.
- **Scope (historical audit, 2026-10-11).**
  - The falsified operational claim is: "binary identity can be ignored when comparing performance; only the nominal algorithm/configuration matters".
  - The evidence is 7.45 vs 12.97 tok/s under the matched recorded model/runtime plan, so binary identity cannot be ignored and must be fingerprinted.
  - EXP-011 does **not** identify which build or source difference caused the gap: llama.cpp version, OpenMP and linkage are confounded.
  - "Nearly the same source" was not established.

### FH-010 — The IQ2 speed gain is +65% — FALSIFIED

- **Evidence.** [MEASURED, Argine README] At equal threads the gain was +19%; the rest came from the thread change.
- **Conclusion.** Two variables changed at once.

### FH-011 — The read queue is shallow because the threads are few — REHABILITATED

- **First verdict.** Falsified with "0.75 misses per node: misses are rare, not serialised".
- **Why the verdict was wrong.** 0.75 was measured at 49% residency. At the budgets that matter, misses per node are 2.30 (4 GiB) and 4.99 (1.5 GiB) and reads were serialised; async reads then gave up to +10.8%.
- **Conclusion.** A falsification valid in one regime was applied to another. The regime is a property of the measurement, not a constant of the program.

### FH-012 — Bigger fixed read blocks are better — FALSIFIED

- **Evidence.** [MEASURED] Block 4: same speed, thread waiting ×2.26, wall +8%. Block 2: best of the block variants, later indistinguishable from synchronous on one binary.
- **Conclusion.** Blocks couple batch size to load balance (45 slices / 12 threads); emit-and-continue decouples them.

### FH-013 — A lower speed/residency exponent means faster — FALSIFIED

- **Evidence.** [MEASURED, Argine] Block-2 had exponent 0.53 vs 0.58 and was slower at all budgets (slower at the top, not faster at the bottom).
- **Conclusion.** The exponent is a slope, not a level.

### FH-014 — PRETOUCH changes the slope of the curve — FALSIFIED

- **Experiment.** [EXP-015](experiments/EXP-015-pretouch/README.md).
- **Evidence.** [MEASURED, Argine README] Over a 2.67× span: 0.54 ± 0.03 without vs 0.58 ± 0.06 with; the earlier difference came from a 1.33× span the tool itself declared insufficient.
- **Conclusion.** *[Historical wording, narrowed by the 2026-10-11 audit: "It raises the intercept (~4%), not the slope. The mechanism (zeroing moved, not added) was confirmed."]*
- **Audited conclusion (historical audit, 2026-10-11).**
  - The failed historical claim is that PRETOUCH materially changes or steepens the speed/residency slope in the way inferred from the short-span measurement (0.70 ± 0.20 vs 0.47 ± 0.15 over an inadequate 1.33× span). The historical 2.67×-span fits (0.54 ± 0.03 without, 0.58 ± 0.06 with) did not reproduce that large slope difference.
  - The historical source interpreted the remaining difference mainly as a level/intercept effect. The "~4% intercept" is a historical fit interpretation, not a measured invariant.
  - This is not an equivalence test and does not show a zero slope effect.
  - The reported per-miss counters (read-path saving 0.301 / 0.290 ms vs explicit new-page cost 0.370 / 0.395 ms) are consistent with substantial read-path work displacement plus extra overhead, but only as provenance-limited consistency evidence: the PRETOUCH flag and those counters are not preserved in the raw logs, so the mechanism cannot be independently re-derived.
  - EXP-015 itself remains SUPPORTED.

### FH-015 — Cost per miss grows as the budget shrinks — FALSIFIED

- **Evidence.** [MEASURED, Argine] Read per miss 1.233 → 1.063 ms from 4 to 1.5 GiB; queue depth 2.09 → 2.49; per-read bandwidth +16%.
- **Conclusion.** The device returns part of the miss growth (+29% speed at 1.5 GiB). See RQ-016.

### FH-016 — Prefill dominates a long conversation; a disk KV cache would give >100× on TTFT — FALSIFIED

- **Experiment.** [EXP-014](experiments/EXP-014-conversation-prefix-reuse/README.md).
- **Evidence.** [MEASURED] With in-RAM prefix reuse TTFT stays ~5 s from 188 to 1467 tokens. *(Original wording; narrowed below to the observed 4.78–5.62 s in one benchmark.)*
- **Conclusion.** The comparison had been against no cache at all. Project cancelled.
- **Scope (historical audit, 2026-10-11).**
  - What failed is the historical premise: that cumulative prefill would dominate TTFT in the ordinary same-session growing conversation. In the single six-turn EXP-014 benchmark, with in-RAM prefix reuse, observed TTFT stayed within 4.78–5.62 s, against 5.11 → 40.12 s without reuse. "~5 s" is that observed range, not a constant.
  - The ">100×" figure was an engineering estimate against a no-cache baseline, not a measured disk-vs-RAM result. EXP-014 did not benchmark a disk-backed KV cache and does not show such a cache has zero value.
  - The cancellation was a project-prioritization decision for the ordinary conversation case. Restart, persistence and cross-conversation use were not tested.
  - EXP-014 itself remains SUPPORTED for its measured result.

### FH-017 — With the store, model size has no ceiling — PARTLY FALSIFIED

- **Evidence.** [MEASURED] 2.67× less residency costs 1.63-1.76× speed on the 30B.
- **Conclusion.** No memory ceiling, but a time ceiling.
- **Scope note (historical audit, 2026-10-11).** Read as: no sharp capacity ceiling *within the tested streaming regime* (30B and 80B, 1.5–4 GiB), but a practical time/performance ceiling. This is not a general claim about arbitrarily large models ([audit](docs/research/HISTORICAL-AUDIT-EXP001-006.md)).

### FH-018 — Commit/decommit cost is ~7% everywhere — FALSIFIED as a general statement

- **Evidence.** [MEASURED, Argine] 7% of thread time on the 30B; 39% of the I/O bucket on the 80B.
- **Conclusion.** A per-model measurement had been written as a property of the residency layer.

### FH-019 — A templated test text measures quantisation damage — FALSIFIED

- **Experiment.** [EXP-017](experiments/EXP-017-calibration-text/README.md).
- **Evidence.** [MEASURED] +2.06% on templated text vs +36.3% on real prose (+26.0% after recalibration).
- **Conclusion.** *[Historical shorthand, narrowed by the 2026-10-11 audit: "The instrument produced the verdict."]*
- **Audited conclusion (historical audit, 2026-10-11).**
  - The +2.06% measurement itself is not false: it is valid for its recorded synthetic signature (`ctx=512;chunk=18;testo=bf8011647e36fad6`).
  - The falsified operational claim is that this historical synthetic templated evaluation was sufficient to support the broad negligible-damage verdict.
  - On real prose the same old IQ2_XXS showed +36.3% versus Q4. After separate real-prose calibration, the recorded penalty was +26.0%.
  - The synthetic vs real comparison also changes the chunk count (18 vs 60), so text choice is not causally isolated, and there is no full calibration × evaluation factorial.
  - No universal claim is made that templated texts can never be useful. No equivalence test supports "within noise".

### FH-020 — Runs of the same configuration agree within 5% — FALSIFIED (Argine2)

- **Experiment.** [EXP-018](experiments/EXP-018-reproducibility-audit/README.md).
- **Evidence.** [MEASURED, LEGACY RECORDED-CONFIGURATION] The historical low-variance assumption failed operationally: within-invocation CV reached ≈6.43%, and legacy records matching the historical grouping key showed ≈15-16% min-max spreads (30B: ≈15.36%, ≈16.04%). A supplementary 80B 4 GiB raw comparison was ≈19% lower (7.10 -> 5.74), but incomplete runtime provenance prevents treating these values as pure same-configuration session noise.
  - *[Historical wording, narrowed by the 2026-10-11 audit: "15-19% between runs of identical plans on the same binary"]*
- **Conclusion.** Use complete runtime fingerprints and interleaved/rotated replication for modern performance claims. The historical ~15% figure is a conservative legacy guardrail, not a universal significance threshold.
  - *[Historical wording, narrowed by the 2026-10-11 audit: "Interleave inside one session; record runtime flags; treat cross-session differences under ~15% as unestablished."]*

### FH-021 — The runtime hit rate is the expert-level hit rate — FALSIFIED (Argine2)

- **Hypothesis.** Implicit in every use of the 91.2% figure.
- **Experiment.** [EXP-020](experiments/EXP-020-hit-rate-semantics/README.md).
- **Evidence.** [MEASURED, test model] Up/down nodes re-pin slices loaded by the gate node and always hit: 97.7% runtime-style vs 93.1% gate-slice (`expert_level_hit_rate`).
- **Conclusion.** Runtime miss rate ≈ expert miss rate / 3.
- **Follow-up (2026-10-12, EXP-020 modern audit).** The approximate conclusion above is kept as recorded. Later evidence confirms the mechanism:
  - EXP-021 found all misses in gate nodes, record for record, on real 30B runs;
  - EXP-007's 252 qualified runs (30B and 80B, seven workload families, DEV/EVAL, 4 / 2 / 1.5 GiB) give a gate-slice (`expert_level_hit_rate`) / runtime-style miss ratio of exactly 3.000 in every run.
  - Scope: the tested Argine2 expert store and instrumented binaries. The replay's expert-event hit rate is a different metric.

### FH-022 — One thread per P-core is optimal on this CPU — DOWNGRADED (Argine2)

- **Experiment.** [EXP-001](experiments/EXP-001-thread-sweep/README.md).
- **Evidence.** [MEASURED] True for the MoE on the OpenMP build; on the official build the dense 8B preferred 8-12 threads.
- **Conclusion.** Not falsified, not general: INCONCLUSIVE until re-measured on the current build (RQ-014).
- **Historical audit (2026-10-11).** EXP-001 is closed as INCONCLUSIVE. Re-measuring belongs to a new experiment and does not reopen EXP-001 ([audit](docs/research/HISTORICAL-AUDIT-EXP001-006.md)).
- **Follow-up (2026-10-11).** EXP-027 measured the current build. See FH-030: the optimum is model-dependent.

### FH-023 — A free list of reusable slabs removes commit and zeroing — IMPOSSIBLE as designed

- **Evidence.** Slice addresses are fixed by ggml tensors: slots are not interchangeable.
- **Conclusion.** Would need decoupling physical memory from virtual addresses (VirtualAlloc2 placeholders / MapViewOfFile3). Estimated +13% on the 30B, larger on the 80B. Not attempted.

### FH-024 — A well-formed trace is a complete trace — FALSIFIED (Argine2)

- **Hypothesis.** If the raw log has no truncated line and no malformed record, it holds every event of the run.
- **Why plausible.** The parser checks both, and the first Windows trace passed both.
- **Experiment.** [EXP-019](experiments/EXP-019-instrumentation-validation/README.md), Windows smoke run.
- **Evidence.** [MEASURED] The last request has 126 of 127 decode steps and one partial step: ~263 ms of records buffered after the last 250 ms flush were lost when llama-server was terminated.
- **Conclusion.** Completeness is checked against an independent count (llama.cpp's `predicted_n` per request); a flush sentinel request now follows every workload.
- **Follow-up (2026-10-12, EXP-019 fresh Windows validation).** The negative result stands. With the flush sentinel, the measured workload was complete: 127 / 127 decode steps per request, 0 incomplete workload nodes, 0 replay-relevant sequence gaps, and replay event-exact over the workload. The well-formed trace (0 malformed records, no truncated line) still lost 5 records and left 1 node incomplete in the flush-sentinel tail. Completeness is therefore asserted only for the workload, and only from independent counts (`predicted_n`, per-shard sequence numbers, node completeness), never from well-formedness.

### FH-025 — `sharded-lru-16` replay at expert granularity reproduces the runtime — FALSIFIED (Argine2)

- **Hypothesis.** The runtime uses 16 sharded LRUs, so replaying the trace through 16 sharded LRUs over whole experts reproduces its hit rate. This was the EXP-021 hypothesis H1 and the "runtime model" row of every earlier policy table.
- **Why plausible.**
  - Same shard count, same budget, same replacement rule.
  - On the one-shard test fixture it was within 0.5 pp.
- **Experiment.** [EXP-021](experiments/EXP-021-replay-fidelity/README.md), nine 30B runs; [write-up](docs/research/REPLAY-FIDELITY-VALIDATION.md).
- **Evidence.** [MEASURED / DERIVED]
  - It was +3.5…+4.8 pp above the runtime's figure. That came mostly from comparing different metrics: decode-only expert hits against gate-slice hits over prefill + decode.
  - At matched metric it is still +1.5…+2.1 pp: the runtime caches slices, not experts, is not event-atomic, and keys shards by slice address.
  - With 16 shards under 1 MiB it is off by −12 pp.
- **Conclusion.**
  - A runtime reproduction must transcribe the store's state machine (`store-replay`, 0 disagreements on v2 traces) and use the runtime's metric definition.
  - The expert-unit row is kept as a counterfactual named `sharded-lru-16-expert`.
  - Results recorded before 2026-10-03 that used it as the runtime model are [SUPERSEDED].

### FH-026 — The DP allocation is the optimal static per-layer split — FALSIFIED as implemented (Argine2)

- **Hypothesis.** `allocator.dp_optimal` returns the miss-minimising static split of the budget across layers.
- **Why plausible.**
  - It is a textbook knapsack DP over exact LRU stack-distance curves.
  - It agreed with brute force on tests with equal expert sizes.
- **Experiment.** EXP-021 audit of `allocator.py` and `elasticity.py`.
- **Evidence.** [MEASURED]
  - On the 30B at 4 GiB it gave 38,206 misses vs 36,473 for the uniform split.
  - The DP charged sizes on a 512 KiB grid (budget / 8192), so 2,654,208-byte experts cost 3 MiB (+18.5%).
  - The uniform split then needed 4.68 GB of a 4.29 GB budget, so it was excluded from the feasible set.
- **Conclusion.**
  - The function solved a different, mis-specified problem.
  - It is now exact: min-plus convolution per size class, exact combination. On the same curves it gives 34,091.
  - It is guarded by `dp_optimal_checked`, which refuses a result worse than any feasible baseline, and by tests with the real sizes and with brute force on unequal sizes.
  - Earlier "optimal static" and "uniform excess" numbers are [SUPERSEDED].

### FH-027 — Pins are node-scoped for every thread — FALSIFIED (Argine2)

- **Hypothesis.** While node N is served, the pinned slices are exactly those already requested in node N, because every thread drops its pins when it enters a node. This was the pin model of `store-replay` until 2026-10-04.
- **Why plausible.**
  - Every thread of a pool enters every node and unpins before the barrier.
  - The model was exact on the C++ driver and on two Linux end-to-end runs (0 of ~180k decisions different).
- **Experiment.** [EXP-021](experiments/EXP-021-replay-fidelity/README.md), first Windows format-v2 run.
- **Evidence.** [MEASURED]
  - 5.29% of decisions differed. The first divergence: the runtime passed over a slice pinned by thread 12 since the previous decode graph.
  - llama-server ran with `--threads 4 --threads-batch 12`, so prompt graphs run on threads {0..11} and decode graphs on {4, 12, 13, 14}. Threads outside the current pool keep their pins.
  - 78,394 of 144,689 evictions passed over such slots.
  - With per-thread pins, the same trace replays with 0 differences.
- **Conclusion.** Pins are per thread: node-scoped only when one pool computes every graph. The Linux validation could not see the difference, because it used one pool. The C++ driver now has a two-pool mode, and the test requires the single-pool model to fail there.

### FH-028 — `adaptation_steps = 16` means the policies converge equally fast — FALSIFIED (Argine2)

- **Hypothesis.** Implicit in reading `adaptation.csv`: many variants report 16 adaptation steps at 2 and 1.5 GiB, so they converge equally and quickly after transitions.
- **Why plausible.** The metric is a convergence time, and the values agree across variants.
- **Experiment.** EXP-007 30B transition trace, temporal analysis ([write-up](docs/research/TRANSITION-ADAPTIVE-GAIN.md) §7).
- **Evidence.** [DERIVED]
  - The search starts at k = window = 16, so 16 is the smallest value the metric can return.
  - The tolerance is 10% of the steady miss rate: 17–20 misses per step under pressure, which is wider than the transients (median first-window excess +2.1 / −4.1 misses per step).
  - 27 of 30 rows at 2 GiB and 28 of 30 at 1.5 GiB sit at the floor.
- **Conclusion.** The definition is kept (historical files stay reproducible) and documented as limited. Absolute transient metrics replace it: excess16/32/64 against the steady state, with a noise reference (`adaptation_v2.csv`).

### FH-029 — The adaptive controller wins by reacting to workload transitions — FALSIFIED for the current controller (Argine2)

- **Hypothesis.** On a workload that switches every 300 tokens, adaptive allocation beats the uniform split because it reallocates after each switch (EXP-023's hypothesis: adaptive allocation shortens the transient).
- **Why plausible.**
  - The gain appears on a trace called `transition`.
  - Segment-optimal splits differ by 5–15% of the budget.
- **Experiment.** EXP-007 30B transition DEV trace, one session, counterfactual ([write-up](docs/research/TRANSITION-ADAPTIVE-GAIN.md)).
- **Evidence.** [DERIVED]
  - The controller's own mean allocation, applied as a fixed split, gains 118–135% of the controller's gain.
  - The post-transition windows (16% of steps) hold 14–15% of the gain.
  - The boundary excess equals that of the frozen split.
  - The reaction takes 157–297 of each segment's 299 steps.
- **Conclusion.** The gain is a better general split (explanation A). Fast reallocation has an oracle value of 13–27% of the static potential on this trace, which remains [HYPOTHESIS] for a faster controller. One session, no between-session replication.
- **Update 2026-10-11 (replication; the conclusion is unchanged).** [DERIVED] The same temporal analysis on the 30B EVAL, 80B DEV and 80B EVAL transition traces (`experiments/EXP-007-router-characterization/analysis-v3/*transition*-temporal/summary.json`) shows the same pattern:
  - the controller's mean allocation, applied as a fixed split, gains ≥ 113% of the controller's gain in every trace and budget;
  - against instantly switched per-segment splits, the reaction-lag cost is negative in every case.

  On the 80B at 2 and 4 GiB the total gain is small (381–1,024 misses), so its post-transition shares are noisy. This replaces "one session, no between-session replication" by four sessions (two models, DEV and EVAL). It is still the same untuned controller.

### FH-030 — One thread per P-core is optimal on the current runtime, and E-cores slow generation — FALSIFIED (Argine2)

- **Hypothesis.** Pre-registered in EXP-027 before any run:
  - (P1) on the 30B, 4 threads is not slower than 6, 8 or 12 by more than between-launch noise;
  - (P2) 12 threads (P-cores plus E-cores) gives the lowest generation tok/s on both models.
- **Why plausible.**
  - ggml synchronises all threads at every graph node, so a slow E-core adds a participant to every barrier.
  - On the OpenMP build, the 30B ran 40-46% faster with 4 threads than with 8 (EXP-001).
- **Experiment.** [EXP-027](experiments/EXP-027-current-thread-sweep/README.md): current `llama-research-v2` binary, CPU backend, 4 GiB Expert Store, 3 independent rotated launches per model × thread count, 24/24 valid.
- **Evidence.** [MEASURED]
  - 30B: 12 threads 6.74 tok/s (CLEAR BEST), 8 threads 6.63, 4 threads 6.51, 6 threads 6.44.
  - 80B: 4 threads 6.91 (CLEAR BEST), 12 threads 6.77, 8 threads 6.64, 6 threads 6.62.
  - All CVs ≤ 2.31%.
- **Conclusion.**
  - The optimum is model-dependent on the same runtime. On the 30B the current build reverses the old ordering.
  - The barrier argument does not predict the current build's behaviour.
  - The claim is scoped to this machine, binary, CPU backend, 4 GiB and the benchmark workload. EXP-001 is unchanged.

### FH-031 — A DEV-selected online adaptive controller keeps an advantage over a DEV-fitted static layer split on EVAL — FALSIFIED (Argine2)

- **Hypothesis.** EXP-010's surviving sub-question: a controller configuration selected using DEV only keeps an advantage over static allocation when it is applied unchanged to EVAL.
- **Why plausible.** In-sample, adaptive-e64 beats per-layer uniform and global LRU in most constrained regimes (EXP-007 C3).
- **Experiment.** [EXP-010](experiments/EXP-010-adaptive-allocation/README.md), frozen protocol EXP010-P1a: a DEV-selected, frozen, EVAL-tested retrospective holdout, leakage B (the EVAL adaptive outcomes had already been seen in EXP-007). Exact replay on 28 EXP-007 traces at 4 / 2 / 1.5 GiB.
- **Evidence.** [DERIVED]
  - DEV selected e32 in both models.
  - On EVAL, adaptive / DEV-fitted workload-family-informed static: G = 1.003747 (30B) and 1.001476 (80B), with the adaptive controller winning 4 of 21 cells in each model.
  - Both models are STATIC_BETTER_OR_EQUAL, so the frozen directional rule gives FALSIFIED.
- **Conclusion.**
  - The online adaptation adds no demonstrated advantage over an appropriately fitted static layer split. The gain over global LRU and uniform is attributable to layer partitioning and layer-aware allocation.
  - The effects are small (+0.37% / +0.15% misses).
  - Layer-aware allocation itself is **not** falsified: the adaptive controller still beats per-layer uniform (G 0.962839 / 0.991859).
  - The claim is scoped to replay miss counts, this controller and its five candidate configurations, and the static comparator, which receives workload-family information.
