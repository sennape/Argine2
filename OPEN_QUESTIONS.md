# Open research questions

Each question has a permanent ID. Work against these, not against features. An entry states what is known, what is unknown, the evidence so far, what blocks progress, and the contribution that would move it. Status: OPEN, PARTIALLY ANSWERED, ANSWERED, ABANDONED (with reason). Answered questions stay here.

Index: [RQ-001](#rq-001--is-expert-residency-better-modelled-as-online-resource-allocation-than-as-cache-replacement) · [RQ-002](#rq-002--do-routing-concentration-and-effective-working-set-predict-behaviour-better-than-size-or-residency) · [RQ-003](#rq-003--are-expert-working-sets-heterogeneous-between-layers) · [RQ-004](#rq-004--are-working-sets-workload-dependent) · [RQ-005](#rq-005--is-uniform-layer-allocation-suboptimal-and-by-how-much) · [RQ-006](#rq-006--can-the-marginal-value-of-memory-be-estimated-online) · [RQ-007](#rq-007--how-close-can-an-online-policy-get-to-the-offline-oracle) · [RQ-008](#rq-008--where-does-expert-streaming-stop-being-useful) · [RQ-009](#rq-009--why-does-the-80b-behave-differently-from-the-30b) · [RQ-010](#rq-010--what-is-the-true-expert-level-hit-rate-of-the-runtime) · [RQ-011](#rq-011--does-replay-reproduce-the-runtime) · [RQ-012](#rq-012--what-does-the-instrumentation-cost) · [RQ-013](#rq-013--can-an-interpretable-model-predict-mstoken-within-10-15) · [RQ-014](#rq-014--how-many-threads-on-the-current-build) · [RQ-015](#rq-015--what-causes-15-19-differences-between-sessions) · [RQ-016](#rq-016--is-hit-rate-the-right-objective) · [RQ-017](#rq-017--what-exactly-is-least-stale-and-does-it-help-here) · [RQ-018](#rq-018--do-slice-level-residency-decisions-matter)

---

## RQ-001 — Is expert residency better modelled as online resource allocation than as cache replacement?

**Status: OPEN** — the central question.

**Question.** With a fixed global expert budget `M = M_1 + ... + M_L`, does allocating memory across layers (and experts) by observed marginal value outperform static residency and conventional cache policies, without changing the model?

**Motivation.** Argine's runtime uses one hash-sharded LRU pool; most published systems use LRU-like replacement or fixed per-layer slots. If layers differ in the value of memory, both waste it.

**Known.** [DERIVED, pilot, in-sample, 127 tokens] On the legacy 30B trace, the uniform per-layer split makes 5-9% more misses than the best static split at 1.5-4 GiB, while LRU makes ~1.9× the misses of the Belady-style oracle at 4 GiB ([EXP-009](experiments/EXP-009-policy-replay/README.md)). (Pilot figures; the 5-9% was superseded by the exact allocator, see RQ-005.) [2026-10-11] EXP-009 is closed as INCONCLUSIVE: its "significant part of the gap" had no pre-specified criterion. Its 28-cell replay shows that layer-aware allocation recovers a non-trivial but minority share of the LRU-to-Belady-style gap (30B 1.5 GiB: 18–23% per-layer LRU, 27–32% in-sample layer-aware static; 80B 4 GiB: −4 … +9%).

**Known (2026-10-11, [EXP-007](experiments/EXP-007-router-characterization/README.md), 28 cells: 2 models × 7 workload families × DEV/EVAL, counterfactual replay, in-sample).**
- [SUPPORTED] Layer-aware adaptive allocation (`adaptive-e64`) beats global LRU in most constrained regimes, and the gain grows as the budget shrinks:
  - 30B: +0.8 … +1.5 pp at 4 GiB, +5.7 … +8.5 pp at 1.5 GiB;
  - 80B: −0.3 … +0.9 pp at 4 GiB, +1.4 … +3.2 pp at 1.5 GiB.

  It is not universally best: 2Q matches or beats it at six 80B points, mostly at 4 GiB.
- [DERIVED] Under pressure most of that gain comes from the static uniform per-layer partition itself. The adaptive split adds +1.4 … +1.8 pp (30B) and +0.4 … +0.7 pp (80B) at 1.5 GiB.
- [SUPPORTED] The gap to the future-informed Belady-style reference (+8.5 … +20.8 pp over adaptive-e64) is several times larger than any re-split gain. Within the tested replay abstraction, the remaining headroom is substantially larger for expert retention/selection than for layer-budget reallocation.
- [OPEN] Whether an online prediction-aware policy can realise a material fraction of that headroom.
- [FALSIFIED for the current controller] Its gain does not come from reacting to transitions (FH-029).

**Unknown.**
- A retrospective DEV-trained → EVAL-tested allocation comparison now exists in [EXP-010](experiments/EXP-010-adaptive-allocation/README.md) (leakage B). An untouched holdout remains open.
- Behaviour in the runtime (the controller exists in replay only).
- Whether prediction-aware retention can close part of the oracle gap.
- ~~Any real-model trace longer than one prompt; the 80B.~~ [SUPERSEDED 2026-10-11: EXP-007 provides traces of both models up to 4.9k decode tokens.]

**Related experiments.** EXP-007, EXP-009, EXP-010, EXP-022, EXP-023.

**Blockers.** ~~Phase-1 traces on the reference machine.~~ [SUPERSEDED 2026-10-11: EXP-007 closed.] ~~A DEV-selected → frozen → EVAL-tested allocation evaluation.~~ [Done 2026-10-11: EXP-010, a retrospective holdout, FALSIFIED for the adaptive controller over the DEV-fitted static split.] Current, none of them urgent:
- an untouched confirmation (new EVAL2 prompts and traces), if stronger generalisation evidence is wanted;
- runtime implementation and validation of a controller above the expert store;
- prediction-aware retention, as a separate open direction.

**Suggested contribution.** [2026-10-11, after EXP-010] EXP-010 completed the retrospective DEV → EVAL allocation comparison: family-matched DEV-fitted static splits and a DEV-selected controller, applied unchanged to EVAL. The earlier suggestion to run `argine research analyze EVAL --train DEV` is superseded by it.
- **Stronger confirmation:** genuinely new EVAL2 prompts and traces, written and frozen after the selection.
- **Separate future step:** a runtime implementation.
- **Separate open direction:** prediction-aware retention, aimed at the oracle gap.

## RQ-002 — Do routing concentration and effective working set predict behaviour better than size or residency?

**Status: PARTIALLY ANSWERED** (2026-10-11). Concentration and EWS are characterized and the formal EXP-008 predictor comparison is complete. Expert-level hit rate is supported within the traced EXP-007 dataset; throughput remains inconclusive. A clean runtime-throughput comparison remains open.

**Motivation.** [FALSIFIED] residency-only prediction of the 80B ([EXP-006](experiments/EXP-006-80b-budget-sweep/README.md)): at ~12% residency the 80B was 1.58× faster than the 30B.

**Known.** Argine's explanation (router concentration) is consistent with the hit counter, which over-states the expert-level hit rate by a factor ~3 on misses ([EXP-020](experiments/EXP-020-hit-rate-semantics/README.md)).

**Known (2026-10-11, [EXP-007](experiments/EXP-007-router-characterization/README.md)).**
- [DERIVED] Global EWS90 fraction: 0.327–0.554 on the 30B and 0.225–0.472 on the 80B, over 7 workload families × DEV/EVAL. Per-layer EWS, Denning working sets (`ws_tau16`), reuse and co-activation are in `analysis-v3/<cell>/`.
- [SUPPORTED] Routing concentration is not a model-agnostic predictor of exploitable inter-layer allocation opportunity:
  - in multi-turn the 80B is more concentrated (EWS90 0.267 / 0.226 vs 0.369 / 0.327);
  - yet its in-sample uniform excess is smaller (0.49–3.85% vs 4.10–7.40%).
- [DERIVED, exploratory, post hoc] **Within a model** there is an association: more concentrated workloads show a larger uniform excess at 4 GiB (ρ = −0.767 on the 30B, −0.934 on the 80B; n = 14 cells per model, unequal trace lengths, no formal predictive test).
- **Across models** it does not transfer: pooled, the sign reverses (`concentration-vs-opportunity.csv`). The formal predictor comparison remains EXP-008.

**Unknown.**
- ~~Whether `budget / EWS95_bytes` orders configurations by *speed* better than `budget / expert_bytes`: the formal predictor comparison of EXP-008, not run.~~ [2026-10-11, [EXP-008](experiments/EXP-008-working-set/README.md), INCONCLUSIVE] Run on the traced EXP-007 data with a plan frozen beforehand.
  - **Throughput:** the models disagree. On the 80B, budget / Denning working set beats resident fraction on all three measures; on the 30B, resident fraction is better on all three.
  - **Expert-level hit rate:** the working-set predictors beat resident fraction in both models on every measure (SUPPORTED within the traced dataset).
  - **Open:** a clean throughput comparison (untraced, matched context, workloads interleaved within sessions).
    - [2026-10-12] EXP-024 is now complete. Under its frozen rule, the 95% CI of the tracing overhead is entirely below 2% at 4G, while the 1.5G threshold question remains statistically inconclusive.
    - A genuinely clean predictor-throughput comparison would still use matched, untraced, interleaved measurements. That follow-up has not been done.
    - *(Earlier wording: "…; EXP-024 first".)*
- Length-normalised EWS (single-turn cells have only 320–389 decode tokens).
- ~~EWS90/95/99 and Denning working sets of either model on real workloads.~~ [SUPERSEDED 2026-10-11 by EXP-007.]

**Related experiments.** EXP-007, EXP-008, EXP-026.

**Suggested contribution.** [2026-10-11] The formal predictor comparison has already been done on the traced EXP-007 dataset ([EXP-008](experiments/EXP-008-working-set/README.md), analysis plan frozen before the statistics were computed).
- **Hit rate:** the working-set predictors beat resident fraction in both models on every measure. Supported within the traced dataset.
- **Throughput:** still INCONCLUSIVE, for two reasons. The 30B and the 80B disagree. And the traced data are confounded by:
  - tracing overhead: unknown when EXP-008 ran. EXP-024 has since measured it for the 30B benchmark configuration, with the 95% CI entirely below 2% at 4G and 1.5G inconclusive; it was not measured on the EXP-007 workloads themselves;
  - different context lengths per workload;
  - workloads recorded in different sessions.
- **What a clean throughput test would need:**
  1. ~~EXP-024 (tracing overhead) first~~: done 2026-10-12, INCONCLUSIVE overall (95% CI entirely below 2% at 4G, inconclusive at 1.5G);
  2. then untraced throughput runs, on workloads matched for context length;
  3. with the workloads interleaved within each session.

  This is a possible future contribution, not an immediate next step.

## RQ-003 — Are expert working sets heterogeneous between layers?

**Status: ANSWERED for routing working sets; PARTIALLY ANSWERED for memory value** (2026-10-11). The scope follows the convention of RQ-010.

**Known.** [MEASURED, pilot] On 127 tokens of one prompt (30B): EWS95 from 15% (layer 47) to 52% (layer 0) of 128 experts, normalised entropy 0.58-0.81, overlap with the previous token 0.34-0.66 ([EXP-007](experiments/EXP-007-router-characterization/README.md)).

**Known (2026-10-11, [EXP-007](experiments/EXP-007-router-characterization/README.md), 28 cells).**
- [SUPPORTED] Routing working sets are heterogeneous between layers in every cell of both models.
  - Per-layer EWS90 fraction: 0.20–0.71 on the 30B and 0.06–0.71 on the 80B.
  - Across-layer CV: 0.09–0.25 on the 30B and 0.16–0.46 on the 80B.
  - The spread survives the long multi-turn traces (3.1–4.9k decode tokens; CV 0.21–0.25 on the 30B, 0.36–0.46 on the 80B).
  - The least concentrated layer is layer 0 (30B), or layer 0 or 2 (80B), in every cell.
  - The pilot's "last layer most concentrated" does not generalise.
- [SUPPORTED] **Memory-value heterogeneity is model- and workload-dependent and not large everywhere.** The in-sample uniform-split excess is 3.5–10.4% on the 30B and 0.5–3.9% on the 80B. The 80B has the larger routing spread but the smaller memory-value spread.

**Unknown.**
- Bootstrap confidence intervals over tokens for the per-layer metrics (not implemented).
- Untouched-holdout confirmation of memory-value heterogeneity (RQ-005). EXP-010 already provides a retrospective DEV → EVAL result.
- ~~Whether the spread survives longer traces, other prompts, and the 80B.~~ [SUPERSEDED 2026-10-11 by EXP-007.]

**Suggested contribution.** Bootstrap confidence intervals over tokens; this is still a good first code contribution.

## RQ-004 — Are working sets workload-dependent?

**Status: PARTIALLY ANSWERED** (2026-10-11; updated 2026-10-12 with EXP-023). Workload dependence is positively evidenced, descriptively; controller exploitation of transitions remains unsuccessful for `adaptive-e64`.

**Known.**
- [SUPPORTED, [EXP-007](experiments/EXP-007-router-characterization/README.md)] Working sets depend on the workload: the global EWS90 fraction ranges 0.33–0.55 across workloads on the 30B and 0.23–0.47 on the 80B, with the DEV/EVAL pattern replicated.
- [DERIVED, transition traces] Within one session per-segment routing differs strongly: on the 30B DEV trace EWS90 is 0.27–0.43 per segment vs 0.55 over the session, and hot-set stability falls from 0.82 to 0.54 ([TRANSITION-ADAPTIVE-GAIN](docs/research/TRANSITION-ADAPTIVE-GAIN.md)). The same temporal analysis exists for 30B EVAL and 80B DEV/EVAL (`analysis-v3/*transition*-temporal/`).
- **Distinct from that:** [FALSIFIED for the current controller] the adaptive controller does *not* exploit transitions by reacting quickly (FH-029). Its reaction takes most of a segment, and a fixed split does as well or better. Workload dependence of the working set is real; fast exploitation of it by this controller is not.
- [DERIVED, retrospective operationalisation, [EXP-023](experiments/EXP-023-workload-drift/README.md), 2026-10-12] Per-layer routing distributions change strongly at workload boundaries on all four transition traces: the median boundary JSD is 2.84–4.02× the within-segment reference per trace, and exceeds the within-segment maximum in 96.5–100% of layer-boundary rows. Descriptive, not confirmatory (specified after the data existed; no p-value, no threshold).
- [DERIVED, EXP-023, 2026-10-12] Cross-trace transient synthesis (boundary transient / signed departure from the new-segment steady state): `adaptive-e64` has a lower signed `excess64` than the frozen split in 11 / 36 trace × budget × boundary comparisons (a lower signed value is not by itself a shorter transient; the temporal-reaction falsification rests on FH-029). FH-029's facts reproduce in 12 / 12 rows. EXP-023 is FALSIFIED for the current controller; this is not a falsification of workload dependence.

**Unknown.**
- ~~Per-layer Jensen-Shannon divergence between workload segments vs within-segment variation, as a reported metric.~~ [Done 2026-10-12, EXP-023, retrospective and descriptive.] A confirmatory test on untouched traces remains open.
- ~~A cross-trace summary of the absolute transient metrics (FH-028). `adaptation_v2.csv` exists for all four transition traces (`analysis-v3/*transition*-temporal/`) but is only written up for the 30B DEV trace.~~ [Done 2026-10-12, EXP-023.]
- Whether a faster controller can capture the oracle value of instant switching (13–27% of the static potential on the 30B DEV trace).

**Related experiments.** EXP-023 (`workloads/v1/transition.json`).

**Suggested contribution.** ~~Transition sessions now exist for both models, DEV and EVAL (EXP-007). Report per-layer segment divergence from their `drift.csv`, and the absolute transient metrics from `adaptation_v2.csv`, across all four traces.~~ [Done 2026-10-12, EXP-023.] Next: a faster or differently designed controller that exploits the measured boundary drift, or a confirmatory drift test on new, untouched transition traces.

## RQ-005 — Is uniform layer allocation suboptimal, and by how much?

**Status: PARTIALLY ANSWERED** (2026-10-11; in-sample characterization plus retrospective DEV → EVAL evidence; untouched holdout still open).

**Known.** [SUPERSEDED, EXP-021] 5-9% excess misses for uniform vs optimal static split (1.5-4 GiB). That pilot figure came from a grid-rounded allocator and v1-legacy replay. [DERIVED, pilot, in-sample, exact allocator, v2] uniform makes 10.1 / 6.4 / 5.2 / 5.4% more misses than the optimal static split at 4 / 3 / 2 / 1.5 GiB. [DERIVED, 30B coding dev run-00, 389 decode tokens, in-sample] 7.1 / 4.2 / 4.5% at 4 / 2 / 1.5 GiB. The earlier "optimal is worse than uniform at 4 GiB" was the allocator bug, not a property of the trace. [DERIVED] Strictly layer-partitioned designs cannot run below one token's expert footprint: 1.02 GiB on the 30B, ~0.67 GiB on the 80B.

**Known (2026-10-11).**
- [SUPPORTED, [EXP-007](experiments/EXP-007-router-characterization/README.md), IN-SAMPLE upper bounds] Uniform is suboptimal in every cell, by a model-dependent amount:
  - 30B: 3.5–10.4% excess misses (median 4.5%);
  - 80B: 0.5–3.9% (median 1.8%).
- These are in-sample optimal static splits on the same trace, so upper bounds on what a static split can gain, not deployable results.

**Unknown.** [EXP-010](experiments/EXP-010-adaptive-allocation/README.md) provides a retrospective, family-matched DEV-fitted → EVAL-tested static comparison. Relative to per-layer uniform, the static split has about 4.1% fewer misses on the 30B and 1.0% fewer on the 80B (geometric mean over 21 cells per model; the split uses workload-family information). Because the EVAL traces had already been inspected during EXP-007, an untouched holdout remains open. ~~The 80B.~~ [SUPERSEDED 2026-10-11 by EXP-007.]

**Suggested contribution.** The DEV → EVAL static comparison is done (EXP-010). If stronger confirmation is wanted: a genuinely untouched EVAL2 prompt set and traces, frozen after selection.

## RQ-006 — Can the marginal value of memory be estimated online?

**Status: PARTIALLY ANSWERED** (2026-10-11).

**Known.** The replay controller uses exact online LRU stack-distance histograms per layer (utility monitors) with decay. It reallocates on the pilot but the pilot is too short to judge.

**Known (2026-10-11, [EXP-007](experiments/EXP-007-router-characterization/README.md), replay).**
- [SUPPORTED] Online estimation finds useful allocations: adaptive-e64 vs its frozen uniform split (`frozen-e64`) ranges from −0.09 to +1.81 pp, so the gain is regime-dependent (RQ-001).
- [SUPPORTED] The epoch size (e32 / e64 / e128) changes the hit rate by ≤ 0.74 pp but the move count by 2.5–11.2×.
- [SUPPORTED] Hysteresis changes hits by ≤ 0.07 pp and moves by up to 2.9×.
- [SUPPORTED] Cost weighting changes hits by ≤ 0.05 pp.
- [FALSIFIED for the current design] Temporal reaction is not the source of the gain (FH-029).
- [DERIVED, EXP-023, 2026-10-12] FH-029's facts reproduce on all four transition traces (12 / 12 trace × budget rows), and the controller's signed `excess64` is lower than its frozen split's in 11 / 36 comparisons. EXP-023 is FALSIFIED for the current controller; it does not address estimator accuracy or other controller designs.

**Unknown.** Estimator accuracy vs the in-sample optimum, out-of-sample; whether a faster or differently designed controller can track drift; controller overhead in a runtime.

**Related experiments.** EXP-010, EXP-023.

## RQ-007 — How close can an online policy get to the offline oracle?

[DERIVED, 2026-10-04, 30B transition DEV trace, one session] Gaps, counted against the frozen uniform split:
- adaptive-e64 realises 66 / 73 / 79% of the best fixed layer split's potential at 4 / 2 / 1.5 GiB.
- It realises none of the extra 13–27% that instantly switched per-segment optima would add.
- Its time variation costs misses relative to a fixed split. See [TRANSITION-ADAPTIVE-GAIN](docs/research/TRANSITION-ADAPTIVE-GAIN.md).

**Status: OPEN.**

**Known.** [DERIVED, pilot] LRU at 4 GiB: 1.9× the Belady-style misses; at 1.5 GiB: 1.8×. Recent literature reports large persistent Belady gaps (44-46%) mostly attributable to knowing which resident block is used furthest in the future (arXiv 2608.07911).

**Known (2026-10-11, [EXP-007](experiments/EXP-007-router-characterization/README.md), 28 cells).**
- [SUPPORTED] A large gap remains: belady-global minus adaptive-e64 is +10.4 … +20.8 pp on the 30B and +8.5 … +19.9 pp on the 80B.
- [DERIVED] This is substantial headroom relative to a future-informed reference. Whether an online prediction-aware policy can realise a material fraction of it remains [OPEN]. The reference is not a proven bound for multi-key events and mixed sizes.
- Re-splitting memory across layers closes only a small part of it: adaptive vs global LRU is ≤ 8.5 pp at 1.5 GiB and ≤ 1.5 pp at 4 GiB.
- 2Q is competitive with adaptive at some high-memory 80B points.

**Unknown.**
- Which part prediction-aware retention (future-use estimates per expert) can close.
- Online policies beyond those replayed (13 policies plus the adaptive variants).
- ~~The gap on real traces of both models.~~ [SUPERSEDED 2026-10-11 by EXP-007.]

## RQ-008 — Where does expert streaming stop being useful?

**Status: OPEN.**

**Known.** [MEASURED, EXP-022] Extreme-pressure measurements now extend below 1.5 GiB. On 30B, all tested performance launches completed down to 512 MiB: mean throughput declined from 4.00 tok/s at 1.5 GiB to 2.71 at 768 MiB and 2.63 at 512 MiB, while diagnostic expert-level hit rate collapsed from 44.01% at 1.5 GiB to 0.232% at 768 MiB and 0% at 512 MiB. The historical H-5 >30% falsifier at 768 MiB was not met; formal status is INCONCLUSIVE because no positive support criterion was pre-specified. On exploratory 80B, 512 MiB produced two DEGRADED completions and one STORE_RESIDENCY_ABORT.

**Open.** The one-token expert footprint is not a hard execution threshold for the measured workloads, but EXP-022 suggests it may mark a model-specific transition scale for expert reuse. This requires prospective confirmation across additional models, workloads and memory points, and a larger study of probabilistic STORE_RESIDENCY_ABORT behaviour near the lowest budgets.

## RQ-009 — Why does the 80B behave differently from the 30B?

**Status: PARTIALLY ANSWERED** (2026-10-11): the 80B router is now directly characterized; the causal question is open. *[2026-10-16: unchanged after EXP-026 (INCONCLUSIVE); data note below.]*

**Candidate explanations.** (a) Router concentration; (b) smaller slices (440-704 KiB vs 930 KiB) give more reads in flight per byte; (c) I/O bookkeeping (commit/decommit/lock = 56% of the 80B's I/O bucket vs 10% on the 30B) rather than reading; (d) different compute (Gated DeltaNet in 36 layers).

**Known (2026-10-11, [EXP-007](experiments/EXP-007-router-characterization/README.md)).**
- [SUPPORTED] The 80B's routing is directly characterized on 14 cells.
  - It is generally more concentrated than the 30B: global EWS90 fraction 0.225–0.472 vs 0.327–0.554.
  - Per-layer spread is larger (CV 0.16–0.46 vs 0.09–0.25).
  - It exposes much less exploitable inter-layer allocation opportunity: in-sample uniform excess 0.5–3.9% vs 3.5–10.4%; adaptive-over-global-LRU gain ≤ 3.2 pp vs ≤ 8.5 pp at 1.5 GiB.
  - Measured runtime expert-level hit rate: 61.4–78.1% at 4 GiB.
- Candidate (a) is thereby measured, not inferred. It is consistent with the 80B reaching similar expert-level hit rates at less than half the resident fraction (11.7% vs 24.5% at 4 GiB), but it does not rule out (b)–(d).

**Unknown.**
- The relative contribution of (b) slice size, (c) bookkeeping and (d) architecture to the speed difference (EXP-026). The I/O decomposition still comes from one run.
  - *[2026-10-16, [EXP-026](experiments/EXP-026-30b-vs-80b-regimes/README.md), INCONCLUSIVE] Still unknown for all four candidates (a)–(d): all are NOT EVALUABLE under the frozen protocol, because the traced-slope agreement gate failed.*

**Data note (2026-10-16, [EXP-026](experiments/EXP-026-30b-vs-80b-regimes/README.md), frozen protocol EXP026-P1).**
- [DERIVED] The historical slope difference is not reproduced at matched budget/F; consistent with a normalization/region difference.
  - Untraced, 4 generation threads, P1 → P2 = 3.92 F → 1.47 F: β30 = 0.4756, β80 = 0.4948, Δβ = +0.0192. The launch-pairing envelopes overlap: NOT DISTINGUISHED.
  - This does not retrospectively falsify EXP-006 or alter F-004 (different budget/F regions, earlier binaries).
- [LIMITATION] The traced diagnostic slopes disagree (traced Δβ = −0.2694), so the preregistered agreement gate failed and no candidate was discriminated.
- ~~No router trace of the 80B exists.~~ [SUPERSEDED 2026-10-11: EXP-007 recorded 9 runs in each of 14 80B cells, plus replacements.]

**Related experiments.** EXP-006, EXP-026.

## RQ-010 — What is the true expert-level hit rate of the runtime?

**Status: ANSWERED** (2026-10-11) for the instrumented binary, both models, the seven EXP-007 workload families, DEV/EVAL and budgets 1.5–4 GiB. Earlier status, kept as history: "ANSWERED for the 30B coding workload; OPEN for the 80B and other workloads".

**Known.** [MEASURED, test model] All misses happen in the first node of a layer. The other two nodes re-pin and always hit. Runtime-style 97.7% vs expert-level 93.1% ([EXP-020](experiments/EXP-020-hit-rate-semantics/README.md)).

**Known.** [MEASURED, 30B, coding dev, 3 runs per budget] Runtime expert-level hit rate, defined as gate-node slice hits over prefill + decode, workload only: 70.60 / 49.75 / 41.46% at 4 / 2 / 1.5 GiB. Runtime-style: 90.2 / 83.2 / 80.5%. The miss ratio is exactly 3. Restricted to decode, the gate-slice hit rate is 75.4 / 53.4 / 44.5% and the expert-event hit rate (all 3 slices resident) is 73.7 / 51.7 / 43.1%. The phase and the definition move the number by up to 4.7 pp ([EXP-021](experiments/EXP-021-replay-fidelity/README.md)); always state which one.

**Known (2026-10-11, [EXP-007](experiments/EXP-007-router-characterization/README.md), `runtime-hit-rates.csv`, 252 qualified runs).** [MEASURED] Workload-only expert-level hit rate (gate-slice hits, prefill + decode):

| | 4 GiB | 2 GiB | 1.5 GiB |
|---|---|---|---|
| 30B | 66.5–79.8% | 44.5–57.7% | 35.8–48.5% |
| 80B | 61.4–78.1% | 46.6–67.4% | 40.7–62.1% |

- Runtime-style 87.1–93.3% at 4 GiB.
- The repetition SD is ≤ 0.13 pp.
- The expert-level / runtime-style miss ratio is exactly 3.000 in every qualified run.
- The earlier expectation of "≈74% expert-level at 4 GiB" for the 80B was derived from Argine v1's 91.2% on a different binary and prompt. The measured 80B coding-DEV value is 65.9%, and the 80B workloads span 61.4–78.1%. The derivation rule (runtime miss = expert miss / 3) holds exactly; the specific 74% figure was workload-specific.

**Unknown.** Other binaries, additional workloads and models, and the generality of the extreme-pressure behaviour observed by EXP-022. ~~The 80B.~~ [SUPERSEDED 2026-10-11.]

## RQ-011 — Does replay reproduce the runtime?

**Status: ANSWERED (2026-10-04).** `store-replay` with per-thread pins reproduces every runtime decision of the measured workload on format-v2 traces. This holds on Linux, on a fresh 30B Windows run, and on all nine 80B Windows runs. The only differences are in the flush-sentinel tail, explained by records lost at process termination (write-up §14). [2026-10-12] A further fresh 30B Windows run (EXP-019 validation, 4 GiB, coding DEV) agrees: replay is event-exact over the workload with 0 tail disagreements, and the record loss (5 records, 1 node) is confined to the flush sentinel.

**Known.** [MEASURED] `store-replay`, the slice-level, per-shard, pin-aware LRU fed the runtime's request order, reproduces every decision of the real store: 0 disagreements in a C++ driver and in 89,892 + 89,928 requests of two native Linux runs with recorded shard maps.

[DERIVED] On the nine Windows 30B runs (format v1, shard map modelled) it is within +0.10…+0.28 pp, with 0.7% event disagreement, the size the hash-map control predicts.

[FALSIFIED, 2026-10-04] On the first Windows v2 trace the replay above disagreed on 5.29% of decisions. Its pins were node-scoped for all threads; the runtime's are per thread. With `--threads 4 --threads-batch 12`, the threads idle in one phase keep the pins of the last node they ran: 78,394 of 144,689 evictions passed over such slots.

[SUPPORTED, same trace] With per-thread pins: 0 of 1,474,909 decisions differ, with identical skip counts, misses, bytes and evictions. The 3 sequence gaps are records lost at termination inside the flush sentinel, so replay-irrelevant. On the nine v1 runs the per-thread model gives −0.07…+0.06 pp. Details: [write-up §13](docs/research/REPLAY-FIDELITY-VALIDATION.md#13-windows-v2-follow-up-2026-10-04).

[FALSIFIED] The former `sharded-lru-16` (expert units) is not a runtime model: +3.5…+4.8 pp on the runtime metric, +1.5…+2.1 pp at matched metric. It is renamed `sharded-lru-16-expert` (counterfactual). [Write-up](docs/research/REPLAY-FIDELITY-VALIDATION.md).

**Unknown.** Fidelity under read failures and in-flight waits, which never occurred in EXP-007. *(A duplicated sentence with a broader, unverified scope was removed by the 2026-10-12 audit.)*

## RQ-012 — What does the instrumentation cost?

**Status: ANSWERED for the tested EXP-024 configuration; the preregistered hypothesis is INCONCLUSIVE overall.** ([EXP-024](experiments/EXP-024-tracing-overhead/README.md): session 20261012T113829-5edc0f3f, 20 / 20 valid launches, frozen protocol EXP024-P1.)

**Known.** [MEASURED, paired] Enabled vs disabled research instrumentation on the same `llama-research-v2` binary: 30B Q4_K_M, asynchronous store, CPU, threads 4 / batch 12, the standard benchmark (128 tokens), 5 paired rounds per budget.
- 4G: mean overhead 1.099%, CI95 [0.605, 1.593]%, so < 2% is supported at 4G.
- 1.5G: mean overhead 0.549%, CI95 [−1.379, 2.478]%, so < 2% is inconclusive at 1.5G. The paired SD is 1.55 pp, against 0.40 pp at 4G.

**Unknown.**
- Whether the 1.5G mean is below 2%. That needs a separate prospective follow-up, not extra EXP-024 rounds.
- Other machines, models, thread settings, budgets and workloads.
- How the ON cost splits between recording and the Windows precise completion wait.
- The cost of the disabled hooks against an unpatched binary.

*(Earlier status, kept as history: "OPEN. EXP-024: interleaved on/off, 5 repetitions. EXP-019's 2026-10-12 validation has no trace-off control and does not measure overhead.")*

## RQ-013 — Can an interpretable model predict ms/token within 10-15%?

**Status: ANSWERED for the tested primary configuration** (2026-10-15, [EXP-025](experiments/EXP-025-performance-model/README.md)). The preregistered primary test was SUPPORTED: the 30B leave-one-configuration-out median absolute percentage error was 2.2546885%, below the 15% success threshold.

**Known.** [SUPPORTED, EXP-025] The model `T_token = T_compute + T_unhidden_IO + T_sync + T_controller` predicts measured token latency accurately across the tested 30B configurations. The result supports the predictive usefulness of the decomposition, not separate causal identification of every term.

**Limitation.** The tested configurations do not independently identify all physical components: the intercept/compute and synchronization contributions are collinear in the design. In particular, a fitted synchronization contribution near zero must not be interpreted as evidence that physical synchronization cost is zero.

**Unknown.** Generalisation to the 80B, other binaries, machines, workloads and pressure regions.

## RQ-014 — How many threads on the current build?

**Status: ANSWERED for the tested configuration** (2026-10-11, [EXP-027](experiments/EXP-027-current-thread-sweep/README.md)). The answer is not one universal thread count: the optimum is model-dependent.

**Known.** [SUPPORTED, EXP-027] On the current `llama-research-v2` binary, CPU backend, asynchronous Expert Store at 4 GiB, with the standard benchmark workload and 3 independent rotated launches per configuration:
- Qwen3-30B-A3B Q4_K_M is fastest with **12** generation threads: 6.74 tok/s, +1.7% over 8, +3.5% over 4.
- Qwen3-Next-80B-A3B-Instruct Q3_K_M is fastest with **4**: 6.91 tok/s, +2.0% over 12, +4.1% over 8.
- Both are CLEAR BEST under the pre-registered criterion, so no fourth round was needed.
- The pre-registered predictions (4 threads not slower on the 30B; 12 threads slowest on both) are FALSIFIED.
- Recommended for future performance-sensitive runs on this exact runtime: 30B `--threads 12`, 80B `--threads 4`, prompt threads 12.

**Unknown.** Other budgets (2 / 1.5 GiB), other binaries and machines, longer generations, and between-session variation.

EXP-001 itself is historically closed as INCONCLUSIVE ([audit](docs/research/HISTORICAL-AUDIT-EXP001-006.md)); EXP-027 is new evidence and does not rewrite it. Earlier status, kept as history: "OPEN". The "one thread per P-core" rule was measured on the OpenMP build; the dense 8B on the official build preferred 8-12 threads ([EXP-001](experiments/EXP-001-thread-sweep/README.md)). Suggested: `argine benchmark --sweep-threads 4,6,8,12` on the current binary, interleaved, with both models.

## RQ-015 — What causes 15-19% differences between sessions?

*[Audit scope, 2026-10-11: "15-19% differences between sessions" in the heading is historical shorthand. It refers to legacy cross-run contrasts under incomplete runtime provenance (30B nominal groups ≈15.4% / ≈16.0%; 80B supplementary pair ≈19%), not a measured session noise floor; part may be hidden configuration.]*

**Status: OPEN.** [MEASURED, SUPPLEMENTARY] 80B 4 GiB: 7.10 vs 5.74 tok/s eight minutes apart on one binary; not a builder-matched group, and the read mode is recorded only for the first record ([EXP-018](experiments/EXP-018-reproducibility-audit/README.md)). Candidates: thermal state, background load, page cache, power plan, unrecorded flags. Suggested: log CPU frequency, temperature and system load per run (not implemented).

## RQ-016 — Is hit rate the right objective?

**Status: OPEN.** [MEASURED, Argine] Cost per miss *falls* as the budget shrinks (1.233 → 1.063 ms per read) because more misses per node raise queue depth. A policy with fewer misses but lower queue depth can therefore save less time than its hit rate suggests. Primary metric remains tokens/s.

[SUPPORTED, EXP-025] A time-based decomposition that includes unhidden I/O predicts measured token latency accurately in the tested 30B configurations (median leave-one-configuration-out absolute percentage error 2.2546885%). This strengthens the evidence that miss count or hit rate alone is insufficient as a performance model, but it does not establish an optimal control objective.

## RQ-017 — What exactly is Least-Stale, and does it help here?

**Status: OPEN.** The available description ("evict blocks untouched in the current forward cycle first") leaves the order among stale blocks unspecified. With LRU order it is provably identical to LRU in a global scope; with MRU order it performed badly in the pilot. [DERIVED, EXP-007 run-00, v2 replay] Still badly: 46.9 / 34.6 / 31.4% vs LRU 75.7 / 53.3 / 44.1% at 4 / 2 / 1.5 GiB, and 'stale-first, LRU among stale' is still identical to global LRU. Suggested: read SpecMD (arXiv 2602.03921) and correct `policies.LeastStale`.

## RQ-018 — Do slice-level residency decisions matter?

**Status: PARTIALLY ANSWERED.**

**Known.** [MEASURED, 30B coding dev] In the runtime, 4,563 decode expert events at 4 GiB (≈3.1%) and 5,205 at 2 GiB found only 1 or 2 of their 3 slices resident.

[DERIVED] Slice granularity, together with non-atomic nodes and slice-keyed sharding, is why the runtime makes 1.5-2.1 pp fewer expert-event hits than the same LRU on whole experts (EXP-021). It is also ~1.8 pp below a global expert LRU at 4 GiB.

**Unknown.** Whether a whole-expert store (one key per expert, or co-sharded slices) would recover that difference in the runtime. That is a design question, deliberately not pursued during fidelity validation.
