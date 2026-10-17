# Where the adaptive gain comes from — 30B transition trace (2026-10-04)

All results here are **counterfactual replays of one router trace**, `EXP-007 raw/30b-transition-dev-v2/run-00-4G`:
- workload `transition@5ba6e15e2b68`;
- 30B Q4_K_M;
- 1,196 counted decode tokens.

They are not runtime measurements. In-sample and oracle rows are labelled as such.

Code: [`tools/transition_analysis.py`](../../tools/transition_analysis.py). Data and figures: [`experiments/EXP-007-router-characterization/analysis-v3/30b-transition-run-00-4G-temporal/`](../../experiments/EXP-007-router-characterization/analysis-v3/30b-transition-run-00-4G-temporal/). The earlier analysis (`analysis-v3/30b-transition-run-00-4G/`) is unchanged. Its aggregate numbers are reproduced exactly here; every policy's total misses match `adaptive.csv`.

Taxonomy as in [`METHODOLOGY.md`](../METHODOLOGY.md).

## 1. Conclusion

[SUPPORTED, one trace] **Explanation A.** The adaptive controller wins because it moves the budget towards a generally better layer allocation, not because it reacts to transitions.
- **A fixed allocation does better.** The controller's own time-averaged allocation, applied as a fixed allocation from the first step, saves **more** misses than the controller itself: 135 / 118 / 116 % of its gain at 4 / 2 / 1.5 GiB.
- **Variation over time costs misses.** The controller's movement over time contributes −1,427 / −965 / −1,153 misses: the cost of learning from a uniform start and of lagging behind each segment.
- **The gain is not concentrated after transitions.** The 64 steps after the three boundaries are 16.1 % of the steps and hold 14.0 / 13.8 / 15.2 % of the gain. The first segment, which follows no transition, holds 27 / 26 / 21 %.
- **The controller does not shorten transition transients.** Its boundary excess equals that of the frozen uniform split.

[SUPPORTED, oracle bounds, one trace] Fast reallocation (B) has value in this trace, but it is smaller than A, and the present controller captures none of it.

| budget | potential of A: best fixed allocation vs frozen uniform | potential of B: per-segment optimum switched instantly vs best fixed allocation |
|---|---|---|
| 4 GiB | 6,147 misses (1.34 pp) | 1,658 misses (0.36 pp) = 27 % of A |
| 2 GiB | 7,516 (1.64 pp) | 1,736 (0.38 pp) = 23 % |
| 1.5 GiB | 9,272 (2.02 pp) | 1,180 (0.26 pp) = 13 % |

The controller's per-segment allocations do carry segment-specific value: applied instantly, they would save 417 / 797 / 512 misses more than its fixed mean. It reaches them so late, though, that the lag costs 1,844 / 1,762 / 1,665 misses.

Verdict on the four options: **A**. B is not material for this controller. B as a mechanism is [HYPOTHESIS]: an upper bound of 13–27 % of the static potential, unrealised.

[LIMITATION] **No policy-level replication is possible from this campaign.** The workload routing of all nine runtime repetitions is bit-identical (`replication.csv`: one SHA-256 over 1,202 workload steps for all nine traces). Replaying the policies on the other eight traces would reproduce these numbers exactly. The confidence intervals below are within-trace block bootstraps; they do not measure between-session variation. That needs other sessions: the eval split, other dev sessions.

## 2. Segments and boundaries

The segments come from the trace's own records (`segment` of every step); the workload metadata agrees.

| segment | counted decode steps | trace steps | boundary (decode index) |
|---|---|---|---|
| natural-language | 0–298 | 11–309 | — |
| coding | 299–597 | 311–609 | 299 |
| reasoning-math | 598–896 | 611–909 | 598 |
| structured-json | 897–1195 | 911–1209 | 897 |

Each turn is one request (prompt step + 299 decode steps) in a single conversation; the earlier turns are cached (`cache_n` 3 → 104). Warm-up and flush sentinel are served but not counted.

## 3. Per-segment performance

Expert-event hit rate per segment. Full table with misses/token, stall/token, bytes, moves, and mean/peak per-epoch movement: `segments.csv`.

| budget | policy | natural-language | coding | reasoning-math | structured-json |
|---|---|---|---|---|---|
| 4 GiB | ablate-frozen (uniform, controller code path) | 81.55 | 76.14 | 71.72 | 75.22 |
| | adaptive-e64 | 82.50 | 77.26 | 72.03 | 76.35 |
| | adaptive-e64-pin50 | **83.36** | 76.71 | 71.09 | 71.06 |
| | global-lru | 81.64 | 75.95 | 70.94 | 75.45 |
| 2 GiB | ablate-frozen | 59.32 | 53.78 | 50.87 | 55.37 |
| | adaptive-e64 | 60.58 | 55.45 | 51.86 | 56.24 |
| | adaptive-e64-pin50 | **61.91** | 53.74 | 50.34 | 51.41 |
| 1.5 GiB | ablate-frozen | 51.95 | 46.78 | 43.45 | 48.10 |
| | adaptive-e64 | 53.29 | 48.63 | 45.07 | 49.63 |
| | adaptive-e64-pin50 | **54.30** | 47.10 | 43.85 | 46.01 |

[DERIVED] adaptive-e64 minus frozen, misses per step, with a within-trace 95 % CI (block bootstrap, blocks of 32 steps):
- **4 GiB:** −3.6 [−5.4, −1.9] in natural language, −4.3 in coding, −1.2 [−3.3, +0.9] in reasoning-math (not distinguishable from 0), −4.3 in structured JSON.
- **2 and 1.5 GiB:** every segment's interval excludes 0 (`bootstrap.csv`).

[MEASURED, code] **Baseline used for "adaptive gain".** `per-layer-lru-uniform` and `ablate-frozen` have the same allocation but different serving code for events larger than a partition (prompt steps).
- `per-layer-lru-uniform` streams the overflow, keeping the last members.
- The controller refuses overflow admissions, keeping the first members.

That alone is worth 340 / 43 / −5 misses at 4 / 2 / 1.5 GiB, concentrated in the first prompt. Gains are therefore stated against `ablate-frozen`, which uses the same code path. Against `per-layer-lru-uniform` the 4 GiB gain reads 8 % larger.

## 4. Boundary-local behaviour

Windows per boundary:
- `pre32`: the last 32 steps before the boundary;
- `post16`, `post32`, `post64`: the first 16, 32 and 64 steps after it;
- `steady`: the **last 128 steps of the new segment**.

Excess is the sum over the window of (misses − the new segment's steady misses per step). `boundaries.csv` also gives the moves, the number of layers changed and the allocation change D in each window.

**Gain of adaptive-e64 over frozen, misses per step:**

| budget | boundary | pre32 | post16 | post32 | post64 | steady |
|---|---|---|---|---|---|---|
| 4 GiB | NL → code | +6.2 | −0.1 | +1.2 | +3.0 | +4.1 |
| | code → math | +4.0 | −0.2 | −1.4 | +2.4 | −0.2 |
| | math → JSON | +2.8 | +1.4 | +1.8 | +3.5 | +5.1 |
| 2 GiB | NL → code | +7.4 | +2.8 | +5.0 | +4.0 | +8.0 |
| | code → math | +4.8 | −3.0 | +2.0 | +2.6 | +5.0 |
| | math → JSON | +1.6 | +4.4 | +5.2 | +5.2 | +3.2 |
| 1.5 GiB | NL → code | +6.6 | +5.9 | +7.2 | +6.9 | +8.1 |
| | code → math | +8.9 | +2.9 | +4.9 | +4.9 | +6.8 |
| | math → JSON | +4.2 | +5.6 | +5.4 | +5.6 | +5.3 |

[DERIVED] Reading the table:
- The adaptive advantage is **smallest right after a transition**, below its `pre32` value in 7 of 9 cases. It recovers as the segment proceeds.
- An allocation specialised to the previous segment is worth less at the start of the next one; nothing is gained by reacting.

**Transient excess.** The transient is a property of the cache content, not of the allocation. In the 64 steps after the NL → code boundary at 4 GiB:
- frozen: excess +1,465 misses;
- adaptive-e64: +1,539;
- global LRU: +1,703.

Summed over the three boundaries:

| | 4 GiB | 2 GiB | 1.5 GiB |
|---|---|---|---|
| adaptive-e64 | 3,218 | 592 | 36 |
| frozen | 3,206 | 314 | −151 |

Allocation movement inside the windows (e64): 2–29 block moves and 4–28 layers changed in the first 64 steps; D ≤ 0.026 of the budget.

## 5. Allocation-vector dynamics

**Distance metric:** D(a, b) = ½ Σ_L |a_L − b_L| / B. It is the minimum fraction of the budget B that must move between layers to turn allocation a into b; it ranges from 0 to 1.

Reference vectors:
- **uniform**;
- **whole-trace optimum:** exact DP on the trace's LRU curves (in-sample);
- **segment optimum:** exact DP on the curves of one segment, with all earlier steps still served.

Files:
- `allocation_timeline.csv`: per-layer slots at every epoch end;
- `allocation_distance.csv`: distances over time;
- `allocation_boundary.csv`: the vectors before / +16 / +32 / +64 / segment end, with gaining and losing layers and the cosine between the controller's shift and the optimum's shift.

[DERIVED] Findings:
- **Segment optima differ from each other:** D = 0.05–0.15 between consecutive segments. They are each fairly close to the whole-trace optimum: D = 0.03–0.08.
- **The controller drifts to a stable compromise.** Its allocation moves away from uniform (D 0.07–0.13) towards the whole-trace optimum (D 0.03–0.06 for its mean allocation) and stays there. It does not follow the segment optima: in the distance figure the distance to the current segment's optimum jumps at each boundary and is only partly recovered before the next one.
- **Direction is partly right:**
  - after 8 of 9 boundaries the allocation ends the segment closer to the new segment's optimum than it began;
  - in 7 of 9 it moves away from the old segment's optimum;
  - the cosine between its shift and the optimum's shift is +0.11 to +0.48 (one case −0.21).
- **Speed is the problem.** The allocation reaches within D ≤ 1 % of its own end-of-segment vector only after 157–297 of the segment's 299 steps. It is still moving when the next segment begins.
- **Why it is slow:** four parameters set the reaction time:
  - one decision per epoch (64 trace steps for e64);
  - the utility histograms decay by ½ per epoch, so the previous phase still carries ~½ of the evidence one epoch after a switch;
  - at most 5 % of the budget moves per epoch (the peak observed per-epoch movement is 5.1 %, so the cap binds at 2 and 1.5 GiB);
  - 10 % hysteresis.

  A segment-optimal shift of 5–15 % of the budget therefore needs 1–3 epochs **after** the estimator has noticed it. That is at least 128–256 steps for e64, in segments of 299.
- **Movement:** e64 makes 175–354 block moves per trace, at most 5.1 % of the budget per epoch.

Figures: `allocation_distance.png`, `allocation_heatmap.png` (per-layer slots vs uniform, 2 GiB).

## 6. Attribution of the gain (A vs B)

Misses over the 1,196 counted steps:

| budget | frozen uniform | adaptive-e64 (online) | fixed: adaptive's mean allocation* | fixed: adaptive's final allocation* | fixed: whole-trace optimum (oracle) | adaptive's per-segment allocations, instant switch* | per-segment optimum, instant switch (oracle) |
|---|---|---|---|---|---|---|---|
| 4 GiB | 109,486 | 105,456 | 104,029 | 104,460 | 103,339 | 103,612 | 101,681 |
| 2 GiB | 207,421 | 201,930 | 200,965 | 201,159 | 199,905 | 200,168 | 198,169 |
| 1.5 GiB | 240,795 | 233,501 | 232,348 | 232,340 | 231,523 | 231,836 | 230,343 |

\* Diagnostic replays. They use the controller's own decisions, taken from the whole trace (future information), as fixed or instantly switched allocations. They are not online policies.

**Cross-segment check** (`cross_segment.csv`): misses of segment s under the static optimum of segment s′.
- Another segment's optimum recovers 21–80 % of what a segment's own optimum gains over uniform; median ≈ 55 %.
- The whole-trace optimum recovers 52–90 %.

Most of the value of a better split is common to all four workloads.

## 7. The `adaptation_steps` metric

[SUPPORTED] At 2 and 1.5 GiB the historical value 16 is a **floor artefact**, not convergence. Rows equal to 16: 27 / 30 at 2 GiB and 28 / 30 at 1.5 GiB (`adaptation_v1_audit.csv`). Two properties of the definition (`replay.adaptation_metrics`) cause it:
1. **The search starts at k = window = 16**, so 16 is the smallest value the metric can return. A segment with no transient at all also returns 16 (regression test).
2. **The tolerance is relative:** 10 % of the steady miss rate. Under memory pressure the steady rate is 160–230 misses per step, so the band is 17–20 misses per step. That is wider than the transient: the median first-window excess is +2.1 misses per step at 2 GiB and −4.1 at 1.5 GiB. Every first window is already "converged". The band is also only 0.2–0.4 standard deviations of per-step noise.

At 4 GiB (steady ≈ 80–110 misses/step) the metric does move (19–91), but it is still threshold-driven.

The old definition and every historical `adaptation.csv` are unchanged; a docstring now states the limitation. **Proposed replacement** (`adaptation_v2.csv`), all in absolute misses:
- excess16, excess32, excess64 against the steady state of the last 128 steps of the new segment, signed and positive-part;
- a noise reference: steady per-step SD × √64, and whether |excess64| > 2 of these;
- the half-life of the cumulative excess;
- allocation convergence: steps until D to the segment-final allocation ≤ 1 %.

With it the policies separate where v1 could not. For example, summed excess64 at 2 GiB: frozen 314, e64 592, pin25 3,376, pin50 5,158.

[LIMITATION] One trace gives three boundaries per budget. Single-boundary excesses at 2 GiB are often below 2 × the noise reference.

## 8. Pinning

[SUPPORTED, one trace; a negative result, not a bug] Hot-set pinning **helps in stationary stretches and hurts after transitions.**
- In the first segment (no preceding transition) pin50 is the best policy at every budget: 83.4 % vs e64 82.5 % at 4 GiB.
- After transitions it is the worst: structured JSON at 4 GiB, 71.1 % vs 76.4 %.
- Over the whole trace it costs 1.0–1.7 pp against e64.

Mechanism, from `pinning.csv`:
- **The pins belong to the previous phase.** Pins are re-chosen every epoch by decayed frequency × stability, where stability is the share of the last 4 epochs (256 trace steps) in which the expert was in the layer's EWS90 set. A 299-step segment is barely longer than that window.
- **Measured at a boundary:**
  - 31–62 % (4 GiB) and 62–85 % (1.5 GiB) of the pinned experts are hot only in the old segment;
  - the share falls by +128 steps to 9–35 % (4 GiB) or 7–68 % (1.5 GiB), and to 0 by +256;
  - 9–29 % of the budget is held by pinned experts that are not accessed in the next 32 steps.
- **Pins do not block redistribution between layers.** The layer-allocation trajectory of pin25 and pin50 is identical to e64 at 4 and 2 GiB (same moves, same bytes moved). The loss is **within** layers: stale pinned slots shrink the LRU capacity the new phase can use.
- **Consequence:** summed excess64 over the three boundaries at 4 GiB is 5,563 (pin25) and 6,506 (pin50), against 3,218 for e64.

Contrast with the stationary coding trace: pinning **helps** there (+0.25 / +0.73 pp for pin25 / pin50 at 4 GiB). The sign of the pinning effect flips with stationarity. This is the clearest stationarity-dependent effect in the data.

Figure: `pinning_stale_share.png`.

## 9. Hysteresis and cost weighting

**Hysteresis** (`ablations.csv`):

| | 4 GiB | 2 GiB | 1.5 GiB |
|---|---|---|---|
| block moves, e64 → no hysteresis | 354 → 487 (+38 %) | 242 → 363 (+50 %) | 175 → 185 (+6 %) |
| misses without hysteresis | +46 | +72 | −22 |
| summed excess64 over the boundaries | 3,218 → 3,222 | 592 → 675 | 36 → −45 |

The miss changes are ≤ 0.016 pp of hit rate. [DERIVED] Hysteresis buys 5–33 % less churn for no measurable change in hits or transients on this trace.

**Cost weighting** is identical at 4 and 1.5 GiB, and 105 misses (0.02 pp) apart at 2 GiB. [SUPPORTED] This is expected, but **not** from equal sizes: the expert sizes differ (2.65 vs 3.06 MB).
- The calibrated cost model has zero per-read overhead, so the stall of a node is bytes / bandwidth(queue depth).
- cost_per_miss(L) / bytes(L) is then the same 1/bandwidth for every layer within an epoch.
- Weighting multiplies every slot value by a common factor, which in exact arithmetic changes no decision.

The decision audit (`cost_weighting_check.csv`) replays the weighted controller and computes the unweighted decision on the same state at every step:
- 3 / 12 / 7 decisions differ;
- **all** of them are exact utility ties between layers of the two expert sizes, where a 1.5 × 10⁻¹⁶ rounding difference in the per-byte factor replaces the layer-index tie-break;
- no hysteresis test ever flips.

Cost weighting becomes meaningful only with a non-zero per-read cost, i.e. a calibrated per-read overhead.

## 10. Transition versus stationary coding

Routing statistics depend on window length, so equal windows are compared (`stationarity.csv`). The coding rows come from the EXP-007 coding trace.

| window | decode tokens | EWS90 | EWS95 | EWS99 | hot-set stability | reuse median | overlap with previous token |
|---|---|---|---|---|---|---|---|
| coding (stationary), first 299 tokens | 299 | 0.367 | 0.456 | 0.611 | 0.705 | 2.17 | 0.426 |
| transition: natural-language | 299 | 0.275 | 0.335 | 0.455 | 0.815 | 2.06 | 0.458 |
| transition: coding | 299 | 0.349 | 0.434 | 0.590 | 0.714 | 2.48 | 0.410 |
| transition: reasoning-math | 299 | 0.396 | 0.489 | 0.631 | 0.638 | 2.60 | 0.370 |
| transition: structured-json | 299 | 0.430 | 0.521 | 0.673 | 0.542 | 2.27 | 0.410 |
| transition: all four segments | 1,196 | 0.552 | 0.653 | 0.806 | 0.497 | 2.38 | 0.411 |

Policy comparison, expert-event hit in pp, against `ablate-frozen`:

| | coding (381 tokens) 4 / 2 / 1.5 GiB | transition (1,196 tokens) 4 / 2 / 1.5 GiB |
|---|---|---|
| adaptive-e64 gain | +0.80 / +1.07 / +1.43 | +0.88 / +1.20 / +1.59 |
| gap to in-sample static optimum | 0.44 / 0.66 / 0.75 | 0.38 / 0.42 / 0.42 |
| pin25 / pin50 vs e64 at 4 GiB | +0.25 / +0.73 | −0.98 / −1.48 |

[DERIVED] What the comparison shows:
- **Working sets:** the union of four workloads is much larger than any one, with EWS90 0.55 vs 0.27–0.43, and hot-set stability over the whole session halves.
- **Individual segments look stationary:** the coding segment of the transition session closely matches the stationary coding trace.
- **Adaptive gain:** it is 0.08–0.16 pp larger on the transition trace. That difference is confounded with trace length, since a 3× longer trace amortises the controller's learning lag. Most of the gain is explained by a fixed allocation either way.
- **Verdict:** [INCONCLUSIVE] that adaptive allocation becomes more valuable as stationarity decreases. Only pinning shows a clear stationarity dependence, and in the opposite direction: it loses value.

## 11. Claim tags

| claim | tag |
|---|---|
| The adaptive controller's gain over the uniform split on the transition trace comes mainly from a better general layer allocation (A) | **SUPPORTED** (one trace, 3 budgets; between-session replication impossible from this campaign) |
| The gain comes from fast reallocation after transitions (B) | **FALSIFIED** for the present controller on this trace: time variation contributes −965 … −1,427 misses, and boundary windows hold no excess gain |
| Fast per-segment reallocation could add 13–27 % to the static potential | **HYPOTHESIS** (oracle bound, one trace) |
| Adaptive allocation shortens transition transients (EXP-023's pre-registered hypothesis) | **FALSIFIED** on this DEV trace for this controller: excess64 is equal to frozen |
| `adaptation_steps = 16` at 2 / 1.5 GiB means equal fast convergence | **FALSIFIED**: a floor and relative-threshold artefact |
| Hot-set pinning increases transition penalties | **SUPPORTED**: stale previous-phase pins, 9–29 % of the budget dead after boundaries |
| Pinning helps in stationary workloads | **SUPPORTED** (coding trace; first transition segment) |
| Hysteresis reduces churn at no measurable cost in hits | **SUPPORTED** (one trace) |
| Cost weighting is a no-op with the current cost model | **SUPPORTED**: exact algebra, with 22 tie-break flips audited |
| Adaptive allocation is more valuable when workloads are less stationary | **INCONCLUSIVE** (confounded by trace length; one session per workload) |

## 12. Limitations

- [LIMITATION] **One session.**
  - The nine runtime repetitions share bit-identical workload routing; policy replays on them are duplicates, not replicates.
  - Between-session variation requires the eval session and further dev sessions.
- [LIMITATION] **In-sample.** All oracle and fitted allocations are in-sample. The diagnostic replays (§6, marked *) use future information by construction.
- [LIMITATION] **Counterfactual model.** The policies are expert-unit and event-atomic, so this analysis says nothing about the runtime's slice-sharded store (EXP-021).
- [LIMITATION] **Untuned controller.** The epoch, decay, max-move and hysteresis values are the prototype's defaults; they were not tuned here, as the task required. §5 identifies them as the reason for the slow reaction. Tuning belongs in EXP-010 on separated dev / eval traces.
- [RESOLVED 2026-10-04, see [TEXT-PROVENANCE](TEXT-PROVENANCE.md)] **Generated text not recorded.** `responses.jsonl` reports `text_sha256 = e3b0c442…` (the SHA-256 of the empty string) for every turn. Cause: the harness hashed only `message.content`, but the 30B's tokens were returned in `reasoning_content` (thinking mode). These runs stay provenance-limited. Two consequences: on the 30B the segments are reasoning-mode generation about each task, and the assistant turns in the session history were empty. Routing identity is established from the trace instead.

## 13. Status

[SUPERSEDED, 2026-10-11: EXP-007 is SUPPORTED and closed (28 / 28 cells). The same temporal analysis on the 30B EVAL and 80B DEV/EVAL transition traces reproduces this document's main finding; see FH-029 and EXP-007 claim C10. The status as written on 2026-10-04 follows.]

**EXP-007: unchanged, RUNNING.** Its full scope is both models, every workload of `workloads/v1`, and dev and eval splits. Done so far:
- 30B: coding dev and transition dev;
- 80B: coding dev.

The routing characterization gained a non-stationary workload, but the closure criterion is not met.
