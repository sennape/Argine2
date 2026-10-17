# EXP-007 — Per-layer routing characterization

| field | value |
|---|---|
| ID | `EXP-007-router-characterization` |
| Status | **SUPPORTED** |
| Collection / analysis | **CLOSED** (2026-10-11): 28 / 28 cells complete and EXP-021-qualified. `CLOSED` is not a registry status, so the machine-readable status stays SUPPORTED. |
| Period | 2026-10-03 → 2026-10-11 |
| Origin | Argine2 |
| Related questions | RQ-001, RQ-002, RQ-003, RQ-004, RQ-005, RQ-006, RQ-007, RQ-009, RQ-010 |
| Supersedes / superseded | — |

## Research question

How do routed expert accesses behave per layer: entropy, concentration, working sets, reuse, co-activation?

## Hypothesis

[HYPOTHESIS] Routing statistics differ significantly between layers and between the 30B and the 80B.

Verdict (2026-10-11): [SUPPORTED]. Per-layer routing statistics differ in every one of the 28 cells, and the two models differ systematically (see Result). The size of the *memory-value* consequence of that difference is model- and workload-dependent, and is much smaller on the 80B.

## Setup

PILOT (done 2026-10-03): the only router trace then available, the legacy `route.log` of 2026-09-27 (30B, mmap path, one 77-token prompt + 127 generated tokens, greedy).

FULL (done 2026-10-03 → 2026-10-11): `argine research trace` with the instrumented binary on both models (Qwen3-30B-A3B Q4_K_M, Qwen3-Next-80B-A3B-Instruct Q3_K_M), seven workload families, DEV and EVAL splits: **28 / 28 model × workload × split cells**. Each cell has 9 runs (4 / 2 / 1.5 GiB × 3, interleaved). Every cell-budget has exactly 3 EXP-021-qualified runs (`runtime-hit-rates.csv`, column `qualified`).

| workload family | file @ sha12 | decode tokens analysed per cell (run-00-4G) |
|---|---|---|
| natural-language, coding, reasoning-math, structured-json, italian | `workloads/v1/*.json` (`71b170bc0fde`, `460949e7abf0`, `de60c25a04b7`, `74e518b464d0`, `6abe317def83`) | 320–389 |
| transition | `workloads/v1/transition.json` @ `5ba6e15e2b68` (v1 history semantics) | 1,132–1,204 |
| multi-turn-v2 | `workloads/v2/multi-turn.json` @ `e978bde0341f`, `n_predict` 1536 | 3,128–4,946 |

Multi-turn uses **`multi-turn-v2`**. Its assistant history is the non-empty final content only, never reasoning; a turn without final content makes the run not EXP-007-eligible ([MULTITURN-V2-HISTORY](../../docs/research/MULTITURN-V2-HISTORY.md)). It is frozen at `n_predict` = 1536 (`e978bde0341f`). The first version, `ff89429e96c0` with `n_predict` = 1024, was used only by the smoke run `raw/30b-multi-turn-v2-smoke-4G`: all four DEV turns ended with `finish_reason` = `length` (content present but truncated). At 1536 (`raw/30b-multi-turn-v2-smoke-1536-4G`) all four ended with `stop`, with non-empty content. Every 30B and 80B multi-turn-v2 run is `eligible` with `finish_reason` = `stop` on all turns. `transition` keeps v1 history semantics.

Analysis: `argine research analyze` on run-00-4G of each cell (`analysis-v3/<cell>/`), decode phase, replay semantics v2, exact allocator, **trained in-sample**. Transition traces also have a temporal analysis (`analysis-v3/*transition*-temporal/`). Cross-cell tables are collected, without recomputation, by `python tools/exp007_summary.py`:
- `analysis-v3/cross-cell-summary.csv`;
- `analysis-v3/concentration-vs-opportunity.csv` (exploratory);
- `runtime-hit-rates.csv` (per run, with its EXP-021 closure verdict).

## Commands

```bat
argine research convert data/raw/legacy/route.log --to data/processed/legacy-30b-route-request1.jsonl.gz --n-experts 128 --expert-bytes 2857370 --request 1
argine research characterize data/processed/legacy-30b-route-request1.jsonl.gz --out data/processed/pilot-legacy-30b
argine research trace MODEL.gguf --expert-store --budgets 4G --workload coding --split dev --out experiments/EXP-007-router-characterization/raw/30b-coding-dev
argine research trace MODEL.gguf --expert-store --budgets 4G,2G,1.5G --repeat 3 --workload multi-turn-v2 --split dev --out experiments/EXP-007-router-characterization/raw/30b-multi-turn-v2-dev
argine research analyze experiments/EXP-007-router-characterization/raw/30b-multi-turn-v2-dev/run-00-4G/trace.jsonl.gz --budgets 4G,2G,1.5G --out experiments/EXP-007-router-characterization/analysis-v3/30b-multi-turn-v2-dev-run-00-4G
python tools/validate_replay_fidelity.py ...   # per campaign -> fidelity/<campaign>/ (EXP-021 method)
python tools/exp007_summary.py                 # cross-cell tables (read-only over the outputs above)
```

## Raw data

Pilot: `data/raw/legacy/route.log` → `data/processed/legacy-30b-route-request1.jsonl.gz` → `data/processed/pilot-legacy-30b/` (per_layer.csv, reuse_cdf.csv, expert_frequency.csv, coactivation.csv, characterization.json, figures/). `results.csv` here = pilot per-layer table.

Full campaign:
- `raw/<model>-<workload>-<split>[-v2]/run-NN-<budget>/` holds `raw.tsv`, `trace.jsonl.gz`, `manifest.json`, `summary.json`, `responses.jsonl` and the CSV tables.
- `*-replacement-*` folders hold the single runs that replaced excluded ones (see Limitations).
- `*-smoke-*` folders hold the multi-turn-v2 budget smoke runs. They are not part of any cell.
- `fidelity/<campaign>/` holds the EXP-021 validation per campaign. For `80b-coding-dev-v2` the current verdicts are in `fidelity/80b-coding-dev-v2-regions/`.
- `analysis-v3/` holds the analyses.

`results.csv` remains the pilot per-layer table, kept as history; the campaign's cross-cell results are in `analysis-v3/cross-cell-summary.csv`.

## Result

### Final campaign (2026-10-11): claims

Source for every number below: `analysis-v3/cross-cell-summary.csv` (counterfactual replay: expert-event hit rate, decode, run-00-4G of each cell) and `runtime-hit-rates.csv` (measured runtime). Counterfactual rows are comparable with each other, never with the measured runtime (EXP-021).

**C1 — [SUPPORTED] The effective working set is well below the nominal expert population, and its size is model- and workload-dependent.**
- [DERIVED] The global EWS90 fraction (experts covering 90% of decode selections, out of 48 × 128 = 6,144 on the 30B and 48 × 512 = 24,576 on the 80B) is 0.327–0.554 on the 30B and 0.225–0.472 on the 80B.
- [DERIVED] Per-layer EWS90 fraction: 0.20–0.71 on the 30B and 0.06–0.71 on the 80B. The across-layer CV is 0.09–0.25 on the 30B and 0.16–0.46 on the 80B.
- [DERIVED] The least concentrated layer is layer 0 in every 30B cell, and layer 0 or 2 in every 80B cell.
- [LIMITATION] These fractions are not length-normalised: single-turn cells have 320–389 decode tokens, transition ≈1.2k and multi-turn 3.1–4.9k.

**C2 — [SUPPORTED] Routing concentration is not a model-agnostic predictor of exploitable inter-layer allocation opportunity.**
- [DERIVED] In multi-turn, the longest and most comparable traces, the 80B is the more concentrated model:
  - global EWS90 fraction: 30B DEV 0.369, EVAL 0.327; 80B DEV 0.267, EVAL 0.226;
  - per-layer routing spread is also larger on the 80B (across-layer CV 0.36–0.46 vs 0.21–0.25).
- [DERIVED] Yet the 80B's static inter-layer allocation opportunity is much smaller. Uniform-split excess misses over the optimal static per-layer LRU split (IN-SAMPLE upper bound):

  | | 4 GiB | 2 GiB | 1.5 GiB |
  |---|---|---|---|
  | 30B DEV | 6.95% | 4.15% | 4.50% |
  | 30B EVAL | 7.40% | 4.10% | 4.53% |
  | 80B DEV | 2.05% | 0.489% | 1.49% |
  | 80B EVAL | 3.85% | 0.962% | 1.49% |

- [DERIVED] Across all cells the uniform excess is 3.5–10.4% on the 30B (median 4.5%) and 0.5–3.9% on the 80B (median 1.8%).
- [DERIVED, exploratory, post hoc, `concentration-vs-opportunity.csv`] **Within-model association.** Within one model, more concentrated workloads tend to have a larger uniform excess at 4 GiB: Spearman ρ(EWS90 fraction, uniform excess) = −0.767 on the 30B and −0.934 on the 80B. The association weakens at 2 and 1.5 GiB.
  - Caveats: n = 14 cells per model, unequal trace lengths (320–389 to 3.1–4.9k decode tokens), and no formal predictive test.
- [DERIVED] **Cross-model non-transferability.** Pooled across the two models the sign reverses (ρ = +0.24 … +0.37), because the model dominates: the more concentrated model (80B) has the smaller opportunity.
- **Conclusion.** Concentration may carry ordinal information about allocation opportunity *within* a model. A concentration value does not transfer *across* models as a predictor of it. The formal predictor comparison belongs to EXP-008.

**C3 — [SUPPORTED] Layer-aware adaptive allocation (`adaptive-e64`) improves over global LRU in most constrained regimes, and the improvement generally grows as the budget shrinks.** It is regime-dependent, not universally the best policy.
- [DERIVED] adaptive-e64 minus global LRU, in pp of hit rate:

  | | 4 GiB | 2 GiB | 1.5 GiB |
  |---|---|---|---|
  | 30B | +0.8 … +1.5 | +2.9 … +4.3 | +5.7 … +8.5 |
  | 80B | −0.3 … +0.9 (negative in 2 cells) | +0.7 … +2.5 | +1.4 … +3.2 |

- [DERIVED] Multi-turn, global LRU → adaptive-e64:

  | | 4 GiB | 2 GiB | 1.5 GiB |
  |---|---|---|---|
  | 30B DEV | 0.802 → 0.812 | 0.579 → 0.616 | 0.487 → 0.548 |
  | 30B EVAL | 0.812 → 0.820 | 0.587 → 0.623 | 0.489 → 0.554 |
  | 80B DEV | 0.762 → 0.766 | 0.636 → 0.649 | 0.585 → 0.605 |
  | 80B EVAL | 0.787 → 0.790 | 0.662 → 0.673 | 0.613 → 0.630 |

- [DERIVED] 2Q matches or beats adaptive-e64 at six points, all on the 80B:
  - multi-turn DEV and EVAL at 4 GiB (0.772 vs 0.766; 0.796 vs 0.790);
  - reasoning-math EVAL at 4 GiB;
  - structured-json DEV and EVAL at 4 GiB, and EVAL at 2 GiB.
- [DERIVED] **Decomposition.** At 2 and 1.5 GiB most of the gain over global LRU is already obtained by the *static uniform per-layer partition* (`per-layer-lru-uniform`):
  - 30B at 1.5 GiB: +4.3 … +6.7 pp from partitioning, plus +1.4 … +1.8 pp from the adaptive split;
  - 80B at 1.5 GiB: +1.0 … +2.6 pp plus +0.4 … +0.7 pp.

  At 4 GiB the uniform partition is roughly neutral (−0.6 … +1.0 pp) and the adaptive part is +0.6 … +1.6 pp on the 30B and −0.0 … +0.5 pp on the 80B.

**C4 — [SUPPORTED] The size of the adaptive benefit is model- and workload-dependent.**
- The 30B exposes more inter-layer allocation opportunity than the 80B (C2, C3), although the 80B's routing is usually more concentrated.
- No causal attribution to architecture is made. The two models also differ in quantisation, expert population, slice size and other properties.

**C5 — [SUPPORTED] DEV → EVAL replication is qualitatively strong.**
- Absolute concentration and hit rates vary between splits: EWS90 fraction up to 0.06 apart, measured expert-level hit rate up to 5.6 pp apart (80B structured-json at 1.5 GiB).
- The main policy trends, the growth of the adaptive gain with memory pressure, and the 30B / 80B difference persist in every DEV/EVAL pair.
- [DERIVED] Signs flip between splits only where the effect is near zero:
  - adaptive-e64 vs global LRU on the 80B at 4 GiB (italian, natural-language);
  - small 30B pinning effects (coding, reasoning-math, structured-json).
- [LIMITATION] This is replication of *patterns* across prompt sets. It is not an out-of-sample test of a fitted policy (no DEV-trained, EVAL-tested allocation was run).

**C6 — [SUPPORTED, workload-specific] Pinning is context-dependent.** adaptive-e64-pin50 minus adaptive-e64:
- **Multi-turn is the clear positive exception.** It is positive on both models, both splits and every budget, at +0.68 … +1.66 pp. Multi-turn EVAL:

  | | 4 GiB | 2 GiB | 1.5 GiB |
  |---|---|---|---|
  | 30B adaptive-e64 | .820 | .623 | .554 |
  | 30B adaptive-e64-pin50 | .835 | .639 | .564 |
  | 80B adaptive-e64 | .790 | .673 | .630 |
  | 80B adaptive-e64-pin50 | .802 | .687 | .644 |

- **Elsewhere it is mostly neutral or harmful.**
  - Range −3.5 … +0.7 pp; positive at 7 of 72 points, all on the 30B (coding DEV at every budget, +0.2 … +0.7 pp).
  - On the 80B outside multi-turn it is negative at all 36 points.
- **Not "pinning helps".**

**C7 — [SUPPORTED] The epoch size barely matters.** e32 / e64 / e128 give nearly identical hit rates: max spread 0.74 pp, median 0.11 pp. They need very different numbers of allocation moves: e32 makes 2.5–11.2× the moves of e128. The update frequency is not the principal source of the gain in these traces.

**C8 — [SUPPORTED] Hysteresis mainly reduces churn.** Removing it changes the hit rate by −0.05 … +0.07 pp and raises allocation moves by 1.00–2.89× (median 1.23×). Hysteresis does not create the hit-rate gain.

**C9 — [SUPPORTED] Cost weighting contributes little under the tested conditions.** The no-cost-weighting ablation is within ±0.05 pp of the default controller in every cell-budget (cf. the exact-algebra no-op finding in [TRANSITION-ADAPTIVE-GAIN](../../docs/research/TRANSITION-ADAPTIVE-GAIN.md)).

**C10 — [FALSIFIED for the current controller] "The adaptive gain on workload transitions comes primarily from rapid reaction to transitions."** ([FH-029](../../FAILED_HYPOTHESES.md))
- [DERIVED] On the 30B transition DEV trace ([TRANSITION-ADAPTIVE-GAIN](../../docs/research/TRANSITION-ADAPTIVE-GAIN.md)):
  - a fixed split equal to the controller's mean allocation captures 118–135% of the controller's gain;
  - the post-transition windows (≈16% of steps) hold ≈14–15% of the gain;
  - the reaction takes 157–297 of each ≈299-step segment;
  - the time variation itself costs misses.
- [DERIVED, replication from `analysis-v3/*transition*-temporal/summary.json`] The 30B EVAL and 80B DEV/EVAL temporal analyses show the same pattern:
  - the fixed mean allocation captures ≥ 113% of the controller's gain in every trace and budget;
  - the reaction-lag cost against instantly switched per-segment splits is negative in every case.

  On the 80B at 2 and 4 GiB the total gain is small (381–1,024 misses), so the shares there are noisy.
- **Conclusion.** The controller mainly finds a better general layer allocation; it is not tracking fast.

**C11 — [SUPPORTED] A large gap remains to the Belady-style oracle.**
- [DERIVED] belady-global minus adaptive-e64: +10.4 … +20.8 pp on the 30B and +8.5 … +19.9 pp on the 80B.
- [DERIVED] Substantial headroom remains relative to the future-informed Belady-style reference.
- [OPEN] Whether an online prediction-aware policy can realise a material fraction of that headroom.
- [LIMITATION] The oracle is a reference, not a proven bound, for multi-key events and mixed sizes.

**C12 — [LIMITATION] In-sample upper bounds.** `static-frequency`, `static-hot-per-layer`, `layer-aware-static`, `oracle-static-lru` and the "optimal static split" behind the uniform-excess numbers are fitted and evaluated on the same trace. They are IN-SAMPLE UPPER BOUNDS, never deployable competitors or out-of-sample results.

**Measured runtime hit rates** ([MEASURED], `runtime-hit-rates.csv`; workload only, 3 qualified runs per cell-budget):

| | 4 GiB | 2 GiB | 1.5 GiB |
|---|---|---|---|
| 30B expert-level (gate-slice, prefill + decode) | 66.5–79.8% | 44.5–57.7% | 35.8–48.5% |
| 30B runtime-style | 88.8–93.3% | 81.5–85.9% | 78.6–82.8% |
| 80B expert-level | 61.4–78.1% | 46.6–67.4% | 40.7–62.1% |
| 80B runtime-style | 87.1–92.7% | 82.2–89.1% | 80.2–87.4% |

- The SD across repetitions is ≤ 0.13 pp.
- The expert-level / runtime-style miss ratio is exactly 3.000 in all 252 qualified runs (EXP-020).

**Cross-cell table** (generated from `analysis-v3/cross-cell-summary.csv`; uniform excess is IN-SAMPLE):

| model | workload | split | global EWS90 fraction | uniform excess, in-sample (4 / 2 / 1.5 GiB) | global LRU → adaptive-e64 (4 / 2 / 1.5 GiB) | Belady-style (4 / 2 / 1.5 GiB) |
|---|---|---|---|---|---|---|
| 30B | natural-language | DEV | 0.337 | 10.40% / 5.56% / 4.94% | 0.811 → 0.821 / 0.551 → 0.592 / 0.450 → 0.516 | 0.928 / 0.778 / 0.709 |
| 30B | natural-language | EVAL | 0.355 | 9.73% / 5.13% / 4.97% | 0.795 → 0.808 / 0.561 → 0.601 / 0.463 → 0.526 | 0.924 / 0.779 / 0.713 |
| 30B | coding | DEV | 0.372 | 7.14% / 4.23% / 4.49% | 0.757 → 0.769 / 0.533 → 0.572 / 0.441 → 0.506 | 0.899 / 0.755 / 0.688 |
| 30B | coding | EVAL | 0.373 | 6.57% / 4.39% / 4.57% | 0.762 → 0.773 / 0.530 → 0.566 / 0.429 → 0.496 | 0.903 / 0.756 / 0.689 |
| 30B | reasoning-math | DEV | 0.423 | 5.00% / 4.81% / 4.46% | 0.747 → 0.757 / 0.514 → 0.557 / 0.419 → 0.486 | 0.887 / 0.746 / 0.683 |
| 30B | reasoning-math | EVAL | 0.417 | 4.13% / 3.54% / 4.17% | 0.740 → 0.749 / 0.539 → 0.571 / 0.436 → 0.502 | 0.886 / 0.749 / 0.689 |
| 30B | structured-json | DEV | 0.452 | 4.20% / 3.53% / 4.76% | 0.730 → 0.741 / 0.506 → 0.545 / 0.380 → 0.465 | 0.873 / 0.733 / 0.668 |
| 30B | structured-json | EVAL | 0.456 | 4.74% / 3.83% / 4.31% | 0.717 → 0.732 / 0.483 → 0.524 / 0.380 → 0.453 | 0.874 / 0.728 / 0.661 |
| 30B | italian | DEV | 0.383 | 5.99% / 4.49% / 4.71% | 0.786 → 0.797 / 0.559 → 0.595 / 0.457 → 0.521 | 0.912 / 0.773 / 0.705 |
| 30B | italian | EVAL | 0.386 | 8.54% / 4.36% / 4.42% | 0.798 → 0.806 / 0.570 → 0.599 / 0.474 → 0.531 | 0.922 / 0.777 / 0.712 |
| 30B | transition | DEV | 0.554 | 5.92% / 3.73% / 3.98% | 0.760 → 0.770 / 0.524 → 0.560 / 0.427 → 0.492 | 0.900 / 0.751 / 0.684 |
| 30B | transition | EVAL | 0.498 | 5.87% / 3.72% / 4.38% | 0.772 → 0.782 / 0.538 → 0.571 / 0.423 → 0.497 | 0.906 / 0.759 / 0.691 |
| 30B | multi-turn-v2 | DEV | 0.369 | 6.95% / 4.15% / 4.50% | 0.802 → 0.812 / 0.579 → 0.616 / 0.487 → 0.548 | 0.920 / 0.781 / 0.716 |
| 30B | multi-turn-v2 | EVAL | 0.327 | 7.40% / 4.10% / 4.53% | 0.812 → 0.820 / 0.587 → 0.623 / 0.489 → 0.554 | 0.924 / 0.787 / 0.720 |
| 80B | natural-language | DEV | 0.261 | 3.78% / 1.18% / 1.87% | 0.752 → 0.754 / 0.623 → 0.634 / 0.568 → 0.589 | 0.884 / 0.781 / 0.733 |
| 80B | natural-language | EVAL | 0.258 | 3.51% / 1.49% / 2.28% | 0.770 → 0.767 / 0.634 → 0.648 / 0.579 → 0.601 | 0.885 / 0.788 / 0.746 |
| 80B | coding | DEV | 0.331 | 2.23% / 0.99% / 2.16% | 0.722 → 0.729 / 0.585 → 0.601 / 0.525 → 0.549 | 0.866 / 0.758 / 0.711 |
| 80B | coding | EVAL | 0.305 | 2.22% / 0.90% / 1.83% | 0.749 → 0.749 / 0.616 → 0.629 / 0.561 → 0.580 | 0.879 / 0.775 / 0.728 |
| 80B | reasoning-math | DEV | 0.338 | 1.78% / 1.81% / 2.36% | 0.678 → 0.680 / 0.525 → 0.546 / 0.470 → 0.498 | 0.835 / 0.722 / 0.672 |
| 80B | reasoning-math | EVAL | 0.324 | 1.82% / 1.34% / 2.16% | 0.663 → 0.673 / 0.521 → 0.539 / 0.467 → 0.491 | 0.841 / 0.718 / 0.666 |
| 80B | structured-json | DEV | 0.293 | 2.45% / 0.95% / 1.95% | 0.737 → 0.739 / 0.598 → 0.609 / 0.538 → 0.559 | 0.868 / 0.765 / 0.714 |
| 80B | structured-json | EVAL | 0.274 | 3.06% / 2.08% / 2.18% | 0.715 → 0.718 / 0.529 → 0.554 / 0.462 → 0.494 | 0.869 / 0.747 / 0.692 |
| 80B | italian | DEV | 0.238 | 3.84% / 1.44% / 2.81% | 0.837 → 0.838 / 0.724 → 0.733 / 0.662 → 0.680 | 0.922 / 0.847 / 0.806 |
| 80B | italian | EVAL | 0.225 | 3.90% / 1.17% / 2.32% | 0.841 → 0.839 / 0.739 → 0.746 / 0.684 → 0.698 | 0.924 / 0.850 / 0.813 |
| 80B | transition | DEV | 0.472 | 1.47% / 0.88% / 1.52% | 0.719 → 0.723 / 0.576 → 0.593 / 0.522 → 0.545 | 0.862 / 0.754 / 0.708 |
| 80B | transition | EVAL | 0.464 | 1.56% / 0.78% / 1.61% | 0.717 → 0.720 / 0.569 → 0.583 / 0.509 → 0.534 | 0.862 / 0.752 / 0.705 |
| 80B | multi-turn-v2 | DEV | 0.267 | 2.05% / 0.49% / 1.49% | 0.762 → 0.766 / 0.636 → 0.649 / 0.585 → 0.605 | 0.887 / 0.786 / 0.743 |
| 80B | multi-turn-v2 | EVAL | 0.226 | 3.85% / 0.96% / 1.49% | 0.787 → 0.790 / 0.662 → 0.673 / 0.613 → 0.630 | 0.900 / 0.807 / 0.765 |

**Fidelity qualification** ([SUPPORTED, EXP-021 method]):
- Every run used in a cell has `exp021_closure` = PASS. That means `workload_fidelity` = PASS, `incomplete_nodes_in_workload` = 0 and `tail_disagreements_unexplained` = 0.
- Where strict full-trace fidelity fails, it is only because of explained flush-sentinel termination-tail record loss. Under the established EXP-021 methodology such runs are valid for workload analysis.
- Exclusions and replacements are listed under Limitations.

### Earlier results (kept as recorded)

[MEASURED, pilot] 127 tokens x 48 layers x 8 selections. Normalised entropy per layer 0.58 (layer 47) to 0.81 (layer 14); Gini 0.69-0.90; EWS95 = 15% (layer 47) to 52% (layer 0) of the 128 experts; mean |S_t ∩ S_t-1| / k = 0.34 (layer 0) to 0.66 (layer 47). Global EWS95 = 2115 of 6144 experts (34%).

[MEASURED, pilot] llama.cpp computes only the output token in the LAST layer of a prompt step: a 77-token prompt touches 77 tokens' experts in layers 0-46 and one token's in layer 47.

[MEASURED, 30B Q4_K_M, coding dev, Windows reference machine] Nine instrumented runs, 4 / 2 / 1.5 GiB × 3: `raw/30b-coding-dev/run-0{0..8}-*`. The workload-only runtime expert-level hit rate (gate-node slice hits, prefill + decode) is 70.60 / 49.75 / 41.46%, with spread across repetitions ≤ 0.2 pp. The router trace is bit-identical across all nine runs and all budgets: the cache does not change what the model computes.

[MEASURED, 80B Qwen3-Next-80B-A3B Q3_K_M, coding dev, format v2, Windows reference machine, 2026-10-04] Nine runs (4 / 2 / 1.5 GiB × 3, interleaved): `raw/80b-coding-dev-v2/`. All have trace_complete = true and 0 budget violations. Workload-only gate-slice hit rate:
- 4 GiB: 65.97 / 65.91 / 65.92%
- 2 GiB: 52.49 / 52.47 / 52.40%
- 1.5 GiB: 46.84 / 46.84 / 46.85%

Runtime-style hit rate: 88.6 / 84.2 / 82.3%.

[SUPPORTED, EXP-021] On every run the replay reproduces the runtime decision by decision over the measured workload: `fidelity/80b-coding-dev-v2-regions/`. The original strict validation output, `fidelity/80b-coding-dev-v2/`, is kept. Two runs differ only in the flush-sentinel tail, explained by termination record loss (`tail-audit-*.json`). No rerun is needed.

[MEASURED, 30B, transition dev (`transition@5ba6e15e2b68`: natural language → coding → math → JSON, 4 × 300 tokens in one conversation), format v2, Windows] Nine runs, 4 / 2 / 1.5 GiB × 3: `raw/30b-transition-dev-v2/`.
- Workload expert-level hit rate (gate slices, prefill + decode): 73.7 / 50.6 / 41.9%.
- All nine runs pass replay closure with the workload verdict; four have flush-sentinel tail differences only (`fidelity/30b-transition-dev-v2/`).
- The workload routing is bit-identical across the nine runs.

[DERIVED, one trace, counterfactual] Temporal analysis of where the adaptive gain comes from: `analysis-v3/30b-transition-run-00-4G-temporal/`, write-up [docs/research/TRANSITION-ADAPTIVE-GAIN.md](../../docs/research/TRANSITION-ADAPTIVE-GAIN.md).
- Per-segment routing differs strongly: EWS90 is 0.27–0.43 per 299-token segment and 0.55 over the session; hot-set stability falls from 0.82 to 0.54.
- The adaptive gain is a better general layer allocation, not a reaction to transitions.

[DERIVED] Analysis of run-00-4G: `analysis-v3/run-00-4G/` (current: replay semantics v2, exact allocator, per-pool `store-replay` pins). `analysis-v2/run-00-4G/` is superseded only in its `store-replay` rows (see its `SUPERSEDED.md`). `analysis/run-00-4G/` is the original v1-legacy analysis, kept unchanged and marked [SUPERSEDED] in its `SUPERSEDED.md` (EXP-021). Its routing characterization is unchanged; its policy, heterogeneity and adaptive numbers are superseded.

## Interpretation

[DERIVED, 2026-10-11] **Routing concentration does not transfer across models as a measure of memory-value heterogeneity.**
- The 80B routes more concentratedly and more unevenly across layers than the 30B (C1, C2).
- Yet a static re-split of a fixed budget across its layers is worth 0.5–3.9% of misses (in-sample), against 3.5–10.4% on the 30B.
- Within each model there is an exploratory association at 4 GiB (C2). Across models, overall concentration does not determine what a layer's extra memory is worth; the shape of each layer's reuse curve at the budget's operating point matters too.

[DERIVED] **Most of the "layer-aware" benefit under pressure is partitioning, not adaptation.**
- At 2 and 1.5 GiB the fixed uniform per-layer partition already recovers most of the gap between global LRU and adaptive-e64 (C3).
- That suggests the main failure of a global LRU under pressure is cross-layer interference: the cyclic sweep of one token's experts through the cache (cf. U-005, EXP-022).
- A better choice of per-layer shares is a smaller second-order effect.
- The adaptive controller's remaining contribution is a better *general* split (C10), not temporal tracking.

[DERIVED] **Within the tested replay abstraction, the remaining headroom is substantially larger for expert retention/selection than for layer-budget reallocation.**
- The gap to the Belady-style reference (C11) is several times larger than any static or adaptive re-split gain.
- That is consistent with the pilot's reading, which is now measured on 28 cells rather than 127 tokens.
- Whether an online policy can realise a material part of that headroom is [OPEN]. The reference uses future knowledge, and it is not a proven bound for multi-key events and mixed sizes.
- It is still not an answer to RQ-001, because no out-of-sample or runtime test exists.

[SUPERSEDED, 2026-10-11, by the full campaign] "Layers are not interchangeable: the last layer is the most concentrated and the most temporally local in this trace, the first the least. Whether this holds beyond one prompt is unknown." (pilot, 127 tokens). On 28 cells:
- [SUPPORTED] The first layer is the least concentrated: layer 0 on the 30B, layer 0 or 2 on the 80B.
- [FALSIFIED as a general statement] The last layer is *not* generally the most concentrated. The minimum per-layer EWS90 is at layers 7–47 on the 30B (most often 30–31 or 18) and at layers 4–47 on the 80B (most often 24 or 12). It is layer 47 in only two cells (30B transition DEV, 80B italian EVAL).
- The pilot's "most temporally local" part was not re-checked layer by layer here.

## Limitations

[LIMITATION, 2026-10-04] **Generated-text provenance** ([TEXT-PROVENANCE](../../docs/research/TEXT-PROVENANCE.md)).
- **30B runs recorded before the fix are provenance-limited:** their text hashes are SHA-256(""). The harness read `message.content`, while Qwen3-30B-A3B's thinking-mode tokens were returned in `reasoning_content`. The 80B runs carry content-only hashes.
- **Routing and cache data are unaffected.**
- **Scope note:** on the 30B the generated tokens are reasoning-mode text about each task, and multi-turn histories contained empty assistant turns.

[LIMITATION, 2026-10-10] **Multi-turn history semantics.**
- **v1 `multi-turn` is superseded for the final comparison.** It appended `content` as returned, possibly empty. No EXP-007 multi-turn run was recorded under it.
- **The final campaign uses `multi-turn-v2`.** v1 results, if any, are never re-labelled as v2.
- **`30b-transition-dev-v2` keeps v1 history semantics**, with empty assistant turns on the 30B.

[LIMITATION] 127 tokens is far too few: plug-in entropy and EWS are biased by small samples (1016 selections over 128 experts per layer). One prompt, mmap-era binary, expert sizes uniform (derived, not read from the GGUF). Development use only; never cited as evidence.

[LIMITATION, 2026-10-11] **Final campaign.**
- **In-sample only.** Every `analysis-v3` analysis was trained in-sample (`analysis_manifest.json → train`). No DEV-trained, EVAL-tested static or adaptive allocation was run, so every "optimal static" and uniform-excess figure is an upper bound (C12). DEV/EVAL agreement (C5) is pattern replication, not out-of-sample validation.
- **One analysed trace per cell.** Policy replays use run-00-4G. Routing identity across the nine repetitions was established for 30B coding DEV and 30B transition DEV. It was not re-checked for the other cells; the analysis assumes it from deterministic greedy decoding.
- **Trace length and small samples.** Single-turn cells have only 320–389 decode tokens, transition ≈1.2k and multi-turn 3.1–4.9k. EWS fractions are not length-normalised, and cross-workload comparisons of concentration are partly confounded with length. No bootstrap confidence intervals over tokens are reported for the per-layer metrics.
- **Counterfactual replay** is expert-unit and event-atomic. Its hit rates are comparable with each other, not with the measured runtime (EXP-021). Stall times are modelled.
- **Exclusions and replacements.** Four runs failed EXP-021 closure with `reconstruct_inconsistencies` ≠ 0:
  - `30b-structured-json-dev-v2/run-07-4G`;
  - `80b-reasoning-math-eval-v2/run-06-1.5G`;
  - `80b-structured-json-dev-v2/run-06-1.5G`;
  - `80b-structured-json-eval-v2/run-02-1.5G`.

  They were kept, not deleted, and each was replaced by one run in a `*-replacement-*` folder. `30b-multi-turn-v2-dev/run-08-2G` has no manifest (incomplete run) and was replaced by `30b-multi-turn-v2-dev-replacement-2G`. Every cell-budget has 3 qualified runs.
- **Trace-completeness flag on 30B multi-turn-v2 (explained).** All 18 cell runs of 30B multi-turn-v2, plus the 1536 smoke, carry `trace_completeness.complete` = false.
  - [DERIVED] In all 76 checked requests the decode-step count equals `predicted_n − 1` exactly. The number of "partial steps" equals the number of non-final 512-token prefill micro-batches, `(prompt_n − 1) // 512`.
  - In those steps layer 47 has no routing record, because llama.cpp computes only the output row in the last layer and a non-final micro-batch has none (cf. U-004). This was verified on `30b-multi-turn-v2-eval/run-00-4G`: the two partial steps are `prefill`, `n = 512`, missing only layer 47.
  - The checker counts these as missing data, so this is a checker false positive, not record loss. The decode-phase analyses and the EXP-021 verdicts (`incomplete_nodes_in_workload` = 0) are unaffected. The checker itself is unchanged; see the open item in the 2026-10-11 report.
- **Not publication-quality by the repository's own fingerprint rule.** All 256 run manifests list `argine2 commit` under `missing`: the Argine2 tree was not a git checkout when the runs were made. The model, store, binary and workload SHA-256 values are recorded. The scientific validity of the runs rests on the EXP-021 qualification above, not on the publication flag.
- **multi-turn-v2 file edited in place.** `workloads/v2/multi-turn.json` was changed from `n_predict` 1024 (`ff89429e96c0`) to 1536 (`e978bde0341f`) after the 1024 smoke run, instead of being added as a new version. Only `raw/30b-multi-turn-v2-smoke-4G` used the 1024 file, and its manifest records that hash. Every cell run records `e978bde0341f`.
  - [Update 2026-10-11] The 1024 file has been reconstructed byte-exactly and preserved, not as canonical, at `workloads/superseded/v2-multi-turn-n1024-ff89429e96c0.json`. Its SHA-256 equals the manifest's `ff89429e96c0e8db…aacfe93`.
  - The in-place edit changed only the nine `n_predict` values (1024 → 1536) and the line endings (LF → CRLF).
- **One machine, CPU backend, two models.** No claim generalises beyond them.

## Status history

2026-10-03 RUNNING (pilot done, instrumented runs pending on the reference machine). 2026-10-03 30B coding-dev runs recorded (9); analysis regenerated after EXP-021 into `analysis-v2/`. 2026-10-04: 80B coding-dev runs recorded (9, format v2), replay-validated over the workload; other workloads and the eval split are still pending, so the experiment stays RUNNING. 2026-10-04: 30B transition dev recorded (9 runs, fidelity-valid) and analysed temporally; status unchanged (closure needs both models, every workload, dev and eval). 2026-10-10: multi-turn workload versioned to `multi-turn-v2` (explicit history semantics, eligibility check); status unchanged. 2026-10-04 → 2026-10-11: the remaining cells recorded on the reference machine (both models, seven workload families, DEV and EVAL), fidelity-validated per campaign, excluded runs replaced, and every cell analysed (`analysis-v3/`, all in-sample); multi-turn-v2 frozen at `n_predict` 1536 after the 1024 smoke truncated every turn. **2026-10-11: SUPPORTED. Experimental collection and analysis are CLOSED: 28 / 28 cells complete and EXP-021-qualified.** Follow-up questions move to EXP-008 (EWS as predictor), EXP-010 (adaptive allocation, out-of-sample), EXP-022 (extreme pressure) and EXP-023 (workload drift).
