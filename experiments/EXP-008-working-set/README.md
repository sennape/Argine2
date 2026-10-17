# EXP-008 — Effective Working Set (EWS) as a predictor

| field | value |
|---|---|
| ID | `EXP-008-working-set` |
| Status | **INCONCLUSIVE** |
| Period | 2026-10-11 |
| Origin | Argine2 |
| Related questions | RQ-002, RQ-003 |
| Supersedes / superseded | — |

## Research question

Does EWS / total experts predict throughput better than resident experts / total experts?

## Hypothesis

[HYPOTHESIS] Routing concentration and EWS predict memory-tiered behaviour more accurately than total parameters or nominal residency (requirement section 2).

## Setup

Use EXP-007 traces of both models at several budgets. For each (model, workload, budget): measured tok/s and expert-level hit rate vs (a) resident fraction, (b) budget / EWS95 bytes, (c) budget / Denning working set ws_tau. Compare rank correlation and fit error; attempt to falsify by finding pairs where EWS ordering contradicts speed ordering.

### Analysis plan (frozen before predictor-comparison statistics were computed)

Frozen 2026-10-11 11:48 UTC. Implemented by `tools/exp008_analysis.py`, SHA-256 at freeze `ef40da64b363c858e7691bc4901b68332ff7ff2e759a6563670655b61d34162d`.

**Not a preregistration.** The EXP-007 dataset already exists, and some descriptive values have already been inspected: per-cell EWS fractions, tok/s and hit rates, and a trace-length check during the 2026-10-11 audit. No correlation, fit error or pair count between any predictor below and either target had been computed when this plan was written.

**Question (unchanged).** Does EWS / total experts predict **throughput** better than resident experts / total experts?
- **Primary target:** generation throughput (tok/s).
- **Secondary, mechanistic target:** expert-level hit rate.
- The secondary result never replaces or overrides the primary status.

**Predictors.** Only those named in the original setup:
- **P0, resident fraction:** budget / total expert bytes. Total expert bytes = sum over layers of the per-layer expert size × number of experts, from the trace metadata.
- **P1:** budget / EWS95 bytes.
- **P2:** budget / Denning working set in bytes, where the Denning working set = sum over layers of the repository's `ws_tau` (mean number of distinct experts in a window of τ = 16 decode tokens, the window reported by `research characterize`) × the per-layer expert size.

EWS90, EWS99 and anything from EXP-007's concentration-vs-opportunity table are excluded from the primary analysis.

**Trace-length normalisation.** P1 and P2 are computed on the **first 320 decode tokens** of the workload. Warm-up and flush sentinel are excluded; prefill is not counted. They come from the run-00-4G routing trace of each model × workload × split; routing is identical across budgets. Full-trace P1 and P2 are reported as exploratory only.

**Observations.**
- **Unit:** model × workload × budget, with budgets 4 / 2 / 1.5 GiB.
- **Primary:** the **EVAL** split (repository methodology: claims on EVAL). That gives 2 models × 7 workloads × 3 budgets = 42 rows, 21 per model.
- **DEV:** the same 42 rows are analysed identically as a **replication**. They do not decide the status.
- **Target value per row:** the median over the qualified EXP-007 runs (EXP-021 closure PASS; 3 expected) of:
  - generation tok/s per run = sum(`predicted_n`) / sum(`predicted_ms`) over the run's workload responses, the formula `research trace` uses for `runs.csv`, recomputed from `responses.jsonl` for every run and cross-checked against `runs.csv` where it exists. This also reconstructs the 8 30B multi-turn DEV runs without `runs.csv`.
  - expert-level hit rate from the run manifest (`runtime-hit-rates.csv`).
- Rows with fewer than 3 qualified runs are reported.

**Model stratification.** All primary statistics are computed separately for the 30B and the 80B. A pooled two-model Spearman ρ is reported as exploratory and descriptive only.

**Statistics (per model, per target, per predictor):**
1. **Rank correlation.** Spearman ρ over the 21 rows, with average ranks for ties; ρ, n and the number of distinct predictor values are reported. P0 takes only 3 values per model (one per budget): it cannot distinguish workloads at a fixed budget. Secondary analysis: the fixed-budget, workload-only ρ of P1 and P2 (n = 7 per budget). P0 is constant there and has no ρ.
2. **Fit error.** Leave-one-workload-out within each model.
   - The single mapping is log(target) = a + b·log(predictor), ordinary least squares, the log-log form of the repository's speed/residency exponent.
   - It is fitted on the other 6 workloads (18 rows) and predicts the held-out workload's 3 budgets, for all 7 workloads.
   - Reported: MAE in target units and relative MAE (MAE / mean target).
   - The same mapping is used for every predictor and both targets.
3. **Falsification pairs.**
   - At fixed model + budget, every workload pair: 21 per budget, 63 per model.
   - A predictor's ordering is **concordant** if it matches the target's, **discordant** if opposite, **tie** if either is equal.
   - Accuracy = (concordant + 0.5 × ties) / n.
   - P0 is tied on every pair by construction, so its accuracy is exactly 0.5.

**Comparison of X ∈ {P1, P2} with P0, per model and target:**
- **Spearman:** X better if |ρ_X| > |ρ_P0|; P0 better if |ρ_P0| > |ρ_X|; otherwise tie.
- **Fit:** X better if MAE_X < MAE_P0; P0 better if the reverse; otherwise tie.
- **Pairs:** X better if accuracy_X > 0.5; P0 better if < 0.5; otherwise tie.

**Decision rule** (directional, no effect-size threshold), applied to each target separately:
- **SUPPORTED** if some X ∈ {P1, P2} is better than P0 in **both** models in at least two of the three families, and P0 is not strictly better in the remaining family, in either model.
- **FALSIFIED** if, in **both** models and for **both** P1 and P2, P0 is better or tied in at least two of the three families.
- **INCONCLUSIVE** otherwise.
- The experiment's registry status is the **primary (throughput)** verdict.

**Required caveats on throughput,** stated with the verdict whatever it is:
- tracing overhead is unknown (EXP-024 not run);
- workloads were recorded in different sessions, and EXP-018 measured 15–19% between-session variation;
- context lengths differ between workloads;
- all EXP-007 runs used 4 generation threads, while EXP-027 later found 12 best for the 30B.

A SUPPORTED throughput verdict is therefore phrased "SUPPORTED within the traced EXP-007 dataset", not as a general runtime-predictor claim.

**Uncertainty (descriptive only).** Workload-level bootstrap: workloads are resampled with replacement within a model, each contributing its 3 budget rows; 2000 resamples, seed 20261011. Reported: percentile intervals of ρ for P0, P1 and P2 and of |ρ_X| − |ρ_P0|. No significance test is used for closure.

**Outputs** (`processed/`): `predictor_table.csv` (every row, predictor values, both targets, per-run values and provenance), `correlations.csv`, `fit_errors.csv`, `falsification_pairs.csv`, `bootstrap.csv`, `summary.json`.

## Commands

```bat
argine research trace ... (EXP-007)
argine research characterize TRACE --out DIR
argine research predict runs.csv
```

## Raw data

No new raw data: the analysis reads existing EXP-007 files only. Those are the run-00-4G routing traces, the qualified runs' `responses.jsonl` and `runtime-hit-rates.csv`, all under `../EXP-007-router-characterization/`.
- Outputs, deterministic: `python tools/exp008_analysis.py` (19 s) writes `processed/`:
  - `predictor_table.csv`: 84 rows (42 EVAL primary, 42 DEV replication), each with the predictor values, both targets, per-run values and the source run directories;
  - `correlations.csv`, `fit_errors.csv`, `falsification_pairs.csv`, `bootstrap.csv`, `summary.json`.
- The tool's SHA-256 is the one recorded at freeze time (`ef40da64…`).
- `results.csv` remains the pilot per-layer EWS table (legacy trace), kept as history.

## Result

[MEASURED inputs, DERIVED statistics; frozen plan applied unchanged; EVAL split primary]

**Observations.**
- 42 primary rows (2 models × 7 workloads × 3 budgets) and 42 DEV replication rows.
- Every row has 3 qualified runs.
- **Throughput reconstruction:** tok/s was recomputed from `responses.jsonl` for all 252 runs. It agrees with the campaigns' `runs.csv` in all 244 runs that have one (maximum difference 0.00005 tok/s). The 8 30B multi-turn DEV runs without `runs.csv` use the reconstructed value.
- **Length normalisation:** every primary (EVAL) row used exactly 320 decode tokens.
- **Deviation from the plan, recorded and not corrected afterwards:** one replication cell, 80B structured-json DEV, has only **312** workload decode tokens, so it used 312. The plan's "320 is the minimum" came from the audit's per-report token counts. The affected cell is in the replication set, not the primary.

**Primary target: throughput (generation tok/s), EVAL.**

| model | predictor | Spearman ρ (n = 21) | LOWO MAE (relative) | pair accuracy (concordant / discordant / ties of 63) | fixed-budget ρ (4 / 2 / 1.5 GiB, n = 7) |
|---|---|---|---|---|---|
| 30B | P0 resident fraction | **0.848** (3 distinct values) | **0.428 tok/s (9.4%)** | 0.500 (0 / 0 / 63) | — (constant) |
| 30B | P1 budget / EWS95 | 0.665 | 0.653 (14.4%) | 0.397 (25 / 38 / 0) | −0.32 / −0.36 / −0.36 |
| 30B | P2 budget / Denning ws16 | 0.725 | 0.518 (11.4%) | 0.492 (31 / 32 / 0) | −0.25 / −0.29 / −0.29 |
| 80B | P0 resident fraction | 0.684 (3 distinct values) | 0.432 (8.5%) | 0.500 (0 / 0 / 63) | — (constant) |
| 80B | P1 budget / EWS95 | 0.727 | 0.471 (9.2%) | 0.635 (40 / 23 / 0) | 0.29 / 0.29 / 0.46 |
| 80B | P2 budget / Denning ws16 | **0.795** | **0.354 (6.9%)** | **0.730 (46 / 17 / 0)** | 0.46 / 0.46 / 0.57 |

Comparison with P0 (spearman / fit / pairs):
- **30B:** P0 is better than both P1 and P2 in **all three** families.
- **80B:** P2 is better than P0 in all three families. P1 is better in two (ρ, pairs) and worse in fit.

The models disagree, so the frozen rule returns **INCONCLUSIVE** (neither the SUPPORTED nor the FALSIFIED condition holds in both models). **DEV replication:** the same pattern and the same verdict, INCONCLUSIVE.

**Secondary target: expert-level hit rate, EVAL.**

| model | predictor | Spearman ρ | LOWO MAE (relative) | pair accuracy (c / d / t) | fixed-budget ρ (4 / 2 / 1.5 GiB) |
|---|---|---|---|---|---|
| 30B | P0 | 0.934 | 0.0363 (6.5%) | 0.500 (0 / 0 / 63) | — |
| 30B | P1 | 0.973 | 0.0221 (4.0%) | 0.841 (53 / 10 / 0) | 0.89 / 0.82 / 0.82 |
| 30B | P2 | **0.975** | **0.0216 (3.9%)** | 0.841 (53 / 10 / 0) | 0.89 / 0.75 / 0.75 |
| 80B | P0 | 0.732 | 0.0673 (11.4%) | 0.500 (0 / 0 / 63) | — |
| 80B | P1 | 0.887 | 0.0511 (8.7%) | 0.794 (50 / 13 / 0) | 0.89 / 0.71 / 0.71 |
| 80B | P2 | **0.912** | **0.0499 (8.5%)** | **0.889 (56 / 7 / 0)** | 0.96 / 0.86 / 0.86 |

Both P1 and P2 beat P0 in **all three** families in **both** models, so the frozen rule returns **SUPPORTED**. DEV replication: SUPPORTED, again all three families in both models.

[DERIVED, descriptive uncertainty] Workload-level bootstrap (2000 resamples) of |ρ_X| − |ρ_P0|, as 2.5–97.5% percentiles:

| | P1 | P2 |
|---|---|---|
| 30B throughput | −0.30 … +0.03 | −0.17 … +0.05 |
| 80B throughput | −0.16 … +0.18 | −0.12 … +0.26 |
| 30B hit rate | −0.04 … +0.08 | −0.04 … +0.07 |
| 80B hit rate | −0.00 … +0.21 | +0.05 … +0.25 |

These are uncertainty estimates, not tests.

[DERIVED, EXPLORATORY: not part of the decision]
- **Full-trace (unnormalised) P1/P2:** the throughput verdict is still INCONCLUSIVE. On the 30B, full-trace P1 beats P0 in fit and pairs but not ρ. The hit-rate verdict is still SUPPORTED.
- **Pooled across the two models** (EVAL, n = 42), Spearman ρ:

  | target | P0 | P1 | P2 |
  |---|---|---|---|
  | throughput | 0.25 | 0.47 | 0.82 |
  | hit rate | 0.49 | 0.76 | 0.92 |

  This is descriptive only, given the model confounding (2 models).


## Interpretation

**Status: INCONCLUSIVE.** The primary throughput result decides it, under the frozen directional rule:
- On the **80B**, the Denning-normalised budget (P2) predicts throughput better than resident fraction on all three measures.
- On the **30B**, resident fraction is better on all three. At a fixed budget the working-set predictors order workloads *opposite* to their throughput (ρ −0.25 … −0.36).
- The models disagree, so the throughput question is not settled by the traced EXP-007 data.

**Secondary (hit rate): SUPPORTED within the traced EXP-007 dataset.** Budget normalised by the length-matched working set (EWS95 or Denning ws16) predicts the runtime's expert-level hit rate better than resident fraction in both models, by every pre-specified measure, and DEV replicates it. This does **not** override the primary status.

[DERIVED, exploratory reading, not tested] On the 30B, workload-to-workload throughput differences do not follow hit rate. Multi-turn has the highest hit rate (0.798 at 4 GiB) but the lowest throughput (4.66 tok/s), and transition (5.27 tok/s) behaves the same way. That fits a factor outside expert memory dominating between-workload throughput on the 30B, consistent with the listed caveats (longer contexts, different sessions). It is not established here.

**Throughput caveats** (part of the result, whatever the verdict):
- tracing overhead is unknown (EXP-024 not run);
- each workload was recorded in a different session, and EXP-018 observed ≈15–19% cross-run differences among legacy records with incomplete runtime provenance (not a pure session noise floor; narrowed by the 2026-10-11 EXP-018 audit; the frozen plan's caveat above is kept verbatim);
- context lengths differ between workloads (multi-turn and transition decode at longer contexts);
- all EXP-007 runs used 4 generation threads, while EXP-027 later found 12 best for the 30B.

A clean throughput test would need untraced throughput, matched context, and workloads interleaved within sessions. It is not run here and not required to record this result: EXP-008 stays INCONCLUSIVE on throughput until such evidence exists.

## Limitations

Needs at least 2 models x 3 budgets x 2 workloads to say anything; 30B vs 80B alone is two points.

- [LIMITATION, 2026-10-11] **Not a preregistration:** the plan was frozen after the dataset existed and descriptive values had been seen, but before any predictor-comparison statistic was computed.
- [LIMITATION] **P0 cannot separate workloads:** within a model and budget it is constant. So the predictors can only be told apart *between workloads at a fixed budget* (falsification pairs, fixed-budget ρ), while the 21-row ρ is dominated by the budget, which all predictors follow.
- [LIMITATION] **Model confounding:** two models only, differing in quantisation, expert count, slice size and expert bytes. Primary results are per model; the pooled ρ is descriptive.
- [LIMITATION] **One trace per cell for the predictors:** run-00-4G, assuming identical routing across repetitions. Predictors come from the first 320 decode tokens; the full-trace versions are exploratory.
- [LIMITATION] **One replication cell at 312 tokens:** 80B structured-json DEV (see Result).
- [LIMITATION] The throughput caveats listed under Interpretation.

## Status history

2026-10-03 PLANNED. 2026-10-11: analysis plan frozen in this README before any predictor-comparison statistic was computed (analysis over existing EXP-007 data only). **2026-10-11 INCONCLUSIVE.** Primary throughput: the models disagree (80B favours P2, 30B favours P0). Secondary hit rate: SUPPORTED within the traced EXP-007 dataset. DEV replicates both. No new inference was run.
