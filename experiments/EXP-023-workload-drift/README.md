# EXP-023 — Routing under workload transitions

| field | value |
|---|---|
| ID | `EXP-023-workload-drift` |
| Status | **FALSIFIED** |
| Period | 2026-10-12 |
| Origin | Argine2 |
| Related questions | RQ-004, RQ-006 |
| Supersedes / superseded | — |

> **Closure note (2026-10-12): CLOSED, FALSIFIED for the current controller.**
> - **Retrospective, analysis-only.** All outputs were derived from existing EXP-007 evidence by a single `tools/exp023_analyze.py --analyze` run under the active [`analysis_freeze.json`](analysis_freeze.json). No new model run, trace, benchmark or llama-server session was performed for EXP-023.
> - **Verdict.** Set by the deterministic logic below (FH-029), independently of the retrospective JSD characterisation. See [Result](#result).
> - **Not "workload dependence falsified".** The retrospective JSD characterisation is positive descriptive evidence of workload-dependent routing (RQ-004). What is falsified is the clause that the current adaptive controller shortens the transient.

> **Preparation note (2026-10-12): retrospective, analysis-only closure.** *[Historical, kept as written before the synthesis was run.]*
> - **Data already exist.** The four transition sessions the question needs were recorded and qualified in EXP-007: 30B DEV, 30B EVAL, 80B DEV and 80B EVAL. No new model trace will be collected for EXP-023, and the old trace commands below are historical.
> - **Plan.** Closure is a deterministic synthesis of that existing evidence (`tools/exp023_analyze.py`, specification in [`analysis_freeze.json`](analysis_freeze.json)).
> - **Not a confirmatory test.** The JSD operationalisation was specified after the data existed. FH-028 and FH-029 were already known.
> - ~~**Verdict not yet recorded.** The final closure awaits the synthesis outputs and their independent review.~~ [Superseded 2026-10-12: synthesis run and reviewed; closed, see the closure note above.]

## Research question

How do working sets, entropy and cache behaviour change when the workload switches (natural language → code → math → JSON), and how fast does a policy adapt?

## Hypothesis

[HYPOTHESIS] Transitions change the per-layer expert distribution measurably (JSD > within-segment noise) and cause a transient miss excess that adaptive allocation shortens.

*(Historical, 2026-10-03, kept verbatim. The hypothesis is conjunctive. Its clause "adaptive allocation shortens the transient" is already FALSIFIED for the current controller: [FH-029](../../FAILED_HYPOTHESES.md), established on the 30B DEV trace and replicated on 30B EVAL, 80B DEV and 80B EVAL.)*

## Setup

`workloads/v1/transition.json` (4 x 300 tokens, one conversation), both models; adaptation metrics from `replay.adaptation_metrics`.

*Closure setup (2026-10-12, retrospective synthesis):*

The four fixed sources were resolved from each temporal directory's `replication.csv`. Each is the qualified `run-00-4G` (EXP-021 closure PASS) in EXP-007's `runtime-hit-rates.csv`.

| key | router trace (EXP-007 `raw/`) | temporal analysis (EXP-007 `analysis-v3/`) |
|---|---|---|
| 30b-dev | `30b-transition-dev-v2/run-00-4G` | `30b-transition-run-00-4G-temporal` |
| 30b-eval | `30b-transition-eval-v2/run-00-4G` | `30b-transition-eval-run-00-4G-temporal` |
| 80b-dev | `80b-transition-dev-v2/run-00-4G` | `80b-transition-dev-run-00-4G-temporal` |
| 80b-eval | `80b-transition-eval-v2/run-00-4G` | `80b-transition-eval-run-00-4G-temporal` |

`--check` verifies, for each source:
- every required file is present;
- the routing identity recomputed from the trace equals the hash recorded in `replication.csv`;
- the counted decode segments and boundaries equal those of the temporal analysis;
- the model, split and workload in the run manifest are as expected;
- the run is qualified in EXP-007;
- budgets are 4G / 2G / 1.5G, with 3 boundaries and 4 segments;
- the required `adaptation_v2.csv` columns are present.

It writes nothing.

**A. Routing-distribution drift** [DERIVED, retrospective operationalisation]
- **Data:** decode phase, counted steps only (warm-up and flush sentinel excluded). Per MoE layer, the distribution of routed expert selections.
- **Boundary JSD:** the last 64 counted steps of the old segment against the first 64 of the new segment, Jensen-Shannon divergence base 2. Zero-probability terms contribute 0; no smoothing.
- **Within-segment reference:** each segment is cut into consecutive, non-overlapping, complete 64-step windows from its start, and JSD is taken between adjacent windows. A boundary's reference is the within pairs of its old and new segment; a pooled reference over all four segments is reported too.
- **Window choice:** 64 matches the existing post-boundary 64-step windows of the temporal analysis. It was not fitted to the JSD results.
- **Outputs:** effect sizes and descriptive summaries only, with no p-value and no success threshold. This part does not set the verdict.

**B. Cross-trace transient synthesis.**
- **Policies:** `adaptive-e64` vs `ablate-frozen`, at 4G / 2G / 1.5G and boundaries 1–3, from the existing `adaptation_v2.csv`.
- **Per row:** signed excess16 / 32 / 64, never converted to positive; the positive-part excess64 as a separate field; the noise reference and significance flag; half-life; and the adaptive allocation convergence steps.
- **From `summary.json`:** controller gain, fixed mean-allocation gain and its share, segment-switching oracle potential, and the reaction-lag cost.
- **Not used:** the historical `adaptation_steps`, because of its floor at 16 (FH-028).
- **Wording:** "boundary transient / departure from the new-segment steady state", always with its sign.

**Verdict logic (deterministic; fixed before the synthesis was run).**
- **Basis.** The hypothesis is conjunctive. Its clause "adaptive allocation shortens the transient" is already FALSIFIED for the current controller by FH-029 (30B DEV), replicated on 30B EVAL, 80B DEV and 80B EVAL.
- **Verdict.** The global verdict is therefore **FALSIFIED for the current controller**, independently of the retrospective JSD characterisation. It takes no data input, so no numeric value (1.0, 1.13, 0.0 or any other) acts as an acceptance criterion.
- **Consistency check.** The documented FH-029 facts are re-checked in the fixed sources as reproducibility evidence only:
  - the mean-allocation gain against the controller's own gain;
  - the sign of the reaction-lag cost.

  The actual numbers are reported.
- **If the sources contradict FH-029.** That is an evidence/provenance inconsistency error to investigate. It never turns the verdict into UNKNOWN, INCONCLUSIVE or anything else.
- **Rounding note.** The observed minimum `A_share_of_gain_mean_alloc` is 1.129 (112.9%, 30B EVAL at 1.5G). FH-029's historical "≥ 113%" is true only after rounding to an integer percentage; this is flagged for a separate documentary correction, and FH-029 is not modified here.
- **What the verdict would not mean:** that workloads are stationary, that workload dependence is falsified, that layer-aware allocation is useless, or that no faster future controller could exploit transitions.

## Commands

```bat
python tools\exp023_analyze.py --check     :: validate the four fixed sources; writes nothing
python tools\exp023_analyze.py --analyze   :: wrote processed/ once, 2026-10-12 (closure run; not to be rerun)
python -m unittest discover -s tests -p "test_exp023.py" -v
```

*Historical commands (2026-10-03), superseded. They are not to be run for closure, because EXP-007 already recorded these sessions:*

```bat
argine research trace MODEL --expert-store --budgets 2G --workload transition --split dev --out experiments/EXP-023-workload-drift/raw
argine research analyze TRACE --budgets 4G,2G,1.5G --out experiments/EXP-023-workload-drift/processed
```

## Raw data

None of its own: EXP-023 records no new trace. Its inputs are the four EXP-007 sessions above, read-only. Their file paths and SHA-256 are frozen in `analysis_freeze.json` and are copied to `processed/source_manifest.json`. No EXP-007 file was modified.

Outputs in `processed/` (written by the single 2026-10-12 `--analyze` run, unmodified since):
- `source_manifest.json`;
- `jsd_by_layer.csv`;
- `jsd_summary.csv`;
- `transient_summary.csv` (36 rows);
- `controller_summary.csv` (12 rows);
- `summary.json`.

`results.csv` at the experiment root is a byte-identical copy of `processed/transient_summary.csv` (the 36 adaptive-vs-frozen comparisons), present because the experiment index requires a `results.csv` for every closed experiment.

## Result

**[FALSIFIED for the current controller]** The hypothesis is conjunctive. Its clause "adaptive allocation shortens the transient" is already falsified for the current (untuned `adaptive-e64`) controller by [FH-029](../../FAILED_HYPOTHESES.md), established on 30B DEV and replicated on 30B EVAL, 80B DEV and 80B EVAL. EXP-023 is therefore globally **FALSIFIED for the current controller**. This verdict is independent of the retrospective JSD result (A), which does not set it.

Source of every number below: `processed/` (single `--analyze` run, 2026-10-12, existing EXP-007 evidence only; no new model run).

**A. Routing-distribution drift** [DERIVED, retrospective operationalisation]

Descriptive and retrospective, not confirmatory: no p-value and no success threshold. Boundary JSD = last 64 counted decode steps of the old segment vs first 64 of the new one, per MoE layer; within reference = adjacent non-overlapping 64-step windows of the old and new segment (`jsd_summary.csv`, `all` rows).

| trace | median boundary JSD | median within reference | median boundary / within ratio | share of layer-boundary rows with boundary JSD > within max |
|---|---|---|---|---|
| 30B DEV | 0.47390358 | 0.10914244 | 3.810259 | 0.986111 |
| 30B EVAL | 0.49591813 | 0.11956621 | 4.021954 | 1.0 |
| 80B DEV | 0.74589325 | 0.22459795 | 3.381423 | 1.0 |
| 80B EVAL | 0.75238965 | 0.26082351 | 2.839054 | 0.965278 |
| 30B (model-pooled) | 0.48434069 | 0.11331654 | 3.926913 | 0.993056 |
| 80B (model-pooled) | 0.74977737 | 0.24022545 | 3.037708 | 0.982639 |

- Workload boundaries are associated with large and consistent changes in per-layer routing distributions across all four analysed traces (both models, both splits).
- This supports workload-dependent routing as a descriptive finding.
- Because the operationalisation was specified after the data existed, it is **not** a confirmatory validation of the historical "JSD > within-segment noise" clause.

**B. Cross-trace transient synthesis** [DERIVED]

Boundary transient / signed departure from the new-segment steady state, `adaptive-e64` vs `ablate-frozen`, 36 trace × budget × boundary comparisons (`transient_summary.csv`, also `results.csv`):
- adaptive signed `excess64` lower than frozen: **11 / 36**; higher: **25 / 36**; ties: 0 / 36.
- 42 of the 72 signed `excess64` values (frozen and adaptive together) are negative. A negative value is a departure *below* the new-segment steady state; it is reported with its sign and is not "zero" or "no transient".

| grouping | adaptive signed excess64 lower | adaptive signed excess64 higher |
|---|---|---|
| 30B DEV | 3 / 9 | 6 / 9 |
| 30B EVAL | 3 / 9 | 6 / 9 |
| 80B DEV | 2 / 9 | 7 / 9 |
| 80B EVAL | 3 / 9 | 6 / 9 |
| 4G | 3 / 12 | 9 / 12 |
| 2G | 1 / 12 | 11 / 12 |
| 1.5G | 7 / 12 | 5 / 12 |
| natural-language → coding | 2 / 12 | 10 / 12 |
| coding → reasoning-math | 3 / 12 | 9 / 12 |
| reasoning-math → structured-json | 6 / 12 | 6 / 12 |

**C. FH-029 consistency** [DERIVED, reproducibility evidence only; does not set the verdict]

All 12 trace × budget rows (`controller_summary.csv`) reproduce the historical FH-029 facts:
- the fixed mean-allocation gain exceeds the controller's own gain in **12 / 12** rows;
- the reaction-lag cost against instantly switched segment allocations is negative in **12 / 12** rows.

`A_share_of_gain_mean_alloc` ranges from 1.129 (30B EVAL, 1.5G) to 2.396 (80B DEV, 4G); the reaction-lag cost ranges from −1844 to −820 misses.

*Rounding note:* 1.129 = 112.9%. FH-029's historical "≥ 113%" wording holds only after rounding to an integer percentage. FH-029 is not modified here; this is flagged for a separate documentary correction.

*Historical early evidence (2026-10-04), kept as recorded.* [DERIVED, early evidence on the pre-registered hypothesis, from the EXP-007 30B transition DEV trace, one session, counterfactual replay, 2026-10-04] Write-up: [docs/research/TRANSITION-ADAPTIVE-GAIN.md](../../docs/research/TRANSITION-ADAPTIVE-GAIN.md).
- **Transitions change the routing measurably:** EWS90 is 0.27–0.43 per segment and 0.55 over the session; segment-optimal layer splits differ by 5–15% of the budget.
- **Every policy, static ones included, shows a transient miss excess after a boundary.** For example, NL → code at 4 GiB: +1,465 misses in 64 steps for the frozen split.
  - *[Scope note, 2026-10-12: this describes the 30B DEV trace only. In the later traces some signed `excess64` values are negative. The synthesis reports the signed departure from the new-segment steady state.]*
- **The adaptive controller does not shorten it** (+1,539). [FALSIFIED for this controller on this trace]. Hot-set pinning lengthens it (+2,196 / +2,340).

## Interpretation

**The workload-dependent routing signal is strong, but the current adaptive controller does not exploit it quickly enough.** Its gain over the frozen split is a better general layer split, not a reaction to transitions (FH-029). Its signed `excess64` is higher than the frozen split's in 25 / 36 comparisons and lower in 11 / 36. A lower signed value means fewer misses relative to the new-segment steady state, not by itself a shorter or smaller transient; the temporal-reaction falsification rests on FH-029.

What the FALSIFIED verdict does **not** mean:
- that workloads are stationary;
- that workload dependence is falsified;
- that layer-aware allocation is useless;
- that no faster or different controller could exploit workload transitions.

## Limitations

- [LIMITATION] One session per model and split (four sessions in all); DEV and EVAL sessions are different texts.
- [LIMITATION, 2026-10-12] **Retrospective synthesis.** The JSD operationalisation was specified after the data existed and after FH-028 / FH-029. It is descriptive, not a confirmatory or untouched-holdout test.
- [LIMITATION, 2026-10-12] **Windows.** 64-step windows on ~299-step segments give only three or four within-segment windows per segment (two pairs in the 80B EVAL's 228-step final segment). The boundary windows may overlap the reference windows of the same segments, so the reference is descriptive, not independent noise.
- [LIMITATION, 2026-10-12] **Counterfactual replay.** The transient results come from the replay of one recorded trace per session, for the untuned `adaptive-e64` controller only.

## Status history

- 2026-10-03 PLANNED.
- 2026-10-12 PLANNED: retrospective, analysis-only closure prepared.
  - The four EXP-007 transition sessions exist and are qualified, so no new trace will be collected.
  - The synthesis tool, its tests and `analysis_freeze.json` are in place. The freeze records the sources, fingerprints, the JSD specification and the verdict logic, and is labelled as frozen after the data existed.
  - Review correction (before any `--analyze`): the verdict was made deterministic from FH-029, and the 12 numerical checks became reproducibility evidence (a contradiction is an evidence error, not a verdict change). The first freeze is kept byte-for-byte as `analysis_freeze_pre_review_2026-10-12.json`.
  - Further pre-analysis amendments, each archiving the previous active freeze byte-for-byte. Sources, source fingerprints and the scientific specification are unchanged throughout:
    - test-only ResourceWarning fix (`analysis_freeze_pre_test-cleanup_2026-10-12.json`);
    - full validation of the active freeze, so that any change to a frozen specification field, source identity or fingerprint is a mismatch, and `--check` fails non-zero on a mismatch (`analysis_freeze_pre_spec-validation_2026-10-12.json`).
  - The verdict is pending the synthesis outputs and independent review.
- 2026-10-12 → **FALSIFIED for the current controller**.
  - A single `--analyze` run wrote `processed/` from the four existing EXP-007 sessions under the active freeze; no new model run, trace or benchmark. The outputs were independently reviewed.
  - Verdict from the deterministic FH-029 logic, independent of JSD. JSD (retrospective, descriptive): median boundary/within ratio 2.84–4.02 per trace. Transient: adaptive signed `excess64` lower than frozen in 11 / 36 comparisons. FH-029 consistency: 12 / 12 and 12 / 12.
  - `results.csv` added as a byte copy of `processed/transient_summary.csv`. FH-029 unchanged.
