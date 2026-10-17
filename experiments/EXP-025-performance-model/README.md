# EXP-025 — Interpretable per-token performance model

| field | value |
|---|---|
| ID | `EXP-025-performance-model` |
| Status | **SUPPORTED** |
| Period | 2026-10-15 |
| Origin | Argine2 |
| Related questions | RQ-013 |
| Supersedes / superseded | — |

## Research question

Can T_token = T_compute + T_unhidden_IO + T_sync + T_controller predict measured ms/token within 10-15% median error?

## Hypothesis

[HYPOTHESIS] Four physically meaningful coefficients, fitted by non-negative least squares, predict leave-one-out ms/token within 15%.

This hypothesis predates the EXP-025 primary fit. The numerical success criterion is the implementation-defined leave-one-out median absolute percentage error strictly below 15%.

## Setup

*Historical setup (2026-10-03), kept as written:*

Features from EXP-007/022 runs (bytes missed, reads, nodes, queue depth per token from per-request timestamps, with its exact/upper-bound status); `argine research predict`.

The primary protocol below refines the dataset and measurement semantics prospectively, before the EXP-025 primary fit, without changing the historical hypothesis.

## Primary protocol

The primary dataset is derived from the existing EXP-022 `30b-diag` campaign for Qwen3-30B-A3B.

Primary configurations are the five expert-memory budgets:

- `1.5G`
- `1152M`
- `1024M`
- `768M`
- `512M`

Each configuration has two independently launched diagnostic replicas.

Replicas are not treated as independent configurations.

The statistical unit for primary leave-one-out validation is the budget configuration.

Per-replica target and features are computed first, then the two replicas belonging to the same budget are aggregated before fitting and validation.

A budget configuration is eligible only if both of its scheduled replicas are eligible. There is no single-replica fallback and no replacement run.

The primary dataset therefore contains exactly five configuration-level observations when fully evaluable.

If any budget configuration is not evaluable, fewer than five primary configurations remain and the primary hypothesis test is `INCONCLUSIVE / NOT EVALUABLE`.

## Replica eligibility and attribution integrity

A replica is eligible only when its existing EXP-022 provenance and validation classify it as a valid diagnostic measurement with complete workload trace and no protocol-invalidating condition.

In addition, deterministic preprocessing must verify all of the following for that replica:

- every selected `IORecord.req` has exactly one matching `Access.req`;
- no selected I/O request is unmatched or multiply matched;
- the matching Access is attributable to a Node under the existing Argine node-attribution semantics;
- the Node graph maps to a trace Step with a known phase;
- the workload trace is complete.

A configuration or replica is never removed because of its EXP-025 residual or prediction error.

## Decode-only measurement window

The prediction target is generation throughput, therefore all primary predictor numerators are computed on the workload decode phase only.

The trace format labels each forward-pass graph as either `prefill` or `decode`.

A step is decode when:

`n_tokens == 1`

I/O attribution uses the trace request identifier rather than the I/O issue timestamp.

The frozen attribution chain is:

`IORecord.req -> Access.req -> Access.t_ns -> Node -> graph -> Step.phase`

The `req` identifier is shared by the authoritative runtime `A` and `I` records.

A pre-freeze audit on EXP-022 30B diagnostic launch 01 at 1.5G found:

- 277700 I/O records;
- 277700 I/O records with exactly one matching Access by `req`;
- 0 unmatched;
- 0 multiply matched.

Access-to-node attribution follows the existing Argine `node_attributed_outcomes()` timestamp semantics.

No primary I/O record is assigned to decode by testing whether `IORecord.issue_ns` lies directly inside a Node residency interval.

That candidate rule was tested before freeze and rejected because only 92594 of 277700 I/O records matched that condition.

## Generated-token denominator

The authoritative generation denominator is the same denominator used by the measured `gen_tok_s` target:

`generated_tokens = sum(predicted_n)` across the three frozen natural-language workload requests.

For the frozen workload:

- 3 requests;
- `predicted_n = 128` for each request;
- `generated_tokens = 384` per replica.

The measured target is therefore:

`ms_per_token = sum(predicted_ms) / 384`

which is equivalent to:

`ms_per_token = 1000 / gen_tok_s`

The workload trace contains 127 decode steps per request because EXP-022 trace completeness is defined against `predicted_n - 1`.

Therefore:

- workload decode steps = `127 * 3 = 381`;
- generated-token denominator = `128 * 3 = 384`.

The whole-lifetime `summary.json` field `trace.decode_tokens` is not used as the EXP-025 denominator because it includes warm-up and flush-sentinel segments.

All per-token predictor features below are divided by the frozen generated-token denominator of 384, not by 381 workload decode steps and not by the whole-lifetime `trace.decode_tokens` value.

The first generated token has no corresponding decode graph under the frozen trace semantics. EXP-025 does not post-hoc reassign prefill I/O to decode; decode-only numerators are retained and normalized to the same generated-token scale as the target.

## Primary features

All predictor numerators are computed from workload decode-attributed records only.

### bytes_missed_per_token

Sum of `IORecord.bytes` for I/O requests whose matching Access belongs to a workload decode graph, divided by `generated_tokens = 384`.

### reads_per_token

Number of decode-attributed I/O records divided by `generated_tokens = 384`.

### nodes_per_token

Number of Node records belonging to workload decode graphs divided by `generated_tokens = 384`.

### qd

`qd` is the time-weighted mean queue depth while the decode-attributed I/O queue is busy, using the same queue-depth reconstruction semantics as `argine.research.iotimeline.queue_depth`.

The queue statistic is computed again after selecting workload decode-attributed I/O records.

The whole-run queue statistic is not reused.

If one or more selected I/O completion timestamps are bounded rather than exact, the resulting queue statistic retains `upper_bound` status.

It is not reinterpreted as an exact measurement.

### controller_ms_per_token

The primary EXP-022 runtime uses the static controller path.

`controller_ms_per_token = 0`

unless existing frozen provenance demonstrates otherwise.

## Model

The pre-existing predictor is kept unchanged for the primary test.

The model is:

`T_token = T_compute + T_unhidden_IO + T_sync + T_controller`

with four non-negative fitted coefficients:

`(c, s, a, b) >= 0`

The predictor design terms remain those implemented before the primary fit:

- constant compute term;
- MoE-node synchronization term;
- bytes divided by the cost-model bandwidth curve at `qd`;
- per-read term.

The current cost-model bandwidth curve and its queue-depth clamping semantics are frozen for the primary analysis.

They will not be changed after inspecting EXP-025 prediction errors.

## Replica aggregation

Feature and target values are first derived independently for each replica.

For each budget, the two eligible replica-level scalar rows are aggregated by the arithmetic mean.

This arithmetic-mean rule applies identically to:

- `ms_per_token`;
- `bytes_missed_per_token`;
- `reads_per_token`;
- `nodes_per_token`;
- `qd`;
- `controller_ms_per_token`.

Because both replicas use the same frozen 384-token workload denominator, equal-weight arithmetic averaging gives each independent launch equal weight.

Configuration-level `q_status` is `exact` only if both replicas are exact; if either replica is `upper_bound`, the configuration-level status is `upper_bound`.

No replica may be selectively discarded based on its effect on prediction error.

## Validation

Primary validation is leave-one-configuration-out over the five aggregate budget configurations.

For each held-out budget:

1. fit non-negative coefficients using the other four configurations;
2. predict the held-out `ms_per_token`;
3. compute absolute percentage error.

The primary summary metric is the median held-out absolute percentage error.

The historical hypothesis is supported by this primary test only if:

`median absolute percentage error < 15%`

The threshold is strict.

A value equal to 15% does not satisfy the hypothesis.

A median error greater than or equal to 15% falsifies the primary predictive claim, provided all five preregistered configurations are evaluable.

## Sensitivity and external validation

EXP-007 may be used after the frozen primary analysis as a sensitivity or external-validation dataset.

EXP-007 does not determine the primary EXP-025 verdict.

Differences in trace generation, instrumentation, queue-depth exactness, runtime provenance, thread semantics, or configuration availability must remain explicit.

The historical Setup statement referring to features from EXP-007/022 is preserved by using EXP-022 for the primary test and EXP-007 for sensitivity/external validation.

## 80B analysis

Qwen3-80B data are exploratory for EXP-025.

They do not enter the primary RQ-013 verdict because the current four-coefficient predictor contains one fitted compute intercept and the primary protocol does not introduce an additional model-specific intercept after seeing the data.

Cross-model conclusions belong primarily to EXP-026.

## Queue-depth limitation

Some completion timestamps are upper bounds.

Consequently, reconstructed `qd` can be an upper bound on true queue depth.

The cost-model bandwidth table also has a finite measured queue-depth range.

Values above that range are handled by the existing `CostModel.bandwidth()` clamping behavior.

Neither limitation is corrected post hoc in the primary fit.

## Tracing limitation

EXP-024 bounded tracing overhead below 2% at the 30B 4G benchmark configuration, while the 1.5G result was inconclusive.

Tracing overhead was not independently measured for every EXP-022 sub-1.5G configuration.

This remains a limitation of the EXP-025 target measurements and must be discussed when interpreting residual error.

## Commands

The canonical prediction command remains:

```bat
argine research predict runs.csv --out prediction.json
```

The deterministic EXP-025 preprocessing step constructed the frozen five-configuration primary `runs.csv` from the existing EXP-022 diagnostic traces.

No new model inference was required to construct the primary dataset.

## Raw data

Primary source data:

`experiments/EXP-022-extreme-pressure/raw/30b-diag/`

Five budgets, two diagnostic replicas per budget.

At protocol freeze, no EXP-025 primary prediction errors had been inspected. The frozen primary dataset was subsequently generated from all ten eligible replicas and committed before the first primary fit.

## Result

[SUPPORTED] The preregistered primary predictive criterion was met.

The frozen five-configuration leave-one-configuration-out evaluation produced:

| held-out budget | measured ms/token | predicted ms/token | absolute percentage error |
|---|---:|---:|---:|
| `1.5G` | 258.11498 | 265.92740 | 3.0267209% |
| `1152M` | 302.79354 | 295.16062 | 2.5208330% |
| `1024M` | 347.33938 | 351.19403 | 1.1097639% |
| `768M` | 391.97368 | 400.53006 | 2.1828968% |
| `512M` | 402.70998 | 393.63013 | 2.2546885% |

Primary metric:

`median leave-one-configuration-out absolute percentage error = 2.2546885%`

Frozen success criterion:

`median APE < 15%`

Therefore the historical predictive hypothesis is **SUPPORTED BY THE PRIMARY TEST**.

`results.csv` transcribes the five leave-one-configuration-out folds from the committed `processed/prediction.json`; nothing was refitted or recomputed.

All five held-out configurations were predicted within 3.03% absolute percentage error; the result is not driven by a single favorable fold.

The full five-configuration NNLS fit returned:

| coefficient | fitted value |
|---|---:|
| `compute` | 85.5716 |
| `sync_nodes` | 1.0843354e-15 |
| `io_bytes_over_bw` | 0.85973422 |
| `reads` | 0.043179181 |

The coefficient values are reported as fitted outputs, but they do not all admit separate physical interpretation in this dataset. In particular, `nodes_per_token = 142.875` for every primary configuration, so the `compute` intercept and `sync_nodes` feature are perfectly collinear across the five observations. The primary test therefore establishes the preregistered predictive claim, not separate identifiability of all four physical coefficients.

[DERIVED, rough consistency only] The earlier default cost-model comparison predicted 72 ms/token of stall for the pilot trace at 4 GiB against an observed ≈55 ms residency gap on the 30B. That historical comparison remains contextual only and was not part of the EXP-025 primary verdict.

## Interpretation

For the frozen Qwen3-30B-A3B natural-language configuration, a compact non-negative performance model using decode-phase missed bytes, reads, queue depth and MoE-node count predicts held-out ms/token across the five tested memory budgets with a median error of 2.2546885%.

This is substantially below the preregistered 15% threshold and supports RQ-013's central predictive claim for this configuration.

The result also clarifies what EXP-025 does **not** establish:

- it does not show that the fitted `sync_nodes` coefficient is physically zero;
- it does not separately identify `compute` and `sync_nodes`, because `nodes_per_token` is constant across all five primary configurations;
- it does not validate the same coefficients for Qwen3-80B or for a different workload/runtime;
- it does not remove uncertainty from upper-bound queue-depth timestamps or from cost-model clamping at high queue depth;
- it does not supersede EXP-026 for cross-model regime interpretation.

The strongest defensible conclusion is therefore that the frozen four-term model has high held-out predictive accuracy on the primary 30B budget sweep, while coefficient-level causal interpretation remains only partial.

## Limitations

Primary limitations after completion:

- exactly five independent configuration-level observations were available for leave-one-configuration-out validation;
- each configuration is an arithmetic mean of two eligible replicas;
- the generation target uses 384 generated tokens while the trace exposes 381 workload decode steps;
- all primary `qd` values retain `upper_bound` completion semantics;
- the highest-pressure configurations reach or exceed the measured queue-depth range and therefore use the frozen CostModel clamp;
- tracing overhead was not independently bounded for every EXP-022 memory budget;
- the primary result is specific to the frozen Qwen3-30B-A3B natural-language EXP-022 diagnostic configuration;
- `nodes_per_token` is constant at 142.875 across all five configurations, making the `compute` and `sync_nodes` columns perfectly collinear and preventing separate identification of those two coefficients;
- the fitted coefficient vector should therefore not be interpreted as a unique decomposition of physical per-token cost;
- EXP-007 remains sensitivity/external-validation material and Qwen3-80B remains exploratory for EXP-025; neither changes the primary verdict.

## Provenance

Primary protocol freeze:

`26da7819f79a114743aa8f38645c71e70e84ed7d`

Frozen preprocessing implementation:

`18d7afb6c0c356d54043349aed1c9f432cd25b4c`

Initial frozen primary dataset commit:

`668701b0f7f9ccfef048203aa54cccc48426512b`

Dataset line-ending normalization:

`af11f2574ff53007f65090a4d440a22cc198ee0d`

Primary prediction result:

`5d12049825deebc9f9fcffa901c21c16ce626827`

The later prediction line-ending normalization changed serialization only and did not alter any numerical result.

## Status history

2026-10-03 PLANNED.

2026-10-15 primary EXP-025 protocol specified before fitting. Primary dataset frozen to the five EXP-022 30B diagnostic budgets; decode-only numerator semantics, generated-token denominator, equal-weight replica aggregation, configuration-level leave-one-out validation, and strict `<15%` success criterion preregistered.

2026-10-15 preprocessing validated all 10 scheduled replicas and produced all 5 preregistered budget configurations.

2026-10-15 primary leave-one-configuration-out evaluation completed with median APE 2.2546885%; the preregistered predictive hypothesis was SUPPORTED BY THE PRIMARY TEST. Coefficient-level interpretation remains limited by perfect collinearity between the constant intercept and constant `nodes_per_token` feature.
