# EXP-025 protocol freeze

Frozen before construction of the EXP-025 primary fitted dataset and before inspection of primary prediction errors.

## Historical hypothesis

[HYPOTHESIS] Four physically meaningful coefficients, fitted by non-negative least squares, predict leave-one-out ms/token within 15%.

The historical hypothesis text is unchanged.

## Historical setup

Features from EXP-007/022 runs (bytes missed, reads, nodes, queue depth per token from per-request timestamps, with its exact/upper-bound status); `argine research predict`.

The protocol below refines the primary dataset and measurement semantics before fitting without replacing the historical setup or hypothesis.

## Primary dataset

Source: EXP-022 `30b-diag`.

Model: Qwen3-30B-A3B.

Workload: natural-language v1.

Runtime: frozen EXP-022 research-v2 runtime.

Threads: 12.

Budgets:

1. 1.5G
2. 1152M
3. 1024M
4. 768M
5. 512M

Two diagnostic replicas exist for each budget.

The budget configuration, not the individual launch, is the unit of leave-one-out validation.

A budget configuration is eligible only if both scheduled replicas are eligible. There is no single-replica fallback and no replacement run.

## Eligibility

A replica is eligible only when its existing EXP-022 provenance and validation classify it as a valid diagnostic measurement with complete workload trace and no protocol-invalidating condition.

Deterministic preprocessing must additionally verify:

- every selected `IORecord.req` has exactly one matching `Access.req`;
- no selected I/O request is unmatched or multiply matched;
- each matched Access is attributable to a Node using existing Argine semantics;
- the Node graph maps to a trace Step with a known phase;
- workload trace completeness is true.

No observation is excluded because of its EXP-025 residual or prediction error.

If fewer than five budget configurations remain evaluable, the primary test is:

`INCONCLUSIVE / NOT EVALUABLE`

No replacement runs are introduced after inspection of EXP-025 prediction errors.

## Measurement window

Target and predictor features refer to generation/decode behavior.

All derived primary predictor numerators are decode-only.

The trace phase semantics are frozen to the existing Argine definition:

- `n_tokens == 1` -> decode;
- otherwise -> prefill.

## I/O attribution

I/O phase attribution uses the existing shared request identifier:

`IORecord.req -> Access.req`

The authoritative runtime trace format records the same request identifier in `A` and `I` records.

Pre-freeze audit on EXP-022 30B 1.5G launch 01:

- I/O records: 277700
- exactly one Access with matching req: 277700
- no matching Access: 0
- multiple matching Access records: 0

The matching Access timestamp is attributed to a Node using the existing Argine node-attribution semantics.

The Node identifies its graph, and the graph identifies its trace Step and phase.

Frozen chain:

`IORecord.req -> Access.req -> Access.t_ns -> Node -> graph -> Step.phase`

The alternative rule:

`IORecord.issue_ns inside Node residency interval`

is explicitly not used.

It was tested pre-freeze and matched only:

`92594 / 277700`

I/O records.

## Generated-token denominator

The authoritative denominator is:

`generated_tokens = sum(predicted_n)`

over the three frozen natural-language workload responses.

The frozen workload has three requests with `predicted_n = 128` each, therefore:

`generated_tokens = 384`

per replica.

The target is:

`ms_per_token = sum(predicted_ms) / 384`

equivalently:

`ms_per_token = 1000 / gen_tok_s`

EXP-022 trace completeness defines expected decode steps as `predicted_n - 1`.

Each of the three workload requests therefore has 127 decode steps, for:

`workload_decode_steps = 381`

The whole-lifetime `summary.json` `trace.decode_tokens` field includes warm-up and flush-sentinel segments and is not used as the EXP-025 denominator.

All per-token predictor features use the same frozen generated-token denominator of 384 as the target.

They do not use 381 workload decode steps and do not use the whole-lifetime `trace.decode_tokens` value.

The first generated token has no corresponding decode graph under the frozen trace semantics. No prefill I/O is post-hoc reassigned to the decode numerator.

## Predictor inputs

For every eligible replica, using workload decode-attributed records only:

`bytes_missed_per_token = sum(IORecord.bytes) / 384`

`reads_per_token = count(IORecord) / 384`

`nodes_per_token = count(Node) / 384`

`qd = q_time_weighted_mean_when_busy(decode-attributed IORecords)`

`controller_ms_per_token = 0`

unless existing frozen provenance demonstrates a non-static controller contribution.

## Queue-depth semantics

Queue depth is reconstructed with the existing Argine I/O timeline implementation.

If any selected completion is bounded rather than exact, queue-depth status remains:

`upper_bound`

An upper-bound `qd` is not relabelled or corrected as exact.

The existing CostModel bandwidth curve and its clamping behavior are frozen.

## Aggregation

Replica-level target and feature rows are constructed first.

For each budget, the two eligible replica-level scalar rows are aggregated by arithmetic mean.

The same equal-weight arithmetic-mean rule applies to:

- `ms_per_token`;
- `bytes_missed_per_token`;
- `reads_per_token`;
- `nodes_per_token`;
- `qd`;
- `controller_ms_per_token`.

Because both replicas use the same 384-token workload denominator, this gives each independent launch equal weight.

Configuration-level `q_status` is `exact` only when both replicas are exact. If either replica is `upper_bound`, configuration-level `q_status` is `upper_bound`.

No leave-one-replica-out analysis may substitute for the frozen leave-one-configuration-out primary test.

## Predictor

The pre-existing four-feature NNLS implementation is frozen.

Coefficients:

`(c, s, a, b) >= 0`

No new predictor term, alternative bandwidth curve, model-specific correction, queue correction, or post-hoc feature transformation may be introduced into the primary test after prediction errors are inspected.

Such analyses, if useful, must be labelled secondary or exploratory.

## Primary validation

Five-fold leave-one-configuration-out validation.

For each fold:

1. four budget configurations are used for fitting;
2. the remaining budget configuration is predicted;
3. absolute percentage error is calculated.

Primary metric:

`median absolute percentage error`

across the five held-out predictions.

## Primary decision rule

If:

`median APE < 15%`

then:

`historical predictive hypothesis supported by the primary test`

If:

`median APE >= 15%`

then:

`historical predictive hypothesis falsified by the primary test`

If fewer than five configurations are evaluable:

`INCONCLUSIVE / NOT EVALUABLE`

The success threshold is strictly less than 15%.

## Secondary analyses

EXP-007:

`sensitivity / external validation only`

Qwen3-80B:

`exploratory only`

Neither can reverse or replace the frozen primary verdict.

## Known pre-fit limitations

1. Only five independent primary configurations.
2. Two replicas per configuration.
3. The generation target uses 384 generated tokens while the trace exposes 381 workload decode steps.
4. Queue-depth completion timestamps are not universally exact.
5. Queue depth above the measured bandwidth curve is subject to the existing clamp.
6. EXP-024 did not bound tracing overhead at every EXP-022 memory budget.
7. The primary result is specific to the frozen 30B EXP-022 diagnostic workload/runtime.
8. The historical rough 4G stall comparison is context only and is not an EXP-025 measurement.

## No new primary inference

The primary EXP-025 dataset is derived from already completed EXP-022 traces.

No additional model inference is required or permitted merely to improve primary model fit after seeing EXP-025 results.
