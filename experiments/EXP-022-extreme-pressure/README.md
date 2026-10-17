# EXP-022 — Extreme memory pressure: where streaming stops being useful

| field | value |
|---|---|
| ID | `EXP-022-extreme-pressure` |
| Status | **INCONCLUSIVE** |
| Period | 2026-10-14 → 2026-10-15 |
| Origin | Argine2 |
| Related questions | RQ-008 |
| Supersedes / superseded | — |

> **Closure note (2026-10-15): EXP022-P1 completed; formal status INCONCLUSIVE.**
> - **Execution.** All 45 frozen launches were executed on the frozen protocol/binary configuration.
> - **H-5.** Both frozen 30B / 768 MiB diagnostic replicas were eligible; mean workload gate-slice hit rate was 0.2322%. The historical >30% falsifier was not met. No positive support threshold was pre-specified, so the formal status is INCONCLUSIVE.
> - **Artifacts.** Raw sessions, results.csv, processed/summary.json, validation-registry entries, and all 18 post-session replay-fidelity diagnostics are retained.

## Research question

What happens below 1.5 GiB (1 GiB, 768 MiB, 512 MiB), and where does each policy become non-interactive?

## Hypothesis

[HYPOTHESIS] Prediction written before measuring (from the EXP-009 pilot): on Qwen3-30B-A3B, below one token's expert footprint (~1.02 GiB, 384 experts) the runtime's sharded LRU collapses (pilot replay: 0-20% expert hits at 1 GiB, ~0-2% at 768 MiB), while frequency-based retention would keep ~30-45%. If the runtime keeps more than 30% expert-level hits at 768 MiB, the prediction is falsified.

*(Historical, 2026-10-03, kept verbatim.)*
- **Provenance of the pilot numbers.** The 0–20% at 1 GiB and ~0–2% at 768 MiB are the global-LRU and `sharded-lru-16-expert` rows of the EXP-009 pilot. That pilot was a counterfactual, expert-unit, decode-only replay of one legacy 127-token mmap-era trace, with uniform expert sizes.
  - `sharded-lru-16-expert` is not a model of the runtime ([FH-025](../../FAILED_HYPOTHESES.md)).
  - "Frequency-based retention" exists only in replay.
- **Operationalisation (2026-10-14, before any data).** The protocol below fixes the term "expert-level hits" to the current EXP-020 metric. It adds no positive support threshold.

## Setup

*Historical setup (2026-10-03), superseded by the protocol below:* `argine research trace` with --budgets 1.5G,1G,768M,512M interleaved, 3 repetitions, on both models (80B footprint per token: 480 experts ≈ 0.67 GiB, so the 80B should collapse later).

<!-- EXP-022 PROTOCOL-BEGIN -->
### Protocol EXP022-P1 (prospective; frozen before any EXP-022 launch)

EXP-022 has **two separate outputs**:
- **A. H-5 mechanism test.**
  - 30B, primary; traced diagnostic block.
  - Measured runtime hit rate, judged by the historical > 30% falsifier.
- **B. RQ-008 usability envelope.**
  - Both models; the 80B arm is **exploratory**.
  - Untraced performance blocks: measured throughput and completion, descriptive only.
  - B never sets the H-5 verdict. A hit-rate collapse and "streaming unusable" are different statements.

#### A. H-5 decision rule

- **Primary point:** Qwen3-30B-A3B at exactly **768 MiB = 805,306,368 B**, in the traced diagnostic block (2 launches).
- **Metric:** the runtime field `expert_level_hit_rate`, read from the launch manifest's `final_store_stats`, with its EXP-020 semantics. It is the workload gate-slice hit rate:
  - slice hits over slice accesses in the gate (first) `MUL_MAT_ID` node of each layer;
  - prefill + decode;
  - workload requests only (warm-up and flush sentinel excluded).

  It is not the decode-only expert-event counterfactual metric of the EXP-009 pilot.
- **Eligibility:** launch outcome VALID, manifest `publication_quality` true, trace complete over the workload, and 0 budget violations.
- **Replicas required (amendment A1):** **both** scheduled 30B / 768 MiB diagnostic launches must be H-5-eligible (`minimum_eligible_launches` = 2).
- **Aggregation:** the arithmetic mean of the two eligible launches. Individual launches and dispersion are reported; there is no launch-level vote.
- **Verdict:**
  - Both eligible and mean **> 0.30**: the historical prediction is **FALSIFIED**.
  - Both eligible and mean ≤ 0.30: **NOT FALSIFIED BY THIS TEST**, and the formal status is **INCONCLUSIVE**.
  - Fewer than 2 eligible launches for any reason, or the 30B diagnostic block incomplete: H-5 is **NOT EVALUABLE** and the formal status is INCONCLUSIVE. This holds even if the single eligible value is far above 0.30. The observed outcomes at 768 MiB are reported.
  - There are no replacement launches and no retries.
- **No positive support threshold.** The 2026-10-03 prediction recorded only a falsification condition. Any support bound chosen now (≤ 10%, ≤ 2%, a fraction of the 1.5 GiB anchor, …) would be selected after the pilot values were known. Agreement with the predicted range may be reported descriptively only.

#### Models and roles

| key | model | role | generation threads | batch threads | one-token footprint F | whole layer |
|---|---|---|---|---|---|---|
| 30b | Qwen3-30B-A3B Q4_K_M | primary (H-5 + envelope) | 12 | 12 | 1,097,072,640 B = 1046.2 MiB = 1.0217 GiB | 391,643,136 B = 373.5 MiB |
| 80b | Qwen3-Next-80B-A3B-Instruct Q3_K_M | exploratory (envelope + diagnostics) | 4 | 12 | 719,585,280 B = 686.2 MiB = 0.6702 GiB | 830,472,192 B = 792.0 MiB |

- **F definition.** The sum over the 48 MoE layers of top_k × the bytes the store needs for one expert of that layer (gate + up + down slices, each rounded up to 4096 B).
  - The 80B's shared expert is not in the store.
  - F is a **derived quantity from existing metadata, not an EXP-022 measurement**.
- **F source.** The tracked EXP-007 files `analysis-v3/{30b,80b}-natural-language-dev-run-00-4G/analysis_manifest.json` (`trace_meta`).
  - `--check` re-derives F from them.
  - `--preflight` re-derives it again from each model's GGUF header, with the same function the runner uses (`argine.research.runner.expert_bytes_from_model`).
- **"Whole layer"** is the largest layer × all its experts. The store warns at startup when the budget is below it ("sotto il minimo teorico"). The warning is recorded, not an error.
- **Threads.** The generation-thread settings are EXP-027's optima **at 4 GiB only**. They are a frozen configuration choice here, not a claim of optimality below 1.5 GiB. Batch threads 12 are planner-resolved and verified.

#### Budgets (exact bytes; main arm only)

The runtime splits the budget into n = min(16, ⌊budget / 32 MiB⌋) shards, each with a hard budget of budget / n (`patches/argine-store.cpp`, `psx_cache_new_n`, certified by BUILD-INFO). Every main-arm budget therefore has **16 shards**.

| model | label | bytes | ≈ F | per shard | role |
|---|---|---|---|---|---|
| 30b | 1.5G | 1,610,612,736 | 1.468 | 96 MiB | in-session anchor |
| 30b | 1152M | 1,207,959,552 | 1.101 | 72 MiB | above the footprint |
| 30b | 1024M | 1,073,741,824 | 0.979 | 64 MiB | approximately at the footprint |
| 30b | 768M | 805,306,368 | 0.734 | 48 MiB | **H-5 primary falsification point** |
| 30b | 512M | 536,870,912 | 0.489 | 32 MiB | far below |
| 80b | 1.5G | 1,610,612,736 | 2.238 | 96 MiB | in-session anchor |
| 80b | 768M | 805,306,368 | 1.119 | 48 MiB | above the footprint |
| 80b | 672M | 704,643,072 | 0.979 | 42 MiB | approximately at the footprint |
| 80b | 512M | 536,870,912 | 0.746 | 32 MiB | below the footprint |

**Excluded from EXP022-P1: 80B at 336 MiB (352,321,536 B).**
- Below 512 MiB the shard count changes (10 at 336 MiB), so pressure and a structural runtime change would vary together.
- If it is ever run, it is a separately labelled exploratory feasibility probe. It is never pooled with, or compared directly against, the 16-shard arm.

#### Runtime (unchanged; EXP-022 measures it as it is)

- **Binary:** `llama-research-v2/llama-server.exe`, SHA-256 `a0063720…5cf3`.
- **BUILD-INFO:** SHA-256 `cc76046a…7d04`; llama.cpp `254b177`; instrumentation `argine2-research-v2`; CPU-only, no Vulkan.
- **Fixed flags:** `--backend cpu`, `--read-mode asincrono` with direct I/O (store line "cache dell'OS scavalcata"), pretouch off, `--ctx 4096 --batch 2048 --ubatch 512`, KV f16/f16, flash attention on, 0 GPU layers, parallel 1, `--expert-stats`, `--repeat 1`, `--retries 0`.
- **Explicit context and batch sizes.** `--ctx`, `--batch` and `--ubatch` are given explicitly so that the planner cannot change them with the free RAM. They equal the planner's choice in EXP-024.
- **Not modified by EXP-022:** llama.cpp, the expert store, the shard and wait logic, the allocator and the binary.

#### Performance blocks (output B; untraced)

- **Tracing off:** `ARGINE_RESEARCH_TRACE` absent. EXP-024 supports < 2% tracing overhead at 4G only, is inconclusive at 1.5G, and nothing is known below that.
- **Command:** `argine benchmark` with the unchanged `argine.bench.DEFAULT_PROMPT` (304 chars, SHA-256 `b1677835…95f3`), `--n-predict 128`, `cache_prompt` false, and the default 8-token warm-up (not measured). This prompt gives continuity with EXP-024 / EXP-027 and with the prompt of the historical pilot trace.
- **Primary metric:** generation tok/s of the single measured request in each launch. The replication unit is one complete launch (fresh process, empty store).
- **Rounds:** 3 rotated rounds per model, each holding every budget exactly once. The 1.5G anchor is in every round. Cooldown is 60 s between launches.

| block | round 1 | round 2 | round 3 |
|---|---|---|---|
| 30b-perf | 1.5G, 1152M, 1024M, 768M, 512M | 1152M, 512M, 768M, 1024M, 1.5G | 512M, 1024M, 1152M, 1.5G, 768M |
| 80b-perf | 1.5G, 768M, 512M, 672M | 768M, 672M, 1.5G, 512M | 672M, 512M, 768M, 1.5G |

- **Order design.**
  - **30B:** a Latin rectangle in which every budget takes three different positions, no ordered neighbour pair repeats, and there are 6 decreasing and 6 increasing transitions.
  - **80B:** the first three rows of a 4 × 4 Williams square.
- **Secondary metrics:** TTFT, prompt tok/s, ms/token, RAM peak and RAM peak − budget. Also the store counters, which are process totals including warm-up and are labelled as such: runtime-style hit %, MiB read, pin-wait time and count, and queue-depth proxy.

**Usability reporting conventions.** These label VALID performance launches. They are **not** hypothesis thresholds and never set H-5.

| label | rule |
|---|---|
| INTERACTIVE | ≥ 5 tok/s |
| DEGRADED | ≥ 1 and < 5 tok/s |
| NON_INTERACTIVE | < 1 tok/s |

- **`I/O DOMINATED` is not used.** The only defined estimator ("residency phase > 50% of token time") cannot be computed from the existing timing counters, which are summed across threads. No new estimator is introduced here.
- **Envelope cell labels.** A cell (model × budget) takes its launches' common label. If the launches disagree, the cell is labelled `MIXED: …`, listing them.
- **Non-VALID launches keep their own label.** `STORE_RESIDENCY_ABORT`, `NON_COMPLETION` and `INVALID` are never converted to `NON_INTERACTIVE`, and the abort causes are counted per cell.

#### Diagnostic blocks (output A for the 30B; traced)

- **When:** each runs only after the same model's performance block is complete.
- **Command:** one `argine research trace` invocation per launch, one budget each, with the same runtime flags.
- **Workload:** `natural-language` v1, split **EVAL**: 3 prompts × 128 tokens, file SHA-256 `71b170bc…05cc`. EVAL follows the repository rule "tune on dev, claim on eval"; nothing in EXP-022 is tuned on traces, and EXP-007 already holds natural-language EVAL runs of both models at ≥ 1.5 GiB.
- **Launches:** 2 per budget, because shard assignment hashes the store's region pointer and may differ between launches. The order is rounds 1–2 of the performance table.
- **Objectives:**
  - workload gate-slice `expert_level_hit_rate`, with its phase semantics;
  - misses and read bytes;
  - pin waits (`w` records) and undo records (`U`);
  - resident bytes vs budget and invariant violations;
  - generated-token completeness;
  - `misses_outside_gate_nodes`, the EXP-020 check of the 3.000 ratio;
  - routing and workload identity across launches;
  - publication quality.
- **Traced throughput** is secondary only.

#### Completion rule (fixed before any sub-1.5G data)

Per-launch wall-clock cap = 600 s + 1800 s × the number of measured requests. 600 s is the readiness timeout of `LlamaServerProcess.wait_ready`; 1800 s is the request timeout of `argine.bench`.
- **Performance launches:** cap **2400 s**. Normal 1.5G reference: EXP-024 30B untraced launches took 46–49 s.
- **Diagnostic launches:** cap **6000 s**. Normal 1.5G reference: EXP-007 natural-language EVAL traced workloads took 108–109 s (30B) and 89–94 s (80B).
- **Margin.** Both caps are about 50× a normal 1.5 GiB launch. A launch can only hit the cap if it runs far below 1 tok/s.
- **Token completeness:**
  - performance: the measured request generates 128 tokens;
  - diagnostic: all 3 workload responses complete without error, each with `predicted_n` = 128 (as in every EXP-007 natural-language EVAL run).

#### Outcome taxonomy (per launch; precedence INVALID > STORE_RESIDENCY_ABORT > NON_COMPLETION > VALID)

- **INVALID: a provenance, configuration or invariant failure, or a harness safety condition. It stops the block.** The checks:
  - the launch's segment of the shared `llama-server.log` has exactly 1 server start, with the frozen command;
  - exactly 1 store-open line, with the frozen store, budget, 16 shards, direct I/O and read mode;
  - the benchmark record or trace manifest matches the frozen plan, binary and backend;
  - resident bytes stay ≤ budget, and violations = 0;
  - no `llama-server` is left running.
- **STORE_RESIDENCY_ABORT: the frozen store-residency abort signature `argine: impossibile rendere residenti gli expert del layer` was observed in the launch's log segment** (amendment A2; previously named INFEASIBLE).
  - **Meaning:** the runtime could not make the requested node's slices resident. The signature alone does not distinguish pinned-capacity exhaustion from a commit or read failure, so the budget is **not** called physically infeasible.
  - **Handling:** it is a scientific outcome, and the block continues.
  - **Cause, recorded by a frozen evidence-only rule (no runtime or binary change):**
    - `capacity_or_pinning`: the abort line names layer L and n requested experts; the launch's log shows full store coverage ("copertura: N tensori di expert su N", so the node pins all three slice kinds); and n × the layer's per-expert bytes (tracked EXP-007 `trace_meta`) exceeds the whole budget. The node's pinned set cannot fit, whatever else happened.
    - `commit_or_read_failure`: the capacity rule is not met, and the launch's research trace records at least one failed read.
    - `cause_unresolved`: every other case. This is the default for untraced launches unless the capacity rule holds.
- **NON_COMPLETION: no abort signature, but the cap was exceeded, the token-completeness rule failed, or the process ended abnormally.** It is a scientific outcome, and the block continues.
- **VALID: provenance verified, the measured workload complete, exit status 0, and no abort signature.**

There are no retries and no replacement runs. No budget is removed adaptively.

#### Sessions, provenance and stopping

- **Blocks:** four, in the frozen order `30b-perf → 30b-diag → 80b-perf → 80b-diag`. Each is a separate session.
  - A block starts only if every earlier block is `complete`, and only from the **same HEAD** as the earlier blocks.
  - A block stopped by an INVALID launch is never resumed or repeated under this freeze.
  - Blocks are not pooled across sessions without explicit labelling.
- **Git gate before a block:**
  - protocol, tool, tests and freeze tracked;
  - working tree clean;
  - HEAD descends from the freeze's recorded commit;
  - freeze hashes unchanged.
- **The one tolerated change** is lines appended to the tracked `llama-research-v2/VALIDATIONS.jsonl` by this experiment's own traced runs, recognised by a `run_dir` under `experiments/EXP-022-extreme-pressure/raw/`. Any other change refuses the run.
- **Machine state before a block:**
  - ≥ 8 GiB free RAM;
  - no browser process (chrome, msedge, firefox, brave, opera);
  - no running `llama-server`;
  - not on battery.

  Free RAM before each launch is recorded, and below 8 GiB it is a warning.
- **Provenance recorded:**
  - Argine2 commit, binary SHA-256 and BUILD-INFO;
  - model and store fingerprints: size and mtime before the block, full SHA-256 afterwards, with the block valid only if size and mtime are unchanged;
  - workload and prompt SHA-256, runtime flags, backend requested / resolved / effective, threads, and the schedule.

#### Replay scope (diagnostic only)

- **Command:** after the sessions, `tools/validate_replay_fidelity.py` per diagnostic launch (commands printed by `--dry-run`).
- **What it may do:** explain mechanism. It never replaces the measured throughput or the measured runtime hit rate.
- **Fidelity is conditional.**
  - The report states whether `w` / `U` records occurred.
  - EXP-021-level fidelity is not claimed where those unvalidated paths occur.
  - Replay disagreement is reported, and never used to exclude a launch.
- **Counterfactuals:** LFU, static-frequency and Belady may be reported as mechanism context. `sharded-lru-16-expert` is not a runtime model.
<!-- EXP-022 PROTOCOL-END -->

## Commands

```bat
python tools\exp022_run.py --check                    :: repository-only validation; starts only git
python tools\exp022_run.py --dry-run                  :: the 45 launches and every command; no subprocess
python tools\exp022_run.py --preflight --block 30b-perf  :: reads GGUF headers + argine benchmark --dry-run; no server
python tools\exp022_run.py --run --block 30b-perf     :: then 30b-diag, 80b-perf, 80b-diag (after review + commit)
python tools\exp022_run.py --analyze                  :: raw/ -> results.csv + processed/summary.json
python -m unittest discover -s tests -p "test_exp022.py" -v
```

*Historical command (2026-10-03), superseded by EXP022-P1. It is not to be run: it measured throughput with tracing on, used the dev split and no thread setting, and had no completion or failure rules.*

```bat
argine research trace MODEL --expert-store --budgets 1.5G,1G,768M,512M --repeat 3 --workload natural-language --out experiments/EXP-022-extreme-pressure/raw
```

## Raw data




All four frozen blocks completed. Primary machine-readable outputs:
- `results.csv` — per-launch classified results.
- `processed/summary.json` — aggregate envelope, diagnostics and frozen H-5 decision.
- `raw/<block>/` — session metadata, records and launch logs.
- `fidelity/<diag-block>/launch-NN/fidelity_runs.json` — 18 post-session replay diagnostics.

`llama-research-v2/VALIDATIONS.jsonl` contains the append-only publication/trace validation records emitted by the EXP-022 diagnostic runs.

## Result




[MEASURED] EXP-022 completed all 45 prospectively scheduled launches.

**30B primary arm.** All 15 performance launches and all 10 diagnostic launches were VALID. Performance remained DEGRADED at every tested budget: 4.00 tok/s at 1.5 GiB, 3.51 at 1152 MiB, 2.99 at 1024 MiB, 2.71 at 768 MiB, and 2.63 at 512 MiB. No STORE_RESIDENCY_ABORT, NON_COMPLETION, or INVALID outcome occurred in the 30B performance arm.

The diagnostic workload gate-slice expert-level hit rate fell sharply with memory pressure: 44.01% at 1.5 GiB, 32.37% at 1152 MiB, 16.19% at 1024 MiB, 0.232% at 768 MiB, and 0.00% at 512 MiB.

At the pre-registered H-5 point, both 30B / 768 MiB replicas were VALID and eligible, with hit rates 0.2433% and 0.2212%; mean 0.2322%. Because the historical falsifier was strictly >30%, H-5 is **NOT FALSIFIED BY THIS TEST**. Formal status remains **INCONCLUSIVE** because no positive support criterion was pre-specified.

**80B exploratory arm.** At 1.5 GiB all three performance launches were INTERACTIVE, mean 5.35 tok/s. At 768 MiB and 672 MiB all launches were DEGRADED, means 4.10 and 3.58. At 512 MiB, two launches completed DEGRADED while one produced STORE_RESIDENCY_ABORT with `cause_unresolved`; the envelope is `MIXED: DEGRADED, STORE_RESIDENCY_ABORT`, not a claim of physical infeasibility.

80B diagnostic hit-rate means were 52.26% at 1.5 GiB, 30.02% at 768 MiB, 9.86% at 672 MiB, and 0.446% at 512 MiB.

All 18 post-session replay-fidelity diagnostics were run. Replay is diagnostic only and does not change launch eligibility, measured throughput, measured runtime hit rates, or the formal H-5 verdict.

## Interpretation




The one-token expert footprint is **not a hard execution threshold** in these workloads. The 30B runtime completed at 768 MiB (0.734× footprint) and 512 MiB (0.489×), while still producing 2.71 and 2.63 tok/s respectively.

Instead, the footprint is associated with a much sharper mechanism-side transition in expert reuse. On 30B, workload gate-slice hit rate falls from 16.19% near the footprint (1024 MiB) to 0.232% at 768 MiB and 0% at 512 MiB. Mechanism/reuse collapse can therefore occur without immediate runtime-completion or throughput collapse.

The exploratory 80B arm shows the same qualitative displacement relative to its smaller one-token footprint: 30.02% at 768 MiB (1.119×), 9.86% at 672 MiB (0.979×), and 0.446% at 512 MiB (0.746×). This is descriptive, not a pre-registered cross-model test, and motivates prospective confirmation of the footprint as a model-specific reuse transition scale rather than a universal feasibility boundary.

The isolated 80B / 512 MiB STORE_RESIDENCY_ABORT further separates usability from hard feasibility: two performance replicas completed at the same budget, so the abort does not establish that 512 MiB is intrinsically impossible.

## Limitations

- **Historical (2026-10-03):** "Store minimum budget: a prompt ubatch can touch every expert of a layer; budgets below psx_cache_min_budget abort by design."
  - *[Corrected 2026-10-14, from the certified source.]* `psx_cache_min_budget` (a whole layer) is checked only at startup, and only logs a warning.
  - What aborts a run is a node whose pinned slices cannot fit in a **shard**: the store waits up to 2000 × 1 ms, then `GGML_ABORT`s.
  - Feasibility therefore depends on the workload's prefill sizes and on the shard assignment, and EXP022-P1 records it as STORE_RESIDENCY_ABORT, with a cause only when the evidence supports one.
- [LIMITATION] **One workload family each:** one performance prompt and one diagnostic workload. Feasibility at the lowest budgets is workload-dependent; longer prefill ubatches pin more slices per node.
- [LIMITATION] **Thread settings:** chosen at 4 GiB (EXP-027) and not re-optimised under pressure.
- [LIMITATION] **Untraced invariant evidence:** in the performance blocks the hard-budget invariant is evidenced by the store's end snapshot (resident ≤ budget) only. The violation counter is exported only in traced runs.
- [LIMITATION] **80B arm:** exploratory, with no pre-data prediction.

<!-- EXP-022 CLOSURE-LIMITATIONS-BEGIN -->
- [LIMITATION] **Small per-budget sample:** performance uses three rotated launches per budget and diagnostics two; EXP-022 maps this workload-specific envelope rather than estimating a population-level failure probability.
- [LIMITATION] **Cross-model transition claim is descriptive:** the 80B arm was exploratory and had no prospective threshold or hypothesis.
- [LIMITATION] **Replay fidelity is diagnostic:** post-session replay can reveal trace-tail incompleteness or replay disagreement but cannot replace measured runtime quantities or alter launch eligibility.
<!-- EXP-022 CLOSURE-LIMITATIONS-END -->

## Status history

- 2026-10-03 PLANNED (prediction recorded).
- 2026-10-14 PLANNED: read-only protocol audit completed, then protocol EXP022-P1, harness and tests prepared and frozen prospectively.
  - The hypothesis is kept verbatim. "Expert-level hits" is operationalised as the EXP-020 gate-slice metric, and no positive support threshold is added.
  - Main-arm budgets have 16 shards. 336 MiB is excluded.
  - Independent review completed. No measurement.
- 2026-10-14 PLANNED: amendments A1–A2, made after review and before any EXP-022 measurement.
  - **A1:** H-5 needs both scheduled 30B / 768 MiB replicas eligible; otherwise it is NOT EVALUABLE.
  - **A2:** INFEASIBLE is renamed to the causally neutral STORE_RESIDENCY_ABORT, with an evidence-only cause label.
  - The first freeze is archived byte-for-byte as `protocol_freeze_pre_review_2026-10-14.json`, and the regenerated active freeze supersedes it.
  - Nothing else changed: hypothesis, 30% threshold, metric, budgets, orders, models, threads, ctx / batch / ubatch, tracing, cooldown, caps, replay role and machine-state rules are as before. No measurement.
- 2026-10-15 COMPLETED: all 45 frozen launches completed under EXP022-P1.
  - 30B: 25/25 VALID; no completion failure down to 512 MiB.
  - H-5 primary point (30B / 768 MiB): both replicas eligible, mean workload gate-slice hit rate 0.2322%; NOT FALSIFIED BY THIS TEST, formal status INCONCLUSIVE because no positive support threshold was pre-specified.
  - 80B exploratory: 19/20 VALID and one STORE_RESIDENCY_ABORT at 512 MiB; two other 512 MiB performance replicas completed DEGRADED.
  - All 18 post-session replay-fidelity diagnostics completed.
  - `python -m unittest discover -s tests -p "test_exp022.py" -v`: 41/41 tests passed.

<!-- EXP-022 CLOSURE-HISTORY-BEGIN -->
- 2026-10-15 CLOSED / INCONCLUSIVE: all 45 frozen launches completed under EXP022-P1.
  - 30B: 25/25 VALID; no completion failure down to 512 MiB.
  - H-5 primary point (30B / 768 MiB): both replicas eligible, mean workload gate-slice hit rate 0.2322%; NOT FALSIFIED BY THIS TEST, formal status INCONCLUSIVE because no positive support threshold was pre-specified.
  - 80B exploratory: 19/20 VALID and one STORE_RESIDENCY_ABORT at 512 MiB; two other 512 MiB performance replicas completed DEGRADED.
  - All 18 post-session replay-fidelity diagnostics completed.
  - Final EXP-022 test suite: 41/41 tests passed before documentary closure.
<!-- EXP-022 CLOSURE-HISTORY-END -->
