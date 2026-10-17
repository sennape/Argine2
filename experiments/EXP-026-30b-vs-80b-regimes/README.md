# EXP-026 — Two experimental regimes: 30B-A3B vs Next-80B-A3B

| field | value |
|---|---|
| ID | `EXP-026-30b-vs-80b-regimes` |
| Status | **INCONCLUSIVE** |
| Period | 2026-10-15 → 2026-10-16 |
| Origin | Argine2 |
| Related questions | RQ-002, RQ-009 |
| Supersedes / superseded | — |

> **Closure note (2026-10-16): EXP026-P1 completed; formal status INCONCLUSIVE.**
> - **Execution.** All frozen blocks completed on the frozen protocol commit `7168b936`: diskbench 60/60, performance 24/24 and diagnostics 12/12 VALID (all 12 diagnostic launches eligible). No retry, no replacement, no adaptive pruning.
> - **Primary explanandum: NOT DISTINGUISHED.** At matched budget/F and 4 generation threads, β30 = 0.4756 and β80 = 0.4948 (Δβ = +0.0192); the launch-pairing envelopes overlap. The historical flatter-80B slope did not reproduce in the matched-F region on the current runtime. This does not retrospectively falsify EXP-006 or alter F-004.
> - **Traced-slope agreement gate: FAIL** (traced Δβ = −0.2694, opposite in sign). All four candidates are therefore NOT EVALUABLE.
> - **RQ-009** stays PARTIALLY ANSWERED, with a data note.

## Research question

Why does the 80B behave differently from the 30B (requirement section 56, question 10)?

## Hypothesis

[HYPOTHESIS] Competing explanations, to be discriminated: (a) routing concentration (smaller EWS fraction); (b) smaller slices (440-704 KiB vs 930 KiB) raise queue depth per byte; (c) the 80B is bookkeeping-bound (commit/decommit/lock 56% of its I/O bucket) rather than read-bound.

*(Historical, 2026-10-03, kept verbatim.)*

- **Candidate (d), carried forward separately.** RQ-009 (`OPEN_QUESTIONS.md`) already lists a fourth candidate, "(d) different compute (Gated DeltaNet in 36 layers)". It predates this protocol and is not part of the quotation above. EXP026-P1 tests it with its own frozen rule.
- **Operationalisation (2026-10-15, before any EXP-026 data).** The protocol below fixes the explanandum, the endpoints and a rule per candidate. None of the four candidates had a binary threshold historically, and no single cause is forced.

## Setup

*Historical setup (2026-10-03), superseded by the protocol below:* Same workloads and budgets on both models; characterization + I/O timeline + node timing + replay.

<!-- EXP-026 PROTOCOL-BEGIN -->
### Protocol EXP026-P1 (prospective; frozen 2026-10-15 before any EXP-026 measurement)

Machine-readable copy: `protocol_freeze.json` (spec) and `config.json` (`protocol`). Harness: `tools/exp026_run.py`. Tests: `tests/test_exp026.py`. Nothing below may be changed after the freeze by anything observed in the data.

#### Question, hypothesis and candidates

- Historical question (unchanged): *Why does the 80B behave differently from the 30B?*
- Historical EXP-026 hypothesis (2026-10-03, quoted verbatim in the Hypothesis section): candidates (a), (b) and (c).
- Additional pre-existing candidate, carried forward from RQ-009 (`OPEN_QUESTIONS.md`) with its wording unchanged and **not** inserted into the historical quotation: "(d) different compute (Gated DeltaNet in 36 layers)".
- None of the four candidates was originally given a binary threshold. EXP-026 is a discriminating explanatory experiment. It does not force a single cause, and every causal statement is limited to "consistent with" / "falsified as stated".

#### Primary explanandum

- For model m: `β_m = Δ ln(gen_tok_s) / Δ ln(budget/F)` between P1 and P2, i.e. `β_m = (ln s_P1 − ln s_P2) / (ln x_P1 − ln x_P2)` with s = cell median of the 3 untraced generation tok/s values and x = budget / F.
- Primary explanandum: `Δβ = β_80B − β_30B`, from **untraced** performance launches at **4 generation threads for both models**.
- Historical direction: the 80B had the flatter speed/residency response (F-004: 0.384 vs 0.50–0.58).
- Envelope: for each model, β over all 3 × 3 pairings of one P1 launch with one P2 launch; [min, max].
- Outcomes:
  - `REPRODUCED`: β80 < β30 and the envelopes do not overlap (max env80 < min env30);
  - `REVERSED`: β80 > β30 and the envelopes do not overlap (min env80 > max env30);
  - `NOT DISTINGUISHED`: the envelopes overlap;
  - `NOT EVALUABLE`: a P1 or P2 primary cell of either model is not 3/3 VALID.
- No pooled-SD threshold, no CV threshold, no p-value. The absolute cross-model ratio R = ms/token(80B) / ms/token(30B) is secondary and descriptive. EXP-026 is not "which model is faster".

#### Models and frozen geometry (verified 2026-10-15; re-derived by `--check` from tracked EXP-007 `trace_meta`)

| | Qwen3-30B-A3B Q4_K_M | Qwen3-Next-80B-A3B-Instruct Q3_K_M |
|---|---|---|
| F (one-token routed-expert footprint) | 1,097,072,640 B | 719,585,280 B |
| MoE layers / routed experts per layer / per token per layer | 48 / 128 / 8 | 48 / 512 / 10 |
| routed expert uses per token | 384 | 480 |
| expert bytes per layer | 3,059,712 B (24 layers), 2,654,208 B (24 layers) | 1,622,016 B (layers 0–2), 1,490,944 B (layers 3–47) |
| whole routed-expert layer | max 391,643,136 B (373.5 MiB) | 830,472,192 B (792 MiB, layers 0–2); 763,363,328 B (728 MiB, layers 3–47) |

Distinct expert-store slice sizes (aligned store-read bytes; every (layer, kind) group uniform):

| Model | Kind | Bytes | KiB | Layers |
|---|---|---:|---:|---|
| 30B | gate, up | 884,736 | 864 | all 48 |
| 30B | down | 1,290,240 | 1260 | 0–5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41–47 |
| 30B | down | 884,736 | 864 | 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40 |
| 80B | gate, up | 450,560 | 440 | all 48 |
| 80B | down | 720,896 | 704 | 0–2 |
| 80B | down | 589,824 | 576 | 3–47 |

Expert size is **not** uniform within either model. Reads per decode token at zero hits: 30B 960 × 884,736 + 192 × 1,290,240; 80B 960 × 450,560 + 450 × 589,824 + 30 × 720,896 (both reconstruct F exactly).

#### Pressure grid (primary normalization: budget / F)

| Point | 30B bytes | 30B label | 80B bytes | 80B label | budget/F (30B / 80B) | Role |
|---|---:|---|---:|---|---|---|
| P1 | 4,294,967,296 | 4096M | 2,818,572,288 | 2688M | 3.9149 / 3.9169 | primary slope endpoint |
| P2 | 1,610,612,736 | 1536M | 1,056,964,608 | 1008M | 1.4681 / 1.4689 | primary slope endpoint |
| P3 | 1,073,741,824 | 1024M | 704,643,072 | 672M | 0.9787 / 0.9792 | secondary sub-F probe |

- P1→P2 spans exactly 8/3 (≈ 2.667×) for both models; P3 never defines the primary β.
- All budgets are whole MiB, divisible by 16, ≥ 512 MiB: 16 shards everywhere (`n = min(16, floor(budget / 32 MiB))`).
- No 512 MiB block, no ≈1.10 F block, nothing below 512 MiB.
- At P3 the 80B is below every whole-layer size (672 MiB < 728 MiB); at P1 and P2 both models are above every whole layer.

#### Threads, runtime and store read mode

- Primary performance and diagnostic blocks: `--threads 4` for **both** models; prompt/batch threads 12 (planner `max(threads, min(logical, physical))` = 12 on the i5-1340P, verified in every launch record and llama-server command as `--threads-batch 12`).
- Sensitivity: 30B only, `--threads 12`, at P1 and P2, 3 untraced launches per cell. Descriptive; not part of the primary Δβ decision. No 80B-at-12 block, no thread sweep.
- Runtime as measured in EXP-022 (`llama-research-v2`, server SHA-256 `a0063720…`, BUILD-INFO `cc76046a…`, llama.cpp `254b1773…`): backend cpu, ctx 4096, batch 2048, ubatch 512, KV f16, flash-attn on, `--repeat 1`, `--retries 0`, `--expert-stats`.
- Store read mode: `--read-mode asincrono` (`ARGINE_EXPERT_READ=asincrono`), direct I/O `ARGINE_EXPERT_DIRECT=1`, pretouch 0, 16 shards. Windows semantics: per-thread handles with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, 4096-aligned reads, several overlapped reads in flight per handle, synchronous fallback only when no handle/event is available. The effective settings are recorded and checked per launch (store-open line; diag manifests).

#### Workloads

- Performance (untraced): `argine.bench.DEFAULT_PROMPT` (SHA-256 `b1677835…`, 304 chars), `n_predict` 128, the EXP-022/027 benchmark semantics (8-token warm-up, one measured request).
- Diagnostics (traced): the trace tool accepts only versioned workloads, so the frozen workload is **natural-language v1 EVAL** (`workloads/v1/natural-language.json`, SHA-256 `71b170bc0fde211e3694fede63192c72ea70ef7f561791ac28b82ebac25105cc`, 3 requests × 128 tokens), with the EXP-022 diagnostic semantics. No other workload is substituted.

#### Blocks and fixed run matrix (frozen order: diskbench → perf → diag)

| Block | Content | Count |
|---|---|---:|
| diskbench | 5 slice sizes × 4 reader counts × 3 rounds | 60 device runs |
| perf, primary | 2 models × 3 points × 3 launches, 4 threads | 18 |
| perf, sensitivity | 30B × P1/P2 × 3 launches, 12 threads (integrated in the perf rounds) | 6 |
| diag | 2 models × 3 points × 2 launches, 4 threads, traced | 12 |
| **model launches** | | **36** |

- **No retries. No replacements. No adaptive pruning. No launch is added after a failure.** Categorical failures never increase the count.
- Run orders are seeded and frozen: round r of block B is the canonical cell list shuffled by `random.Random(seed × 100 + r)`; seeds perf 2602601, diag 2602602, diskbench 2602603. Perf: 3 rounds × 8 cells (6 primary + 2 sensitivity); diag: 2 rounds × 6 cells; diskbench: 3 rounds × 20 cells. The expanded orders are stored in the freeze and re-derived by `--check`.
- Cooldown 60 s between model launches (EXP-022/027 precedent); no cooldown between diskbench runs.

#### Outcome taxonomy and eligibility

- Per model launch: `INVALID > STORE_RESIDENCY_ABORT > NON_COMPLETION > VALID` (the EXP-022 classifier, unchanged). A store-residency abort is causally neutral and never called physical infeasibility; its cause label stays evidence-only. An INVALID launch stops its block (no resume). Every scheduled launch remains in the record.
- Performance cells: **3/3 VALID** for the cell to enter any decision. Otherwise the cell is `NOT EVALUABLE`; categorical outcomes are reported; numeric values of VALID launches may be shown labelled `DESCRIPTIVE ONLY` (for 2 VALID launches the center is the explicitly labelled two-value mean) and never feed a candidate or lifecycle decision.
- Diagnostic launch eligibility: VALID, `publication_quality` true, trace complete, `budget_violations` 0, node intervals disjoint and node residency ≤ T in every eligible decode graph, at least one eligible decode graph.
- Diagnostic cells: **2/2 eligible** (EXP-022 A1 precedent); otherwise every mechanism decision that needs the cell is `NOT EVALUABLE`.
- Diskbench run: `INVALID > NON_COMPLETION > VALID`; a device comparison needs all 3 round values VALID for every (slice, readers) cell it uses.

#### Aggregation

- Performance: cell median of 3 VALID launches; every launch value retained; β from the P1/P2 cell medians; envelope from the 3 × 3 launch pairings.
- Diagnostics: per-launch means over eligible decode graphs; cell values only for 2/2 cells; envelopes from the 2 × 2 launch pairings.

#### Accounting: bound decomposition only

Verified on existing EXP-022 traces (no new trace): node intervals within a graph are disjoint, node residency never exceeded the step period, read intervals overlap heavily (mean concurrency 6–9), no read time fell outside thread-0 residency, and 34–48% of completions are upper bounds. An exact `T = C + E_r + E_nr` identity is therefore **not** frozen. For each eligible decode graph g (n_tokens = 1, followed by decode graph g+1 in the same workload request; warm-up and flush-sentinel excluded):

- `T_g = t0(g+1) − t0(g)`, t0 = the graph's earliest router timestamp;
- `Res_g = Σ_nodes (ready − start)` (thread-0 view; disjointness is checked);
- `N_g = T_g − Res_g`, exact. **N is non-residency (compute-inclusive) time, not pure compute**: non-MoE layers, attention, Gated DeltaNet, MoE matmul outside residency, sampling, synchronisation outside the store, tracing cost;
- reads of g = store reads issued in [t0(g), t0(g+1)); `U_all` = union of [issue, completion_bound] over them; `U_ex` = union of [issue, exact_completion] over the reads with exact completion;
- `E_r ∈ [ |U_ex ∩ Res_g| , |U_all ∩ Res_g| ]`;
- `E_nr = Res_g − E_r ∈ [ Res_g − |U_all ∩ Res_g| , Res_g − |U_ex ∩ Res_g| ]`;
- unions, never sums of read latencies; `M_g` = aligned store-read bytes of the reads of g (a wait on another thread's read is an access, not a read).

Per launch: means over eligible decode graphs of T, N, Res, E_r and E_nr bounds and M; `N/T = ΣN / ΣT`; `q_lo/hi = ΣE_r_lo/hi / ΣM`; `m = mean M / F`; depth = Σ read latency / Σ |U_all| (recorded intervals, so bound-derived).

#### Traced-slope agreement gate

- `β_traced` per model from traced decode T at P1/P2 (`β = (ln T_P2 − ln T_P1) / (ln x_P1 − ln x_P2)`), point from cell means, envelope over the 2 × 2 launch pairings.
- FAIL if the sign of `Δβ_traced` differs from the untraced primary `Δβ`, or if the traced envelopes overlap while the primary outcome is `REPRODUCED` or `REVERSED`. NOT EVALUABLE if the primary is NOT EVALUABLE or a P1/P2 diagnostic cell is not 2/2.
- If the gate is not PASS, **every candidate test is NOT EVALUABLE**. The untraced performance result stands; traced results are never primary throughput results.

#### Routing metric taxonomy (three different quantities; never substituted)

1. **Historical EWS-fraction metric**: EWS90 routed experts / total routed experts. It is the literal "smaller EWS fraction". EXP-007 values (30B 0.355 EVAL / 0.337 DEV; 80B 0.258 / 0.261) are historical, descriptive premises only. Because expert size varies by layer, EWS90 bytes / store bytes is **not** equivalent; it is reported separately as a labelled byte-weighted variant.
2. **Pressure-normalized predictor**: EWS90 bytes / F with exact per-layer expert sizes. A predictor under budget/F normalization; not the historical statement and not a decision endpoint.
3. **Candidate-(a) mechanism consequence**: per diagnostic launch `m = decode store-read bytes / eligible decode graphs / F`; elasticity `e_a = Δ ln m / Δ ln(budget/F)` from P1 to P2, 2 × 2 launch-pairing envelope.

#### Diskbench: device-level calibration only

- **Role and limitation.** Diskbench is not an emulator of the expert store. The store issues overlapped asynchronous uncached reads from per-thread handles, several in flight per handle, in per-layer waves. `argine diskbench --kinds direct` issues one blocking uncached `ReadFile` per Python thread (`FILE_FLAG_NO_BUFFERING`, no `FILE_FLAG_OVERLAPPED`) at random 4096-aligned offsets. It reproduces only: uncached/direct reads, alignment, slice size and a controlled number of concurrent reads. It does **not** reproduce the store's `asincrono` semantics. `moe` mode is not used.
- File: `C:\modelli\Qwen3-Next-80B-A3B-Instruct-Q3_K_M.psxs` (one file for every size, so only size and readers vary).
- Sizes (the five verified distinct slice sizes): 450,560; 589,824; 720,896; 884,736; 1,290,240 B. Readers {2, 4, 8, 16}. 3 repetitions, one per round, each round a frozen seeded permutation of the 20 cells. No separate drift-control block. **60 runs.**
- Command per run: `argine diskbench <PSXS> --kinds direct --no-sequential --slice <BYTES> --threads <N> --seconds 4 --json` (parser checked statically by `--check`).
- Telemetry: each run gets `ARGINE_HOME = raw/diskbench/argine-home`, created by the harness before the first run so that no other history is imported; the telemetry line is copied into `raw/diskbench/records.jsonl`, stdout/stderr into `raw/diskbench/logs/`.
- Device endpoint: `B_mix(N) = Σ bytes / Σ (bytes / B(slice_size, N))`, bytes = reads per token × slice size (30B: 960 × 884,736, 192 × 1,290,240; 80B: 960 × 450,560, 450 × 589,824, 30 × 720,896), one value per model, readers count and round. Comparisons at approximately equal bytes in flight: 30B N=2 vs 80B N=4; 30B N=4 vs 80B N=8; 30B N=8 vs 80B N=16.

#### Candidate rules (frozen; evaluated only when the gate is PASS)

**(a) routing concentration.** Premise ("smaller EWS fraction") reported separately. Decision endpoint e_a.

- UNRESOLVED first if the four e_a values do not all have the same sign, or some m ≤ 0.
- SUPPORTED (consistent with routing concentration contributing to the regime difference): max |e_a|80 < min |e_a|30 **and** the miss-byte counterfactual moves β80 toward β30 at both bounds.
- FALSIFIED AS STATED: min |e_a|80 ≥ max |e_a|30.
- WEAKENED: the |e_a| envelopes overlap.
- UNRESOLVED: otherwise (internally inconsistent).
- NOT EVALUABLE: diagnostic eligibility or gate failure. No routing causality is claimed.

**(b) slice size / reads in flight.** `q = E_r / M` (bounded).

- Device: a comparison favours the 80B if min over rounds B_mix80 > max over rounds B_mix30, disfavours if max80 < min30, else indeterminate; device FAVOURS / DISFAVOURS only if all three comparisons agree, else INDETERMINATE.
- Trace: at a point "80B lower q" if max(80B q_hi over eligible launches) < min(30B q_lo); mirror condition for "80B higher"; trace LOWER / HIGHER only if so at both P1 and P2.
- SUPPORTED: device FAVOURS and trace LOWER. FALSIFIED AS STATED: device DISFAVOURS and trace HIGHER.
- WEAKENED: one layer points in the predicted direction while the other is indeterminate.
- UNRESOLVED: device and trace contradict; a relevant traced depth (P1/P2, both models) lies outside the calibrated [2, 16]; any other evaluable combination.
- NOT EVALUABLE: required diagnostic cells, an incomplete diskbench comparison, or gate failure.

**(c) bookkeeping.** Whole-process store counters (commit, decommit, lock, read) are DIAGNOSTIC ONLY.

- FALSIFIED AS STATED: at BOTH P1 and P2, in BOTH eligible 80B diagnostic launches, `E_nr_hi < E_r_lo` (even the upper bound on exposed non-read residency is below the lower bound on exposed read residency).
- WEAKENED: the falsification condition holds at exactly one endpoint; or, at both P1 and P2, max over 80B launches of `E_nr_hi / Res` ≤ min over 30B launches of `E_nr_lo / Res` (the 80B is demonstrably not more non-read dominated).
- UNRESOLVED: otherwise.
- **SUPPORTED is not reachable** with existing instrumentation (E_nr also contains barrier waits and hit lookups). No positive bookkeeping rule exists, and the inability to support (c) does not make (c) false.
- NOT EVALUABLE: diagnostic eligibility or gate failure.

**(d) different compute / non-residency.** Endpoint N/T: non-residency (compute-inclusive) time share; never "pure compute"; Gated DeltaNet itself is not isolated.

- SUPPORTED: at P1 and P2, min(80B N/T) > max(30B N/T) **and** the N counterfactual moves β80 toward β30.
- FALSIFIED AS STATED: at both P1 and P2, max(80B N/T) < min(30B N/T).
- WEAKENED: otherwise when evaluable. NOT EVALUABLE: diagnostic eligibility or gate failure.

#### Accounting counterfactuals (DESCRIPTIVE ACCOUNTING ONLY)

For a term X ∈ {N, E_r, E_nr, M}, using traced cell means, the 80B P2 value is replaced by X80(P1) × X30(P2) / X30(P1), with the other 80B P2 terms held at their observed values (for M, E_r is scaled with the bytes at fixed q); β80 is recomputed with the observed T80(P1). "Moves β80 toward β30" means |β80′ − β30| < |β80 − β30| (traced β). Bounded terms are evaluated at both admissible bound versions (lo: E_r = E_r_lo, E_nr = Res − E_r_lo; hi: E_r = E_r_hi, E_nr = Res − E_r_hi), and a directional statement counts only when both versions move β80 toward β30. These are not causal interventions and carry no significance.

#### Experiment lifecycle (during the freeze: PLANNED, `measurements_made = false`)

- **SUPPORTED** only if the explanandum is `REPRODUCED`, the gate is PASS, and at least one of (a), (b), (c), (d) is SUPPORTED under its own rule (the others may be anything).
- **FALSIFIED** only if the explanandum is `REPRODUCED`, the gate is PASS, and **all four** candidates satisfy their own FALSIFIED AS STATED rule; (c) must itself be falsified. UNRESOLVED, WEAKENED or NOT EVALUABLE never count as falsification.
- **INCONCLUSIVE** in every other case: explanandum NOT EVALUABLE, NOT DISTINGUISHED or REVERSED; gate failure; no supported candidate while at least one candidate is not falsified.

Frozen wording when the explanandum does not reproduce ([not distinguishable from] for NOT DISTINGUISHED, [steeper than] for REVERSED):

> EXPLANANDUM NOT REPRODUCED. At matched budget/F (P1→P2, 4 generation threads, current runtime) the 80B's pressure-response exponent was [not distinguishable from | steeper than] the 30B's. The historical regime difference did not reproduce in the matched-F region on the current runtime. This does not retrospectively falsify EXP-006 or alter F-004. Those were measured over different budget/F regions on earlier binaries and remain valid descriptions of those sessions. The candidate-mechanism hypothesis is INCONCLUSIVE: its object was absent at matched budget/F, so the candidates could not be discriminated against it. Secondary results are reported descriptively.

#### RQ-009 mapping (separate from the lifecycle status; `OPEN_QUESTIONS.md` is not modified by the freeze)

- ANSWERED: the explanandum is evaluable and all four candidates are resolved (SUPPORTED, WEAKENED or FALSIFIED AS STATED).
- PARTIALLY ANSWERED: the explanandum is evaluable and at least one candidate is resolved; the Unknown text is narrowed to the remaining candidates. If the slope difference does not reproduce, RQ-009 may record "the historical slope difference is not reproduced at matched budget/F; consistent with a normalization/region difference" (never "caused by normalization").
- Otherwise RQ-009 remains PARTIALLY ANSWERED with a data note. It never regresses.

#### Secondary and descriptive reporting

R at P1, P2, P3 (cell medians, launch ranges); the 30B 12-thread β and envelope (operational comparison with the 80B at 4 threads); P3 performance and mechanism values; whole-process store counters; routing metrics per launch; traced/untraced comparison. None of these sets a decision.

#### Tracing limitations

- Traced and untraced launches are never pooled; traced T is used only for β_traced, shares and bounds.
- No tracing-overhead claim is made outside EXP-024's tested condition.
- Node residency is the thread-0 view; other threads' reads are counted by interval.
- Some completions are upper bounds, so E_r, E_nr, q and depth are bounds or bound-derived; N includes tracing cost.

#### Known limitations (before measurement)

- Observational cross-model comparison: no randomization across models is possible; architecture, quantization, expert count, slice sizes, routing and F differ together; only the run order is randomized.
- 4 generation threads is the 80B's EXP-027 optimum at 4 GiB; the 30B 12-thread sensitivity covers P1/P2 only; thread × pressure interaction is otherwise unknown.
- The 80B is below every whole-layer size at P3.
- One session per block; the diagnostic workload differs from the performance prompt (the gate links them); 2 diagnostic launches per cell give weak non-overlap evidence.
- Diskbench is device-level; the 80B 2N vs 30B N pairing has about 5% more bytes in flight on the 80B side (mean read 499,712 vs 952,320 B).
- The matched-F region (3.92 F → 1.47 F) differs from the historical 80B region (5.97 F → 2.24 F).

#### Provenance and freeze semantics

- `protocol_freeze.json` records the pre-freeze HEAD `527540cee4e97b7ecc337c57f95d767f47798473`, the SHA-256 of this block (UTF-8, after CRLF→LF normalisation), of the machine-readable spec, of the `config.json` protocol object (parsed JSON without `measurements_made`), of the harness and tests (after CRLF→LF), and of `tools/exp022_run.py` (after CRLF→LF), whose per-launch validation functions EXP-026 reuses unchanged; plus the workload identity, model paths, runtime identity, seeds, expanded orders, the counts 36 / 60 and `measurements_made = false`.
- `--check` refuses any changed hash. `--run` additionally requires the protocol files committed, the pre-freeze HEAD as an ancestor, a clean tree (only append-only `llama-research-v2/VALIDATIONS.jsonl` lines for this experiment's `raw/` are tolerated), the earlier blocks complete, and the same HEAD for every block.
- Every scheduled launch and run is kept in `raw/<block>/records.jsonl` whatever its outcome. Outputs are written only under this experiment's `raw/`, `processed/` and `figures/`. The standard per-user Argine telemetry of the model launches is copied into `raw/` exactly as in EXP-022.
<!-- EXP-026 PROTOCOL-END -->

## Commands

```bat
python tools\exp026_run.py --check
python tools\exp026_run.py --dry-run
python tools\exp026_run.py --preflight --block diskbench
python tools\exp026_run.py --run --block diskbench
python tools\exp026_run.py --preflight --block perf
python tools\exp026_run.py --run --block perf
python tools\exp026_run.py --preflight --block diag
python tools\exp026_run.py --run --block diag
python tools\exp026_run.py --analyze
python -m unittest discover -s tests -p "test_exp026.py" -v
```

*Historical commands (2026-10-03), superseded by the harness above:*

```bat
argine research trace ... on both models
argine research analyze ...
```

## Raw data

<!-- EXP-026 CLOSURE-RAW-BEGIN -->
All three frozen blocks completed, on the frozen protocol commit `7168b936853b2c019e1c55000d81a11174a58682` (the same HEAD for every block, as recorded in each `session.json`). Machine-readable outputs, generated by the frozen `python -B tools/exp026_run.py --analyze` and kept as generated:

- `results.csv`: one row per scheduled run (60 diskbench, 24 performance, 12 diagnostic), with outcome and main values.
- `processed/summary.json`: performance cells, explanandum, traced gate, diskbench B_mix and device comparison, diagnostic mechanism values, candidate statuses, lifecycle and the RQ-009 mapping.
- `raw/diskbench/`: `session.json`, `records.jsonl`, run logs, and the isolated `argine-home/` telemetry.
- `raw/perf/` and `raw/diag/`: `session.json`, `records.jsonl`, launch logs; `raw/diag/launch-NN/run-00-<budget>/` hold the research traces.
- `llama-research-v2/VALIDATIONS.jsonl`: 12 append-only validation records emitted by the diagnostic launches (line endings normalised to LF at closure; content unchanged).

Sessions (UTC): diskbench 2026-10-15 20:59–21:05; perf 2026-10-15 21:10–21:55; diag 2026-10-16 19:05–19:44.

Provenance: runtime `llama-research-v2` (server SHA-256 `a0063720…`, BUILD-INFO `cc76046a…`, llama.cpp `254b1773…`); Qwen3-30B-A3B Q4_K_M and Qwen3-Next-80B-A3B-Instruct Q3_K_M from `C:\modelli`; diagnostic workload natural-language v1 EVAL (`71b170bc…`); performance prompt `argine.bench.DEFAULT_PROMPT` (`b1677835…`).
<!-- EXP-026 CLOSURE-RAW-END -->

## Result

<!-- EXP-026 CLOSURE-RESULT-BEGIN -->
[MEASURED] EXP-026 completed every scheduled run: diskbench 60/60 VALID, performance 24/24 VALID, diagnostics 12/12 VALID and 2/2-eligible in every cell. Nothing was retried, replaced or pruned.

**Primary explanandum (frozen rule; untraced, 4 generation threads, matched budget/F).**

| Quantity | 30B | 80B |
|---|---:|---:|
| P1 median tok/s (launches) | 6.25 (4.95, 6.25, 6.41) | 6.06 (5.11, 6.06, 6.06) |
| P2 median tok/s (launches) | 3.92 (3.63, 3.92, 3.96) | 3.73 (3.42, 3.73, 4.80) |
| β P1→P2 | 0.4756076 | 0.4947870 |
| β envelope (3 × 3 pairings) | [0.2275050, 0.5797407] | [0.0638067, 0.5832506] |

- [DERIVED] Δβ = β80 − β30 = +0.0191794.
- [DERIVED] The two envelopes overlap; the 30B envelope lies entirely inside the 80B envelope. **Primary classification = NOT DISTINGUISHED.**

Frozen closure wording:

> EXPLANANDUM NOT REPRODUCED. At matched budget/F (P1→P2, 4 generation threads, current runtime) the 80B's pressure-response exponent was not distinguishable from the 30B's. The historical regime difference did not reproduce in the matched-F region on the current runtime. This does not retrospectively falsify EXP-006 or alter F-004. Those were measured over different budget/F regions on earlier binaries and remain valid descriptions of those sessions. The candidate-mechanism hypothesis is INCONCLUSIVE: its object was absent at matched budget/F, so the candidates could not be discriminated against it. Secondary results are reported descriptively.

**Traced-slope agreement gate.**

| Quantity | 30B | 80B |
|---|---:|---:|
| β_traced (decode T, P1→P2) | 0.4960028 | 0.2265779 |
| traced envelope (2 × 2 pairings) | [0.4913977, 0.5005968] | [0.2041768, 0.2490282] |

- [DERIVED] Traced Δβ = −0.2694249. On its own the traced classification would be REPRODUCED; the untraced primary is NOT DISTINGUISHED and the sign differs.
- **Agreement gate = FAIL**, under the frozen rule: the sign of Δβ_traced differs from the primary Δβ.

**Candidate statuses (frozen rule: gate not PASS → every candidate NOT EVALUABLE).**

| Candidate | Status |
|---|---|
| (a) routing concentration | NOT EVALUABLE |
| (b) smaller slices / reads in flight | NOT EVALUABLE |
| (c) bookkeeping | NOT EVALUABLE |
| (d) different compute (non-residency, compute-inclusive) | NOT EVALUABLE |

No candidate is SUPPORTED, WEAKENED or FALSIFIED by EXP-026. The descriptive mechanism values below are not used to choose among the candidates.

**Lifecycle (frozen rule): INCONCLUSIVE**, because the primary explanandum is NOT DISTINGUISHED (and the gate failed).

**Secondary and descriptive results (no decision).**

- [MEASURED / DERIVED] R = ms/token(80B) / ms/token(30B), from cell medians at 4 threads: P1 1.0314; P2 1.0509; P3 0.7409. At P1 and P2 the medians are close, with the 80B slightly slower; at P3 (≈ 0.98 F) the 80B is faster (3.59 vs 2.66 tok/s). P3 was a secondary probe and is not a new primary result.
- [MEASURED] P3 medians at 4 threads: 30B 2.66 tok/s (2.60, 2.66, 3.04); 80B 3.59 tok/s (3.17, 3.59, 3.59).
- [DERIVED] Sensitivity, 30B at 12 threads (DESCRIPTIVE; NOT PART OF THE PRIMARY DECISION): P1 6.62 tok/s (5.13, 6.62, 6.62), P2 3.60 tok/s (3.48, 3.60, 4.09); β = 0.6210679, envelope [0.2309889, 0.6556320].
- [DERIVED] Run-order pattern in the performance block: each of the eight round-1 launches (the first eight launches of the session) is the lowest value of its cell, and free RAM before launch rose from 8.5 to 10.7 GiB across the session. Under the frozen rule all launches are kept, and the round-1 values contribute to the width of the β envelopes. No launch is excluded and no alternative β is computed.
- [DERIVED] Diskbench device comparison (B_mix, three rounds each, MB/s): INDETERMINATE overall.
  - 30B N=2 (2639–2813) vs 80B N=4 (3102–3211): favours the 80B;
  - 30B N=4 (3907–3973) vs 80B N=8 (3819–3876): disfavours the 80B;
  - 30B N=8 (4365–4374) vs 80B N=16 (4083–4100): disfavours the 80B.
  - Device-level only; it does not reproduce the store's asynchronous reads. Candidate (b) stays NOT EVALUABLE.
- [MEASURED] Traced decode step T (ms, launch means): 30B 150.2 / 149.8 (P1), 243.1 / 244.8 (P2), 323.5 / 328.6 (P3); 80B 172.4 / 176.6 (P1), 215.8 / 220.1 (P2), 283.0 / 282.1 (P3). Per-launch N/T, m, q bounds, E_r / E_nr bounds and depth are in `results.csv` and `processed/summary.json`; they are descriptive only.
<!-- EXP-026 CLOSURE-RESULT-END -->

## Interpretation

<!-- EXP-026 CLOSURE-INTERPRETATION-BEGIN -->
**What EXP-026 establishes.**

- On the current runtime, at 4 generation threads and matched budget/F (P1 = 3.92 F → P2 = 1.47 F), the untraced pressure-response exponents of the two models are not distinguishable under the frozen envelope rule.
- The historical observation (F-004: 80B flatter, 0.384 vs 0.50–0.58) therefore did not reproduce in the matched-F region on the current runtime. That observation came from equal absolute budgets 4 → 1.5 GiB, i.e. different budget/F regions for the two models, on earlier binaries.
- The result is consistent with a normalization/region difference. It does not establish one.
- The traced and untraced measurements disagree about the slope difference, so the frozen gate failed. Mechanism quantities from the traced block cannot be attributed to the untraced explanandum.

**What EXP-026 does not establish.**

- It does not establish that normalization or budget/F caused the historical difference.
- It does not establish that the two models behave the same: the envelopes are wide, and the secondary P3 ratio shows a large difference below F.
- It does not show that EXP-006 or F-004 was wrong. Both remain valid descriptions of their own sessions.
- It does not support, weaken or falsify any of candidates (a)–(d). All four are NOT EVALUABLE.
- It does not explain the traced/untraced disagreement. The protocol did not pre-specify a test for it.
<!-- EXP-026 CLOSURE-INTERPRETATION-END -->

## Limitations

The 80B uses Gated DeltaNet attention in 36 of 48 layers: compute differs too.

The limitations known before measurement are listed in the frozen protocol (Setup).

<!-- EXP-026 CLOSURE-LIMITATIONS-BEGIN -->
- [LIMITATION] **Traced/untraced disagreement.** The traced slope difference (−0.269) is opposite in sign to the untraced one (+0.019), so the frozen agreement gate failed. Contributing factors could include the different workloads (default prompt vs natural-language v1 EVAL), tracing cost, and the different sessions (perf on 2026-10-15, diag on 2026-10-16); EXP-026 did not test which.
- [LIMITATION] **Wide primary envelopes.** Three launches per cell and a run-order pattern (every round-1 launch is its cell's minimum) make the envelopes wide. NOT DISTINGUISHED is a statement under the frozen rule, not evidence of equal slopes.
- [LIMITATION] **One session per block, one runtime, one machine, one workload per block type.**
- [LIMITATION] **Diskbench is device-level** and does not reproduce the store's asynchronous reads; the device comparison was INDETERMINATE.
- [LIMITATION] **P3 and sensitivity results are descriptive.** They were never part of the primary decision and are not promoted to findings.
<!-- EXP-026 CLOSURE-LIMITATIONS-END -->

## Status history

2026-10-03 PLANNED.

2026-10-15 PLANNED: protocol EXP026-P1 written and frozen (`protocol_freeze.json`); no measurement has been made.

<!-- EXP-026 CLOSURE-HISTORY-BEGIN -->
- 2026-10-15: frozen protocol committed as `7168b936`. Diskbench (60/60 VALID) and performance (24/24 VALID) blocks run.
- 2026-10-16: diagnostic block run (12/12 VALID, all eligible). Measurements completed. Frozen analysis run (`--analyze`).
- 2026-10-16 CLOSED / INCONCLUSIVE.
  - Primary explanandum NOT DISTINGUISHED: β30 = 0.4756, β80 = 0.4948, Δβ = +0.0192, envelopes overlap.
  - Traced agreement gate FAIL (traced Δβ = −0.2694). Candidates (a)–(d): NOT EVALUABLE.
  - RQ-009 stays PARTIALLY ANSWERED, with a data note.
  - Documentary closure: protocol block, harness, analysis and results unchanged.
- 2026-10-16 amendment A1 (test-only, post-measurement).
  - The frozen suite exposed a lifecycle defect: `test_34_no_outputs_required_before_measurement` assumed the live repository stays in its pre-measurement state, so it failed once measurements existed (38/39).
  - A1 made only that test lifecycle-aware. The original freeze is archived byte-for-byte as `protocol_freeze_pre_test_amendment_2026-10-16.json`, and the active freeze records A1 and the new tests hash.
  - No scientific protocol, measurement, analysis, decision rule or result changed.
  - The amended targeted suite passes 39/39.
<!-- EXP-026 CLOSURE-HISTORY-END -->
