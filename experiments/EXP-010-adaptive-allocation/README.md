# EXP-010 — AdaptiveResidencyController (replay prototype) and ablations

| field | value |
|---|---|
| ID | `EXP-010-adaptive-allocation` |
| Status | **FALSIFIED** |
| Closure | CLOSED 2026-10-11 under frozen protocol EXP010-P1a: both models STATIC_BETTER_OR_EQUAL against the DEV-fitted, workload-family-informed static comparator (verdict for the surviving retrospective-holdout sub-question only) |
| Period | 2026-10-03 → 2026-10-11 |
| Origin | Argine2 |
| Related questions | RQ-001, RQ-006, RQ-007, RQ-008 |
| Supersedes / superseded | — |
| Protocol | EXP010-P1a, pre-execution amendment frozen 2026-10-11 before any EXP-010 replay. P1a supersedes P1 before execution; no scientific decision rule changed. Design: DEV-selected, frozen, EVAL-tested **retrospective holdout** (leakage class B). Executed and closed 2026-10-11 under P1a (selection SHA-256 `dfcc9895c338bb80…`). |

> **This is not a preregistration.** The EVAL traces and the EVAL outcomes of every adaptive candidate configuration were already inspected during EXP-007, so the new evaluation is a retrospective holdout. The primary new comparison is a DEV-selected adaptive controller against a DEV-fitted, workload-family-informed static allocation applied unchanged to EVAL.

## Research question

**Original question (2026-10-03, kept as written).** Can an online, interpretable controller that moves memory between layers by estimated marginal value outperform static allocations and approach the oracle?

**Surviving sub-question (2026-10-11).** Does a controller configuration selected using DEV only keep an advantage over static allocation when it is applied unchanged to EVAL? The registry verdict of EXP-010 applies to this sub-question only.

## Hypothesis

**Original hypothesis (2026-10-03, kept as written).** [HYPOTHESIS] The marginal value of a byte of expert memory differs between layers and over time, and an online controller can exploit it (requirement section 6).

What is already known before EXP-010 runs:
- **Already falsified for the current controller:** the *temporal-reaction* mechanism ([FH-029](../../FAILED_HYPOTHESES.md), EXP-007 C10).
  - On all four transition traces (30B and 80B, DEV and EVAL), the controller's own mean allocation applied as a fixed split gains ≥ 113% of the controller's gain.
  - The reaction-lag cost against instantly switched per-segment splits is negative in every case.
  - So the gain does not come from fast reaction to workload transitions. EXP-010 does **not** re-test or reclaim that mechanism.
- **Still testable:** whether the controller's advantage (a better general layer split, found online) survives when its configuration is selected on DEV and frozen before EVAL. That is a retrospective generalisation and performance question, and it is what the frozen protocol below tests.

## Setup

**Original setup (2026-10-03, kept as written).** Replay prototype exists (`argine/research/controller.py`): per-layer LRU partitions, online stack-distance utility monitors, epochs 32/64/128 tokens, hysteresis, optional pinning 0/25/50%. Evaluation protocol: constants tuned on DEV traces only, claims on EVAL traces (section 25); ablations: frozen, frequency estimator, no cost weighting, no hysteresis, no pinning. Runtime implementation only after ARGINE2-MEMORY-CHARACTERIZATION.md is complete.

### Frozen protocol EXP010-P1a (2026-10-11)
> **Pre-execution amendment P1a.** EXP010-P1 was frozen before execution. Independent final review identified terminology and validation-hardening issues only. P1a changes no candidate, controller parameter, metric, threshold, workload, budget, aggregation rule, DEV selection rule or registry closure rule. No EXP-010 DEV or EVAL replay/result had been computed before this amendment. The original P1 freeze is preserved unchanged as `protocol_freeze_P1_initial.json`.

The text between the two markers below is the frozen protocol. Its SHA-256 is computed over the UTF-8, LF-normalised text strictly between the markers and recorded in `protocol_freeze.json`, together with the analysis tool and the replay code (see "Protocol freeze record"). Text outside the markers (results, status) can change later without changing the protocol hash.

<!-- EXP-010 PROTOCOL-BEGIN -->

#### P1. Framing and leakage

- **Design.** DEV-selected, frozen, EVAL-tested retrospective holdout.
- **Leakage class: B — RETROSPECTIVE HOLDOUT.** This is not a preregistration, not a blind holdout and not an untouched holdout.
  - The EVAL traces were recorded and analysed in-sample in EXP-007.
  - The EVAL replay outcomes of all five candidate configurations below were tabulated and reported there (`analysis-v3/*/adaptive.csv`, `cross-cell-summary.csv`, EXP-007 C3–C6).
- **Descriptive only (already seen on EVAL):** adaptive vs global LRU, adaptive vs per-layer uniform, and adaptive vs 2Q. These are corroborative, not new confirmatory evidence.
- **Genuinely new:** the DEV-selected adaptive controller vs a DEV-fitted, workload-family-informed static allocation applied unchanged to EVAL. No DEV-fitted → EVAL static row has ever been computed.
- **Clean elements:**
  - the candidate grid and the fixed constants date from 2026-10-03, before any EVAL trace existed;
  - the DEV static fit and the DEV selection are mechanical;
  - every degree of freedom (objective, aggregation, selection level, closure rule) is fixed here, before the new EVAL comparison is computed.

#### P2. Controller (unchanged)

The existing `AdaptiveResidencyController` (`argine/research/controller.py`), not redesigned:
- per-layer LRU partitions;
- decayed online stack-distance utility estimates;
- value-per-byte reallocation every epoch;
- hysteresis;
- max-move cap;
- optional frequency × stability pinning;
- a uniform initial split.

It never sees the future and has no fitted state.

#### P3. Frozen DEV candidate space

Only these five configurations are eligible, as implemented by `argine research adaptive` (pinning exists only at epoch 64):

| id | `adaptive.csv` variant | epoch | pinned fraction |
|---|---|---|---|
| e32 | adaptive-e32 | 32 | 0 |
| e64 | adaptive-e64 | 64 | 0 |
| e128 | adaptive-e128 | 128 | 0 |
| e64-pin25 | adaptive-e64-pin25 | 64 | 0.25 |
| e64-pin50 | adaptive-e64-pin50 | 64 | 0.50 |

- **Selection level:** one configuration per model (30B, 80B). Never per workload, never per budget.
- The space is not expanded.

#### P4. Fixed parameters (not tuned)

- **Prototype defaults, kept:** decay 0.5, hysteresis 0.10, max_move 0.05, estimator = stack, cost_weighting on, stability_epochs 4, min_slots = top_k, uniform initial allocation, default `CostModel` (the diskbench curve recorded in each `analysis_manifest.json`).
- **Reaction-speed parameters are not re-tuned.** Their EVAL behaviour on transition traces is already known (FH-029).
- **Diagnostics only, never in selection:**
  - `ablate-frozen`, `ablate-freq-estimator`, `ablate-no-cost-weighting`, `ablate-no-hysteresis`, and `global-lru` within the `adaptive` output;
  - any other controller switch.
- **Pinning wording:** the original "no pinning" ablation is the pin-0 default, i.e. candidates e32 / e64 / e128.

#### P5. DEV selection objective and tie-break

- **Objective:** expert-event misses (decode, replay semantics v2, expert-unit, event-atomic), exact under the replay abstraction. No tok/s, no measured runtime throughput, no modelled stall.
- **Ratio:** for each model, candidate and DEV cell (7 workloads × 3 budgets = 21 equal-weight cells), ratio = misses(candidate) / misses(per-layer-lru-uniform), both from the same `adaptive` run.
- **G_DEV(candidate)** = geometric mean of the 21 ratios.
- **Selection:** the lowest G_DEV wins. Tie-break:
  1. lower G_DEV;
  2. if |ΔG| < 1e-4 (relative to the lowest G_DEV): fewer total bytes moved over the 21 cells;
  3. then the larger epoch;
  4. then no pinning.
- **No EVAL during selection.** `select` has no EVAL argument and reads only `dev/`.

#### P6. Static comparator (DEV-fitted, workload-family-informed)

- **Fit:** for every model × workload family × budget, `layer-aware-static` (exact DP on per-layer LRU miss curves, `allocator.dp_optimal_checked`) is fitted on the corresponding DEV trace.
- **Apply:** that exact allocation is frozen and applied unchanged to the corresponding EVAL trace, via `argine research compare EVAL --train DEV`. Example: 30B coding DEV at 2 GiB → static layer split → the same split replayed on 30B coding EVAL at 2 GiB.
- **Side information.** It receives workload-family information (it knows which family it is serving). It is a deliberately strong comparator, never called simply "deployable".
- **No-fitting static baseline:** per-layer uniform LRU.

#### P7. Baselines and references (EVAL)

- **Primary comparison / operational rows:**
  - global LRU — no fitting;
  - per-layer uniform LRU — no fitting;
  - 2Q — online baseline, no workload-family fitting;
  - the DEV-fitted workload-family-informed static allocation — fitted on the matching DEV workload family and frozen before EVAL; not a generic deployable baseline;
  - the DEV-selected frozen adaptive controller — online replay controller whose configuration is selected once per model on DEV.
- **References only, never competitors:**
  - EVAL-fitted static (`oracle-static-lru`): in-sample, an upper bound for static layer splits on that trace;
  - the Belady-style row: future-informed, a reference, not a proven bound for multi-key events with mixed sizes.
- **"Approach the oracle":** gets no SUPPORTED / FALSIFIED verdict, because the original experiment never defined a threshold for it. Only descriptive gap-closed shares ((LRU − X) / (LRU − Belady-style)) are reported.

#### P8. Decomposition (EVAL, per model)

1. global LRU → per-layer uniform: layer isolation / partitioning gain;
2. per-layer uniform → DEV-fitted static: workload-family-informed static allocation gain. **This stage uses DEV workload-family information.**
3. DEV-fitted static → DEV-selected adaptive: dynamic adaptive gain.
4. Also reported: per-layer uniform → adaptive, and global LRU → adaptive.

Each stage is a geometric-mean miss ratio over the 21 cells and per budget.

#### P9. Metrics

- **Primary:** expert-event misses.
- **Secondary:** expert-level hit rate; bytes read per token; stall ms/token, which is **MODELLED** (cost model) and never measured performance; moves; bytes moved.
- No tok/s.

#### P10. EVAL aggregation (per model; the two models are never collapsed into one headline)

- **G_pool** = geometric mean over the 21 EVAL cells of misses(adaptive, DEV-selected) / misses(static, DEV-fitted, family-matched).
- **W** = number of EVAL cells where the adaptive controller has fewer misses than the static comparator; exact ties count 0.5.
- **G_uniform** = geometric mean over the same 21 cells of misses(adaptive) / misses(per-layer uniform).
- **Also reported:**
  - G per budget;
  - per-workload ratios;
  - wins, losses and ties;
  - the transition and multi-turn subgroups;
  - adaptive vs global LRU and vs 2Q (descriptive);
  - a pooled two-model G (descriptive only).

#### P11. Closure rule EXP010-CR1 (directional; no effect-size threshold)

**Per model:**
- **ADAPTIVE_BETTER** if all hold: G_pool < 1.0, W > 10.5, and G_uniform ≤ 1.0.
- **STATIC_BETTER_OR_EQUAL** if both hold: G_pool ≥ 1.0 and W ≤ 10.5.
- **MIXED** otherwise. MIXED is an internal model-level label, not a repository status.

**Experiment verdict:**
- **SUPPORTED** if both models are ADAPTIVE_BETTER.
- **FALSIFIED** if both models are STATIC_BETTER_OR_EQUAL, or if G_uniform > 1.0 in both models (the adaptive controller loses in aggregate even to per-layer uniform).
- **INCONCLUSIVE** otherwise.

**Reporting rules:**
- Budget-specific reversals are reported explicitly. Not every budget needs the same sign.
- The 1% threshold proposed during the audit is **not** a registry criterion. Effect sizes of 0.5%, 1% and 2% are reported as descriptive sensitivity only and never alter the verdict.
- No new threshold is introduced after results are visible.

#### P12. Budgets and workloads

- **Budgets:** 4 GiB, 2 GiB and 1.5 GiB, all primary with equal cell weight. 4 GiB is not removed or demoted because previously seen results are weaker there; it is described as the lowest-pressure regime.
- **Workloads:** all seven families (natural-language, coding, reasoning-math, structured-json, italian, transition, multi-turn-v2). None is excluded after seeing outcomes.
- **Subgroups:** transition and multi-turn-v2 are pre-specified subgroups. They are reported separately and do not decide the status.

#### P13. DEV/EVAL pairs (verified 2026-10-11, metadata only)

- **Trace used:** `experiments/EXP-007-router-characterization/raw/<campaign>/run-00-4G/trace.jsonl.gz` of the original campaign folder, never a `*-replacement-*` folder.
- **Checked by `tools/exp010_analysis.py verify-pairs`** (full output and SHA-256 values in `protocol_pair_verification.csv`):
  - model;
  - workload family and version (identical SHA within each pair);
  - trace and run-00-4G existence;
  - trace loads with the current reader;
  - meta split;
  - manifest budget 4G;
  - EXP-021 closure and `qualified`;
  - multi-turn-v2 history semantics v2.
- Routing is identical across budgets (EXP-007), so one run-00-4G trace per cell serves every budget.

| model | family | DEV campaign | EVAL campaign | decode steps DEV / EVAL | DEV / EVAL trace SHA-256 (prefix) | note |
|---|---|---|---|---|---|---|
| 30B | natural-language | 30b-natural-language-dev-v2 | 30b-natural-language-eval-v2 | 381 / 381 | 69891597ae76 / 92d602fe764f | |
| 30B | coding | 30b-coding-dev | 30b-coding-eval-v2 | 381 / 381 | ea777c4ef9a7 / c0888e89263b | DEV recorded with raw instrumentation format v1 (EXP-021 validated it there); the router-trace file format is the same as every other cell |
| 30B | reasoning-math | 30b-reasoning-math-dev-v2 | 30b-reasoning-math-eval-v2 | 381 / 381 | 2fb5e9648a8c / 828e32fbf606 | |
| 30B | structured-json | 30b-structured-json-dev-v2 | 30b-structured-json-eval-v2 | 381 / 381 | 984c546e337a / 0190dac911f9 | original run-00-4G (the replacement folder replaced run-07 only) |
| 30B | italian | 30b-italian-dev-v2 | 30b-italian-eval-v2 | 381 / 381 | b5b18f46cffc / c5af75b6721a | |
| 30B | transition | 30b-transition-dev-v2 | 30b-transition-eval-v2 | 1196 / 1196 | 4dc1a0a97ac0 / 832b9a88716e | |
| 30B | multi-turn-v2 | 30b-multi-turn-v2-dev | 30b-multi-turn-v2-eval | 4939 / 4349 | c583c788731a / d5d670d6f0f5 | length mismatch; history semantics v2 both |
| 80B | natural-language | 80b-natural-language-dev-v2 | 80b-natural-language-eval-v2 | 381 / 381 | 7af0444d8880 / a1a1c612332a | |
| 80B | coding | 80b-coding-dev-v2 | 80b-coding-eval-v2 | 381 / 381 | 9d9fbe796387 / c833c90d4830 | |
| 80B | reasoning-math | 80b-reasoning-math-dev-v2 | 80b-reasoning-math-eval-v2 | 381 / 381 | 1d676a3b8682 / 005ec3266116 | |
| 80B | structured-json | 80b-structured-json-dev-v2 | 80b-structured-json-eval-v2 | 312 / 361 | f9029b8e82a0 / 42c6c8f9f5a5 | length mismatch (DEV ends early) |
| 80B | italian | 80b-italian-dev-v2 | 80b-italian-eval-v2 | 381 / 381 | 16145763c347 / ba5a8c79e2a3 | |
| 80B | transition | 80b-transition-dev-v2 | 80b-transition-eval-v2 | 1196 / 1125 | 55cba7b3c6d9 / 89f2a859210c | length mismatch |
| 80B | multi-turn-v2 | 80b-multi-turn-v2-dev | 80b-multi-turn-v2-eval | 3930 / 3120 | a2d3060bb449 / d0ff5d1f38ec | length mismatch; history semantics v2 both |

Length mismatches are recorded, not excluded.

#### P14. Code state and determinism

- **One frozen code state.** Every result that enters the decision is recomputed under it (SHA-256 values in `protocol_freeze.json`).
- **Why:** some `analysis-v3` DEV outputs (30B coding DEV, 80B coding DEV, 30B transition DEV) predate the 2026-10-04 12:40 change of `replay.py`.
- **Role of `analysis-v3`:** its values are used only in `processed/determinism_check.csv` (new vs prior misses, with differences visible). The new frozen run decides the result by protocol, whichever version looks better.
- **CLI provenance (tools/CLI-only, 2026-10-11, no runtime change).** `argine research compare` and `argine research adaptive` now also write `compare_manifest.json` / `adaptive_manifest.json`, recording the trace (and, for `compare --train`, the DEV training trace) by path, SHA-256 and metadata.
  - `select` checks every DEV manifest against the raw EXP-007 trace.
  - `analyze` checks every EVAL manifest, and that the static comparator was fitted on the matching DEV trace (`train_mode = separate`, train SHA-256 = DEV trace).

#### P15. Analysis tool and outputs

`tools/exp010_analysis.py`:
- **`check-freeze`:** verifies the protocol, tool and code hashes.
- **`select`:** DEV only; writes `selection.json` once and never overwrites it.
- **`show`:** prints the selection.
- **`script-eval-adaptive`:** generates the STEP 5 script from `selection.json`.
- **`analyze`:** refuses without a `selection.json` whose canonical hash verifies and whose provenance matches the frozen state.

The canonical hash is the SHA-256 of `json.dumps(selection, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` in UTF-8. Timestamps live outside the hashed payload.

Output structure:

```
experiments/EXP-010-adaptive-allocation/
  dev/<DEV campaign>/            adaptive.csv, adaptive_manifest.json, ... (STEP 1)
  selection.json                 (STEP 2)
  eval/<EVAL campaign>/          policy_comparison.csv, compare_manifest.json, ... (STEP 4)
  eval-adaptive/<EVAL campaign>/ adaptive.csv, adaptive_manifest.json (STEP 5)
  step5_eval_adaptive.bat        generated from selection.json (STEP 5)
  processed/                     dev_selection.csv, eval_table.csv, decomposition.csv,
                                 subgroups.csv, sensitivity.csv, determinism_check.csv,
                                 summary.json (STEP 6)
```

STEP 5 is explained once, in P16.

#### P16. Planned execution (Windows, Anaconda Prompt, from the repository root)

No new trace or fidelity run is required for EXP-010, because it reuses EXP-021-qualified EXP-007 traces. Commands use single `%` (interactive prompt). EXP-010 becomes RUNNING when STEP 1 is first executed.

```bat
cd /d C:\path\to\Argine2
set R=experiments\EXP-007-router-characterization\raw
set O=experiments\EXP-010-adaptive-allocation
python tools\exp010_analysis.py check-freeze
```

STEP 1 — DEV adaptive replay (14 jobs; the full grid plus diagnostics; selection reads only the five candidates and the uniform baseline):

```bat
for %D in (30b-natural-language-dev-v2 30b-coding-dev 30b-reasoning-math-dev-v2 30b-structured-json-dev-v2 30b-italian-dev-v2 30b-transition-dev-v2 30b-multi-turn-v2-dev 80b-natural-language-dev-v2 80b-coding-dev-v2 80b-reasoning-math-dev-v2 80b-structured-json-dev-v2 80b-italian-dev-v2 80b-transition-dev-v2 80b-multi-turn-v2-dev) do call argine.bat research adaptive %R%\%D\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --epochs 32,64,128 --pinned 0,0.25,0.5 --out %O%\dev\%D
```

STEP 2 — DEV selection, and STEP 3 — show it:

```bat
python tools\exp010_analysis.py select --dev %O%\dev --out %O%\selection.json
python tools\exp010_analysis.py show --selection %O%\selection.json
type %O%\selection.json
```

STEP 4 — EVAL static / baseline comparison (14 jobs; the static split is fitted on the matching DEV trace):

```bat
for %P in (30b-natural-language-dev-v2:30b-natural-language-eval-v2 30b-coding-dev:30b-coding-eval-v2 30b-reasoning-math-dev-v2:30b-reasoning-math-eval-v2 30b-structured-json-dev-v2:30b-structured-json-eval-v2 30b-italian-dev-v2:30b-italian-eval-v2 30b-transition-dev-v2:30b-transition-eval-v2 30b-multi-turn-v2-dev:30b-multi-turn-v2-eval 80b-natural-language-dev-v2:80b-natural-language-eval-v2 80b-coding-dev-v2:80b-coding-eval-v2 80b-reasoning-math-dev-v2:80b-reasoning-math-eval-v2 80b-structured-json-dev-v2:80b-structured-json-eval-v2 80b-italian-dev-v2:80b-italian-eval-v2 80b-transition-dev-v2:80b-transition-eval-v2 80b-multi-turn-v2-dev:80b-multi-turn-v2-eval) do for /f "tokens=1,2 delims=:" %a in ("%P") do call argine.bat research compare %R%\%b\run-00-4G\trace.jsonl.gz --budgets 4G,2G,1.5G --train %R%\%a\run-00-4G\trace.jsonl.gz --out %O%\eval\%b
```

STEP 5 — the selected adaptive configuration on EVAL (14 jobs; the script is generated from `selection.json`, so the configuration cannot be chosen by hand):

**STEP 5 runs the DEV-selected configuration of each model on every EVAL cell, even when the selected configuration is e64.** The `adaptive` command may additionally emit mandatory baseline/base rows (per-layer uniform, global LRU and, for a pinned selection, the base `adaptive-e64` row); only the frozen DEV-selected variant enters the primary adaptive comparison. `compare` also writes an `adaptive-e64` row, but `policy_comparison.csv` has no moves or bytes-moved columns, which are required secondary metrics. Duplicate deterministic rows produced by both `compare` and `eval-adaptive` (per-layer uniform, and `adaptive-e64` when present) are a hard consistency invariant: they are recorded in `determinism_check.csv` (stage `eval-internal`), and any miss-count mismatch, with no tolerance, blocks the analysis with `ProtocolError` before any model summary or registry verdict is produced.

```bat
python tools\exp010_analysis.py script-eval-adaptive --selection %O%\selection.json --out %O%\step5_eval_adaptive.bat
type %O%\step5_eval_adaptive.bat
call %O%\step5_eval_adaptive.bat
```

STEP 6 — final analysis, and STEP 7 — show results:

```bat
python tools\exp010_analysis.py analyze --selection %O%\selection.json --eval %O%\eval --eval-adaptive %O%\eval-adaptive --dev %O%\dev --out %O%\processed
type %O%\processed\summary.json
type %O%\processed\dev_selection.csv
type %O%\processed\decomposition.csv
```

#### P17. Limitations frozen before execution

- Retrospective holdout: the EVAL adaptive outcomes were previously observed (EXP-007). No untouched holdout exists.
- One hardware platform. The controller is replay-only; there is no runtime implementation.
- One routing trace per cell is used for policy replay. Routing repetitions were identical where checked, so repetitions add no policy-level replication.
- DEV/EVAL trace lengths differ for 30B and 80B multi-turn-v2, 80B structured-json and 80B transition.
- 30B coding DEV was recorded with raw instrumentation format v1.
- The static comparator receives workload-family information. Adaptive selection is per model, while static fitting is per family × budget.
- Single-turn traces (381 decode steps; 312 for 80B structured-json DEV) are short relative to the e64 / e128 epochs: about 6 / 3 controller decisions.
- Modelled stall is not measured runtime. Counterfactual replay is expert-unit and event-atomic, comparable with other counterfactual rows but not with the measured runtime (EXP-021).
- The Belady-style row is a reference, not a proven bound.

#### P18. Future untouched confirmation (not created now)

- **What it would require:**
  - a new EVAL2 prompt set, written and hashed after the EXP-010 selection is frozen;
  - new traces;
  - both models, if the aim is workload generalisation within the same model family.
- **Not enough:** re-splitting the existing prompts, because the existing traces and results have already been seen.
- **Optional:** a third model, which addresses model generalisation separately.

<!-- EXP-010 PROTOCOL-END -->

### Protocol freeze record

The canonical EXP010-P1a freeze is `protocol_freeze.json`, generated by:

`python tools/exp010_analysis.py freeze`

The original pre-amendment EXP010-P1 freeze is preserved unchanged as:

`protocol_freeze_P1_initial.json`

P1a supersedes P1 before the first EXP-010 replay. No scientific decision rule changed and no EXP-010 DEV/EVAL result existed when the amendment was made.

`python tools/exp010_analysis.py check-freeze` must print `frozen state OK` before STEP 1.

## Commands

Original command (2026-10-03): `argine research adaptive EVAL_TRACE --budgets 4G,2G,1.5G --out DIR`.

EXP-010 was executed and closed on 2026-10-11 under the frozen EXP010-P1a protocol, using STEPS 1–6 as defined in P16 above. The commands remain preserved there as the frozen execution procedure.

## Raw data

Pilot smoke test: `data/processed/pilot-legacy-30b-v2/adaptive.csv` (v1-legacy: `pilot-legacy-30b/adaptive.csv`, superseded). First 30B replay: `experiments/EXP-007-router-characterization/analysis-v2/run-00-4G/adaptive.csv`.

`results.csv` in this folder holds the historical 127-token pilot rows (2026-10-03). It is kept unchanged: no new `results.csv` was written at closure, so no archiving was needed. The closure results are in `processed/` (as for EXP-008).

EXP-010 inputs (read-only): the 28 run-00-4G traces listed in P13. EXP-010 outputs (2026-10-11, scientific evidence; do not regenerate): `dev/` (14 DEV `adaptive` runs with manifests), `selection.json`, `eval/` (14 `compare --train DEV` runs), `eval-adaptive/` (14 selected-configuration runs), `step5_eval_adaptive.bat` (generated from `selection.json`) and `processed/` (P15). Protocol files: `protocol_freeze.json` (P1a, canonical), `protocol_freeze_P1_initial.json` (P1, preserved) and `protocol_pair_verification.csv`.

## Result

### Frozen-protocol result (EXP010-P1a, executed and closed 2026-10-11)

[DERIVED, exact deterministic replay, counterfactual: expert-unit, event-atomic, decode, replay semantics v2. DEV-selected, frozen, EVAL-tested retrospective holdout, leakage class B.]

**Sources.** `selection.json` (canonical SHA-256 `dfcc9895c338bb80d429478872b73f24084b9612e9cea2b489d67bfb24bbfa51`), `processed/summary.json`, `processed/eval_table.csv`, `processed/decomposition.csv`, `processed/subgroups.csv`, `processed/sensitivity.csv`, `processed/determinism_check.csv`, and `dev/`, `eval/`, `eval-adaptive/`. `python tools/exp010_analysis.py check-freeze` printed `frozen state OK` before the analysis.

**DEV selection (DEV only).** G_DEV = geometric mean over 21 DEV cells of misses(candidate) / misses(per-layer uniform). Lower is better.

| candidate | 30B G_DEV | 80B G_DEV |
|---|---:|---:|
| e32 | 0.961341 | 0.992689 |
| e64 | 0.963114 | 0.993185 |
| e128 | 0.968289 | 0.994391 |
| e64-pin25 | 0.966327 | 1.003696 |
| e64-pin50 | 0.969643 | 1.030368 |

e32 has the lowest G_DEV in both models. It was selected independently for each, by G_DEV alone (no tie-break used).

**Primary EVAL comparison.** The DEV-selected adaptive controller vs the DEV-fitted, workload-family-informed static allocation, applied unchanged to EVAL. 21 equal-weight cells per model; a ratio below 1 means fewer misses for the adaptive controller.

| model | selected | G_DEV | G_pool adaptive/static | W (wins / losses / ties) | G_uniform adaptive/uniform | model classification |
|---|---|---:|---:|---:|---:|---|
| 30B | e32 | 0.961341 | 1.003747 | 4.0 / 21 (4 / 17 / 0) | 0.962839 | STATIC_BETTER_OR_EQUAL |
| 80B | e32 | 0.992689 | 1.001476 | 4.0 / 21 (4 / 17 / 0) | 0.991859 | STATIC_BETTER_OR_EQUAL |

**EXP010-CR1 verdict: FALSIFIED.**
- Both models are STATIC_BETTER_OR_EQUAL, because in both G_pool ≥ 1.0 (30B 1.003747, 80B 1.001476) and W ≤ 10.5 (4.0 in both).
- The frozen rule gives FALSIFIED when both models are STATIC_BETTER_OR_EQUAL.
- The uniform guard did not trigger: G_uniform < 1 in both models.
- The verdict applies **only** to the surviving retrospective-holdout sub-question. It is not a verdict on Argine2's central research question or on layer-aware allocation.
- "Approach the oracle" receives no verdict, because the original experiment never defined a threshold for it.

**Effect size (descriptive; does not alter the verdict).**
- The DEV-fitted static comparator has 0.37% fewer misses than the adaptive controller on the 30B and 0.15% fewer on the 80B (geometric means).
- In both models |G_pool − 1| < 0.5%, so the primary difference is small. `sensitivity.csv` reports the 0.5 / 1 / 2% bands as descriptive only.
- The pooled two-model G = 1.002611 is descriptive only.

**Per budget, adaptive vs DEV-fitted static.** No budget reverses the aggregate sign in either model. All four 30B wins are at 4 GiB.

| model | budget | G adaptive/static | adaptive wins | G adaptive/uniform |
|---|---|---:|---:|---:|
| 30B | 4G | 1.001338 | 4 / 7 | 0.951317 |
| 30B | 2G | 1.004051 | 0 / 7 | 0.971052 |
| 30B | 1.5G | 1.005859 | 0 / 7 | 0.966260 |
| 80B | 4G | 1.001601 | 1 / 7 | 0.990663 |
| 80B | 2G | 1.001277 | 2 / 7 | 0.996937 |
| 80B | 1.5G | 1.001549 | 1 / 7 | 0.987998 |

**Decomposition (EVAL, geometric-mean miss ratio over all 21 cells; below 1 = fewer misses in the destination policy).**

| stage | 30B | 80B |
|---|---:|---:|
| 1. global LRU → per-layer uniform (layer isolation / partitioning) | 0.951852 | 0.976732 |
| 2. per-layer uniform → DEV-fitted static (uses DEV workload-family information) | 0.959245 | 0.990397 |
| 3. DEV-fitted static → DEV-selected adaptive (dynamic adaptive gain) | 1.003747 | 1.001476 |
| direct: per-layer uniform → DEV-selected adaptive | 0.962839 | 0.991859 |
| direct: global LRU → DEV-selected adaptive | 0.916481 | 0.968780 |

- **30B:** per-layer uniform substantially improves over global LRU (−4.8% misses). The DEV-fitted static split improves further over uniform (−4.1%). The adaptive controller then loses slightly to the DEV-fitted static comparator (+0.37%).
- **80B:** the same ordering, with materially smaller effects: −2.3%, then −1.0%, then +0.15% for adaptive vs static.

**Pre-specified subgroups (descriptive; they do not decide the status).** The adaptive controller loses to the DEV-fitted static split in all 3 cells of each subgroup in both models:
- transition: G_static 30B 1.006513, 80B 1.001690;
- multi-turn-v2: 30B 1.005654, 80B 1.001953.

**Descriptive comparisons already seen on EVAL during EXP-007** (corroborative, not new evidence):
- adaptive vs global LRU: G 30B 0.916481, 80B 0.968780;
- adaptive vs 2Q: G 30B 0.912112, 80B 0.969945;
- adaptive vs per-layer uniform: as above.

**Oracle reference (descriptive only).** Share of the LRU-to-Belady-style miss gap closed, minimum–maximum over the 21 cells:
- 30B: adaptive 6.7–28.1%, DEV-fitted static 6.7–28.9%, EVAL in-sample static (reference) 9.2–29.3%;
- 80B: adaptive −2.0–14.0%, DEV-fitted static −1.2–14.0%, EVAL in-sample static (reference) 2.8–16.1%.

The Belady-style row is a future-informed reference, not a proven bound.

**Determinism / validation.** `determinism_check.csv` has 1050 rows, 0 non-identical:
- 462 DEV `adaptive` rows vs `analysis-v3`, including the three DEV cells that predated the 2026-10-04 `replay.py` change;
- 420 EVAL `compare` rows vs `analysis-v3`;
- 126 EVAL `adaptive` rows vs `analysis-v3`;
- 42 compare ↔ eval-adaptive rows (the hard P1a invariant).

No `ProtocolError` occurred.

### Pre-protocol history (kept as recorded)

The rows below are pre-protocol history and do not decide the EXP-010 status.

No result yet (DEV trace only, in-sample comparisons, no EVAL split). [DERIVED, smoke test] On the 127-token pilot the controller reallocates (23-73 moves) and stays within 2.5% of the uniform split's misses; with epochs of 64-128 tokens there are only 1-2 decisions in 127 tokens, so the pilot cannot test the idea.

[DERIVED, EXP-007 run-00-4G DEV trace, replay v2, same trace semantics for every row] adaptive-e64 expert-event hit 76.9 / 57.2 / 50.6% at 4 / 2 / 1.5 GiB vs per-layer uniform 75.7 / 56.1 / 49.1% (+1.2 / +1.1 / +1.4 pp) and vs the exact in-sample optimal static split 77.4 / 57.9 / 51.3% (below it by 0.4-0.8 pp). [LIMITATION] No comparison with the measured runtime is made: the controller is expert-unit and event-atomic; its runtime-equivalent counterpart is not built (EXP-021).

[DERIVED, EXP-007 30B transition DEV trace, one session, counterfactual, 2026-10-04] Write-up: [docs/research/TRANSITION-ADAPTIVE-GAIN.md](../../docs/research/TRANSITION-ADAPTIVE-GAIN.md).
- adaptive-e64 gains 0.88 / 1.20 / 1.59 pp over the frozen uniform split at 4 / 2 / 1.5 GiB.
- A fixed allocation equal to its own time-averaged allocation gains MORE: 118–135% of that.
- The 64 steps after each transition hold no more than their share of the gain, and the transition transients are not shortened.
- Oracle bounds on the same trace:
  - best fixed split: 1.34 / 1.64 / 2.02 pp;
  - per-segment optima switched instantly add 0.36 / 0.38 / 0.26 pp on top of that, which this controller does not capture. Its reaction takes 157–297 of 299 steps per segment, limited by epoch, decay, max-move and hysteresis.
- Pinning helps in stationary stretches and hurts after transitions, because the pins are stale previous-phase experts.
- Hysteresis reduces churn by 5–33% at no measurable cost.
- Cost weighting is a no-op with the current cost model.

[DERIVED, EXP-007, 28 cells, in-sample, 2026-10-11] What EXP-007 already established, before this protocol ([EXP-007 README](../EXP-007-router-characterization/README.md)):
- **C3:** adaptive-e64 improves over global LRU in most constrained regimes, and the improvement grows as the budget shrinks. At 2 and 1.5 GiB the fixed uniform per-layer partition already recovers most of that gap.
- **C4 / C5:** the adaptive benefit is model- and workload-dependent, and the pattern replicates across DEV/EVAL pairs. That is pattern replication, not out-of-sample validation.
- **C6:** pinning is context-dependent.
- **C9:** cost weighting contributes little (±0.05 pp).
- **C10 / FH-029:** the gain is a better general split, not fast temporal reaction.
- **C11:** the Belady-style reference stays far above the controller.
- **C12:** every static row is in-sample.

## Interpretation

**Status: FALSIFIED, for the surviving retrospective-holdout sub-question only.** In this DEV-selected, frozen, EVAL-tested retrospective holdout, a DEV-selected online adaptive controller does **not** keep an advantage over the DEV-fitted, workload-family-informed static allocation on EVAL, in either model.

**What this does not falsify:** layer-aware resource allocation.
- The adaptive controller still beats global LRU (G 0.916481 / 0.968780) and per-layer uniform (G 0.962839 / 0.991859) in aggregate, especially on the 30B.
- The decomposition shows that this gain is explained at least as well by layer isolation plus a layer-aware static split as by online reallocation.
- This is consistent with EXP-007 C10 / [FH-029](../../FAILED_HYPOTHESES.md): the controller's gain was already shown to be a better general split, not fast temporal reaction.

**Where the evidence points the architecture:**
- **first-order:** layer isolation / partitioning;
- **a strong practical research baseline:** workload-family-informed static allocation;
- **second-order and not justified by EXP-010:** dynamic online reallocation.

**Scope:**
- The primary static-over-adaptive differences are small (+0.37% misses on the 30B, +0.15% on the 80B). FALSIFIED follows from the directional closure rule, which has no effect-size threshold.
- These are replay miss counts. They are not a runtime or throughput result; no tok/s claim is made, and modelled stall is not measured.
- The DEV-fitted static row uses workload-family information. It is not a generic deployable baseline, and it is not claimed to be universally optimal: it is the frozen comparator defined by the protocol.
- H-3 ("an online controller can track drifting marginal value and beat static splits on evaluation traces") remains open only for a differently designed controller.

## Limitations

Requires traces of thousands of tokens with workload transitions (EXP-023) and a DEV/EVAL split. Fidelity of replay vs runtime: EXP-021.

[LIMITATION, 2026-10-11] The limitations of the frozen protocol are listed in P17. The most important is that EVAL is a retrospective holdout: the EVAL adaptive outcomes were inspected during EXP-007.

[LIMITATION, 2026-10-11, closure]
- **Retrospective holdout (leakage B).** The EVAL adaptive candidate outcomes had already been seen during EXP-007. The result is stronger than an in-sample comparison, but it is not an untouched confirmation.
- **Small primary effects.** Static vs adaptive differs by +0.37% misses on the 30B and +0.15% on the 80B.
- **Scope.** One hardware platform; counterfactual replay only; no runtime controller implementation; modelled stall is not measured throughput.
- **Side information.** The static comparator knows the workload family. Adaptive selection is per model, while static fitting is per family × budget.
- **Single-turn length.** Single-turn traces are short relative to the controller epochs, though the selected e32 makes about 12 decisions per 381-step trace.
- **One routing trace per cell.**

## Status history

2026-10-03 PLANNED (controller code exists for replay; deliberately not wired into the runtime). 2026-10-11: audit and protocol EXP010-P1 frozen before any EXP-010 replay. Design: DEV-selected, frozen, EVAL-tested retrospective holdout (leakage B). `tools/exp010_analysis.py` and its tests were added, and `compare` / `adaptive` now write provenance manifests (tools/CLI only). Status stays PLANNED. It becomes RUNNING when STEP 1 (the first DEV replay) is executed under this frozen protocol.

2026-10-11: pre-execution amendment EXP010-P1a supersedes P1 before the first replay. Independent final review corrected comparator terminology, hardened deterministic EVAL consistency into a blocking invariant, and clarified STEP-5 wording. No candidate, controller parameter, metric, threshold, workload, budget, aggregation rule, DEV selection rule or closure rule changed. No EXP-010 DEV/EVAL replay or result had been computed. The original P1 freeze is preserved as `protocol_freeze_P1_initial.json`. Status remains PLANNED.

2026-10-11: EXP-010 executed and closed under frozen protocol EXP010-P1a (STEPS 1–6; the RUNNING interval during execution was not recorded separately in the registry). DEV selected e32 independently for both 30B and 80B (selection SHA-256 `dfcc9895c338bb80d429478872b73f24084b9612e9cea2b489d67bfb24bbfa51`). On EVAL, both models classified STATIC_BETTER_OR_EQUAL against the DEV-fitted workload-family-informed static comparator (30B G_pool=1.003747, W=4.0; 80B G_pool=1.001476, W=4.0). EXP010-CR1 therefore yields FALSIFIED. The adaptive controller remained better than per-layer uniform in aggregate (G_uniform=0.962839 / 0.991859), so the result falsifies the surviving adaptive-over-static sub-question, not layer-aware allocation itself. Determinism check: 1050 rows, 0 non-identical. `check-freeze`: frozen state OK; the protocol, tool and freeze files are unchanged. **CLOSED, FALSIFIED.**
