# Research history

Meaningful conceptual changes only, not every commit. Dates from the raw logs (`data/raw/argine-v1/runs.jsonl`) and Argine's README.

| date | change | evidence |
|---|---|---|
| 2026-09-27 | Project starts as **PicoStar16**: "a bigger compressed model knows more" on 16 GB. Memory Planner refuses configurations that would not fit. | Argine README |
| 2026-09-27 | Repack buffer disabled by default on this CPU (slower and +3 GiB). | EXP-002 |
| 2026-09-27 | Router logging (patch 0001) and GGUF reorder by co-activation; the historical comparison reports 4.69 vs 5.17 tok/s, with the tested reordered layout slower (wording narrowed by the 2026-10-11 audit). | EXP-016 |
| 2026-09-27 | diskbench: page faults 310 MB/s vs direct reads ~1300 MB/s — the path, not the device. | EXP-003 |
| 2026-09-27 → 29 | **Expert Store** (patch 0002) replaces uncontrolled mmap residency: hard budget, sharded LRU, unbuffered reads. Four silent concurrency bugs found by tests. | Argine README, patches/README.md |
| 2026-09-30 | Perplexity measurement design corrected: the initial synthetic/templated comparison reported +2.06% for the historical IQ2_XXS. The same old quantized model measured +36.3% relative PPL on the later real-prose test (which also used 60 instead of 18 chunks), while a new IQ2_XXS quantization calibrated on separate real prose measured +26.0%. The initial general negligible-damage verdict was superseded (wording narrowed by the 2026-10-11 audit). Thread sweep on the old build: 4 threads best for the MoE. | EXP-017, EXP-001 |
| 2026-10-01 | Matched-plan binary change: 7.45 → 12.97 tok/s (1.74×). The llama.cpp version, OpenMP and linkage were confounded, so this establishes binary identity as an experimental variable rather than isolating one build flag (wording narrowed by the 2026-10-11 audit). Quality/bit curve: IQ2_S (2.56 expert bpw) is the descriptive empirical elbow among the tested Qwen3-30B quantization points on the recorded PPL corpus/signature (wording narrowed by the 2026-10-11 audit). | EXP-011, EXP-013 |
| 2026-10-02 | Dense 14B control: the larger compressed 30B showed no advantage on the matched-text perplexity proxy; the original "bigger compressed knows more" hypothesis is retained as FALSIFIED only in that operationalized sense. A six-turn same-session conversation test showed in-RAM prefix reuse holding observed TTFT to 4.78–5.62 s while cumulative prompt footprint grew to 1467 tokens; the historical disk-KV project was cancelled for that ordinary conversation use case. Persistent/restart/cross-conversation disk caching was not tested (wording narrowed by the 2026-10-11 audit). Project renamed **Argine**. | EXP-012, EXP-014 |
| 2026-10-02 → 03 | Speed vs budget curve on the 30B (exponent 0.50-0.58); PRETOUCH was identified as an unlogged historical configuration contaminant; the preserved Argine v1 counter summary is consistent with substantial read-path timing displacement, while the wider 2.67x-span fits (0.54 ± 0.03 without, 0.58 ± 0.06 with) did not reproduce the earlier large slope-change interpretation. PRETOUCH attribution/counters are not independently raw-reconstructable (wording narrowed by the 2026-10-11 audit). Cost per miss falls as budget shrinks (queue depth). | EXP-005, EXP-015 |
| 2026-10-03 | **Asynchronous expert reads** (emit-and-continue); one-binary comparison shows the benefit is regime-dependent (−0.6% … +10.8%). | EXP-004 |
| 2026-10-03 | **80B experiment falsifies the residency-only scaling model**: predicted 4.0-5.0, measured 6.87 tok/s; exponent 0.384. Hard memory ceiling verified on a second architecture. | EXP-006 |
| 2026-10-03 | **Argine2 begins** as a research fork: requirement to turn the project into an open research artifact around expert-memory allocation. | ARGINE2 requirement |
| 2026-10-03 | Argine's raw logs were imported (sanitised) and 16 historical experiments reconstructed in Argine2 from the surviving Argine v1 evidence. Most are re-derived from raw logs where provenance permits; EXP-015 is a provenance-limited exception whose PRETOUCH attribution and per-miss counters survive only in the historical README. The re-reading downgrades the thread rule (EXP-001), finds that the historical ≤5% reproducibility assumption is unsafe: nominally matched legacy records show ≈15-16% cross-run range spreads and a supplementary 80B comparison differs by ≈19%, but missing runtime flags prevent interpreting this as pure session variance (EXP-018). *[Historical wording, narrowed by the 2026-10-11 audit: "finds 15-19% between-session dispersion (EXP-018)."]* | data/raw/argine-v1 |
| 2026-10-03 | Research instrumentation (patch 0003) and the `argine research` toolchain (trace format, characterization, event-atomic replay, oracle, elasticity, allocators, replay controller, predictor, report). Pinned llama.cpp 254b177 (last commit before the buffer-type API change of 2026-10-02). | EXP-019 |
| 2026-10-03 | End-to-end validation on a test model reveals that the runtime's hit counter includes guaranteed re-pins (expert-level ≈ runtime miss × 3). | EXP-020 |
| 2026-10-03 | First Windows research trace (30B, 4 GiB, coding dev): instrumentation and asynchronous store confirmed on Windows; runtime/expert-level hit rate 90.06% / 70.17%; tail loss at termination found. Provenance fixes: resolved and fingerprinted store, requested vs effective read mode, CPU identity, trace-derived final statistics, machine-state RAM, append-only validation registry, queue-depth status, flush sentinel. | EXP-019, EXP-020 |
| 2026-10-03 | Pilot replay of the legacy 30B trace: oracle gap ≫ layer-split effect; LRU collapse predicted below one token's expert footprint. Recorded as prediction, not result. | EXP-009, EXP-022 |
| 2026-10-03 | First 30B campaign slice: coding dev, 4 / 2 / 1.5 GiB × 3. Runtime curve stable to ≤ 0.2 pp; routing bit-identical across all nine runs. The replay of the runtime's policy disagrees with it by 3.5-4.8 pp, and the "DP-optimal" static split is worse than uniform. Campaign stopped. | EXP-007 |
| 2026-10-03 | **Replay fidelity validation.** The gap is mostly metric: the runtime counts gate-slice hits over prefill + decode, the replay counted expert hits over decode only. It is compounded by prefill not being served to the replay state, and by slice granularity and sharding. New runtime-reproduction replay (`store-replay`), equivalent event-for-event to the real store (0 disagreements on v2 traces), and within 0.3 pp on all nine 30B runs. Instrumentation format v2 (shard id, per-shard sequence, undo). The allocator's 512 KiB grid had made uniform infeasible: now exact, with an invariant. Replay semantics v2: all steps served, counted phase only. Categories separated (runtime reproduction / counterfactual / in-sample / oracle). Earlier analyses kept and marked superseded. | EXP-021, [REPLAY-FIDELITY-VALIDATION](research/REPLAY-FIDELITY-VALIDATION.md) |
| 2026-10-04 | **First Windows format-v2 validation fails closure** (5.29% event disagreement, 3 shards with sequence gaps).
- **Divergence:** the replay's pins were node-scoped for every thread. In the runtime they are per thread, and Windows runs prompt and decode graphs on different thread pools (`--threads 4 --threads-batch 12`).
- **Gaps:** 5 hits lost from the file buffer at termination, inside the flush sentinel.
- **After the fix:** per-thread pins give 0 differences on that trace. Validator checks added: sequence-gap classification and node completeness. Linux had passed because it used one pool.
- **Status:** EXP-021 back to RUNNING pending a fresh Windows run. | EXP-021, [REPLAY-FIDELITY-VALIDATION §13](research/REPLAY-FIDELITY-VALIDATION.md#13-windows-v2-follow-up-2026-10-04) |
| 2026-10-04 | **EXP-021 closes.** A fresh 30B Windows run replays exactly. The 80B coding-dev campaign (9 runs) is exact over the workload; 2 runs differ only in the flush-sentinel tail, in shards that lost records at process termination. Fidelity is now reported as three verdicts: full trace (strict, diagnostic), workload, and termination-tail completeness. | EXP-021, EXP-007, [§14](research/REPLAY-FIDELITY-VALIDATION.md#14-independent-confirmation-and-the-80b-campaign-2026-10-04) |
| 2026-10-04 | 30B transition campaign (9 runs, fidelity-valid). The temporal analysis shows the adaptive controller's gain is a better general layer split, not a reaction to workload transitions: a fixed split does better, and transients are not shortened. Pinning flips sign with stationarity. `adaptation_steps` saturates at its 16-step floor under pressure and is replaced by absolute transient metrics. The workload routing of the nine repetitions is identical, so policy-level replication needs new sessions. | EXP-007, EXP-010, EXP-023, [TRANSITION-ADAPTIVE-GAIN](research/TRANSITION-ADAPTIVE-GAIN.md) |
| 2026-10-04 | **Generated-text provenance fixed.** A smoke test showed SHA-256("") for every response despite 128 generated tokens. Cause: the harness hashed only `message.content`, but with `--jinja` and the default reasoning format Qwen3-30B-A3B's tokens come back in `reasoning_content` (an unclosed thinking block); a bare `except` turned every gap into "". Fix: explicit extraction across response formats, documented UTF-8 hashing of reasoning + content (text not stored), per-record status, and failures surfaced in the publication check. Historical 30B runs stay provenance-limited; routing and cache results are unaffected. | [TEXT-PROVENANCE](research/TEXT-PROVENANCE.md) |
| 2026-10-10 | **Multi-turn v2 history semantics.** After the provenance fix, the v1 multi-turn workload was found to send an empty assistant turn back to the model whenever a thinking model gave no final content. New workload `multi-turn-v2` (`ff89429e96c0`):
- the history is the non-empty final `content` only, and `reasoning_content` never enters it (it stays in provenance);
- an empty final content stops and invalidates the sample;
- `n_predict` is 1024 per turn (corrected 2026-10-11: frozen at 1536, `e978bde0341f`, after the 1024 smoke run truncated every turn; the 1024 file is preserved at `workloads/superseded/`);
- per-turn history audit fields and an EXP-007 eligibility verdict appear in the summary, the manifest and the publication check.

v1 is preserved unchanged and superseded for the final comparison; transition keeps v1 semantics. | EXP-007, [MULTITURN-V2-HISTORY](research/MULTITURN-V2-HISTORY.md) |
| 2026-10-11 | **EXP-007 closed (SUPPORTED).** 28 / 28 cells (2 models × 7 workload families × DEV/EVAL), each cell-budget with 3 EXP-021-qualified runs.
- Routing concentration is not a model-agnostic predictor of allocation opportunity. The 80B is more concentrated but leaves far less to gain from re-splitting memory across layers (in-sample uniform excess 0.5–3.9% vs 3.5–10.4% on the 30B). Within each model there is an exploratory association at 4 GiB.
- Adaptive allocation beats global LRU under memory pressure, mostly through per-layer partitioning. It is not universally best (2Q at high memory on the 80B) and stays far below the future-informed Belady-style reference. Whether an online policy can realise that headroom is open.
- Pinning helps consistently only in multi-turn. Epoch size, hysteresis and cost weighting barely move hits.
- FH-029 is replicated on four transition traces.
- All analyses are in-sample; out-of-sample allocation is the next open step. | EXP-007, [EXP-007 README](../experiments/EXP-007-router-characterization/README.md) |
| 2026-10-11 | **Historical audit of EXP-001…006 closed.** Statuses unchanged: EXP-001 INCONCLUSIVE, EXP-002–005 SUPPORTED, EXP-006 FALSIFIED.
- Claims narrowed to their tested scope: EXP-002 repack on this CPU and build; EXP-003 on the reference machine and access pattern; EXP-005 exponents descriptive, "no memory ceiling" narrowed.
- EXP-004's fixed (non-rotated) mode order corrected where it had been described as interleaved.
- EXP-003's 2026-10-01 session cited.
- EXP-006 reinterpreted with EXP-007.
- No reruns required. | EXP-001…006, [HISTORICAL-AUDIT-EXP001-006](research/HISTORICAL-AUDIT-EXP001-006.md) |
| 2026-10-11 | **EXP-027 closed: the generation-thread optimum is model-dependent.** Current CPU-only research binary, 4 GiB Expert Store, 24 independent rotated launches, all valid, with full post-run fingerprints.
- 30B: 12 threads CLEAR BEST (6.74 tok/s, +3.5% over 4).
- 80B: 4 threads CLEAR BEST (6.91 tok/s, +2.0% over 12).
- The pre-registered predictions (one thread per P-core; 12 slowest) are FALSIFIED (FH-030). No fourth round was needed.
- RQ-014 answered for the tested configuration. EXP-001 stays INCONCLUSIVE. | EXP-027, RQ-014, [EXP-027 README](../experiments/EXP-027-current-thread-sweep/README.md) |
| 2026-10-11 | **EXP-009 closed as INCONCLUSIVE** (methodological): its hypothesis, "layer-aware allocation closes a significant part of the LRU-to-oracle gap", never defined "significant", and no post-hoc threshold is applied.
- Supported findings from the 28-cell replay: layer-aware allocation recovers a non-trivial but minority, model- and budget-dependent share of the gap to the future-informed Belady-style reference.
- 30B 1.5 GiB: 18–23% (per-layer LRU), 27–32% (in-sample layer-aware static). 80B 4 GiB: −4 … +9%.
- Even the in-sample split leaves at least about two thirds of the gap.
- The adaptive controller and the DEV → EVAL test belong to EXP-010. | EXP-009, [EXP-009 README](../experiments/EXP-009-policy-replay/README.md) |
| 2026-10-11 | **EXP-008 analysed and closed as INCONCLUSIVE.** Deterministic analysis of existing EXP-007 data, with a plan frozen before any predictor-comparison statistic (not a preregistration).
- Setup: length-matched working sets (first 320 decode tokens), per model, EVAL primary, DEV replication.
- **Throughput:** the models disagree. The 80B favours budget / Denning working set on all three measures; the 30B favours resident fraction on all three.
- **Expert-level hit rate:** the working-set predictors beat resident fraction in both models on every measure (supported within the traced dataset). DEV replicates both.
- No new inference. | EXP-008, RQ-002, [EXP-008 README](../experiments/EXP-008-working-set/README.md) |
| 2026-10-11 | **EXP-010 protocol EXP010-P1 frozen** before any EXP-010 replay; status stays PLANNED.
- Design: DEV-selected, frozen, EVAL-tested retrospective holdout (leakage B). The EVAL adaptive outcomes were already inspected in EXP-007.
- Candidates: e32, e64, e128, e64-pin25 and e64-pin50, with one configuration selected per model by the geometric-mean DEV miss ratio vs per-layer uniform. Reaction-speed constants are fixed.
- Primary comparison: the DEV-selected adaptive controller vs a DEV-fitted, workload-family-informed static split applied unchanged to EVAL.
- Closure rule: directional, with no effect-size threshold. The original "over time" mechanism is not re-tested (FH-029).
- Tooling: `tools/exp010_analysis.py`; `compare` / `adaptive` now write provenance manifests. | EXP-010, [EXP-010 README](../experiments/EXP-010-adaptive-allocation/README.md) |
| 2026-10-11 | **EXP-010 executed and closed as FALSIFIED** under frozen protocol EXP010-P1a (DEV-selected, frozen, EVAL-tested retrospective holdout, leakage B; P1a was a pre-execution amendment with no rule change).
- DEV selected e32 for both models.
- EVAL: the adaptive controller had slightly more misses than the DEV-fitted, workload-family-informed static split (G 30B 1.003747, 80B 1.001476; W = 4.0 / 21 in both), so both models are STATIC_BETTER_OR_EQUAL.
- It still beat per-layer uniform (G 0.962839 / 0.991859) and global LRU. Layer partitioning plus a fitted static split explains the gain; online reallocation is second-order (FH-031).
- Determinism: 1050 rows, 0 non-identical. Replay miss counts only, no throughput claim. | EXP-010, FH-031, [EXP-010 README](../experiments/EXP-010-adaptive-allocation/README.md) |
| 2026-10-11 | **Historical audit of EXP-011 (first of EXP-011…018).** SUPPORTED retained.
- Title and wording narrowed to "matched model and runtime plan, different binaries". The 1.74× (7.45 → 12.97 tok/s) is an observed contrast under the matched recorded plan, with llama.cpp version, OpenMP and linkage confounded.
- The conclusion concerns binary identity, not a specific build flag.
- No new measurement; raw records and `results.csv` unchanged. | EXP-011, FH-009, [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-11 | **Historical audit of EXP-012.** FALSIFIED retained for the proxy-scoped hypothesis.
- Matched-text PPL: 14B 13.2766 ± 0.3176; 30B IQ2_S 13.6283 ± 0.3290; no predicted 30B advantage observed.
- Equal assistant quality/knowledge was not established.
- Historical throughput 3.83 vs 12.97 tok/s is an observed ~3.39x contrast, not a causal sparse-vs-dense effect because binary and session differ.
- No new measurement; raw records and results.csv unchanged. | EXP-012, FH-004, [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-11 | **Historical audit of EXP-013.** SUPPORTED retained with narrowed descriptive scope.
- Audited four-point Qwen3-30B curve (same PPL signature): Q4_K_M 12.0642, IQ2_S 13.6283, IQ2_XS 14.3578, re-calibrated IQ2_XXS 15.1946.
- Adjacent slopes ≈0.67 / 2.92 / 3.35 PPL per expert bit: a descriptive elbow at IQ2_S (2.56 expert bpw). No formal knee or optimum is established.
- The IQ2_XXS calibration identity (15.1946 re-calibrated vs 16.4396 first) comes from historical text, not from the PPL record.
- The over-inclusive historical extraction (both IQ2_XXS records and a Qwen3-14B row) is documented, not removed.
- No new measurement; raw records and results.csv unchanged. | EXP-013, [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-11 | **Historical audit of EXP-014.** SUPPORTED retained with narrowed operational scope.
- One six-turn Qwen3-30B-A3B IQ2_S benchmark, with the same cumulative prompt footprint under both conditions (188 → 1467 tokens).
- With reuse TTFT was 4.78–5.62 s; without reuse it rose from 5.11 to 40.12 s. The turn-6 observed contrast is ≈7.8x.
- The disk KV cache itself was not benchmarked. The project cancellation is an engineering decision, and the ">100x" figure a historical estimate.
- One invocation, with no rotation or replication.
- No new measurement; raw records and results.csv unchanged. | EXP-014, FH-016, [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-11 | **Historical audit of EXP-015.** SUPPORTED retained with provenance-limited mechanism scope.
- Read/miss without vs with PRETOUCH: 1.233 vs 0.932 ms (4 GiB) and 1.063 vs 0.773 ms (1.5 GiB). The read reductions, 0.301 / 0.290 ms, compare with explicit new-page costs of 0.370 / 0.395 ms: ≈81% / 73% correspondence.
- Throughput 6.14 vs 5.90 tok/s and 3.61 vs 3.32 tok/s, so PRETOUCH was slower at both endpoints.
- Wider-span exponents 0.54 ± 0.03 vs 0.58 ± 0.06. No formal slope-equivalence claim.
- The PRETOUCH flag and the per-miss counters are absent from the raw logs; results.csv is sourced from the historical README; the Origin field was corrected.
- No new measurement. | EXP-015, FH-014, [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-11 | **Historical audit of EXP-016.** FALSIFIED retained with historical-attribution and tested-layout scope.
- The reorder event is preserved: `runs_prima=7.555`, `runs_dopo=4.403`, `guadagno=1.716` (field names kept; not relabelled "mean run length"), 192 tensors, 17,603,493,888 bytes.
- The historical condition comparison is 4.69 vs 5.17 tok/s (reordered slower). The selected raw benchmark groups (medians 5.17 / 4.57 / 4.43) do not independently identify original vs reordered, and 4.69 is one repetition, not the 22:45 group median.
- The page-fault evidence is contextual, not causal.
- No new measurement. | EXP-016, FH-008, [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-11 | **Historical audit of EXP-017.** SUPERSEDED retained with measurement-design scope.
- Synthetic: 3.4258 vs 3.4963 (+2.06%). Old IQ2_XXS on real prose: 12.0642 vs 16.4396 (+36.3%). Recalibrated IQ2_XXS on real prose: 12.0642 vs 15.1946 (+26.0%).
- The original +2.06% numerical measurement is retained; only its general quality verdict is superseded.
- The synthetic and real signatures differ in chunk count (18 vs 60) as well as text. The global raw log preserves the real-prose imatrix/quantize lineage. No factorial causal decomposition.
- No new measurement. | EXP-017, FH-019, [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-11 | **Historical audit of EXP-018.** FALSIFIED retained with recorded-configuration / legacy-provenance scope.
- Maximum within-run CV ≈6.43% (0.06433166), the provenance-cleanest evidence against the 1-4% expectation.
- 30B nominal-key groups: 15.36% (`78ee4096`, 4 GiB) and 16.04% (`757a17a4`, 6 GiB) min-max spreads; read mode unrecorded and PRETOUCH not preserved.
- 80B 7.10 -> 5.74 is a supplementary raw contrast (≈ −19.2% relative to 7.10), not a results.csv group.
- The historical builder grouping omits read mode; the mixed read-mode `3cac5531` rows and the confounded 8B row are excluded from reproducibility inference.
- The CSV `sessions` column is `len(rr)`, not session IDs.
- The ~15% threshold is narrowed to a conservative legacy guardrail.
- No new measurement. | EXP-018, FH-020, [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-11 | **Historical audit block EXP-011 … EXP-018 closed.** All eight experiments were audited; no registry status changed; claims and provenance were narrowed where required; no historical experiment was rerun; immutable measurements remain unchanged. | [HISTORICAL-AUDIT-EXP011-018](research/HISTORICAL-AUDIT-EXP011-018.md) |
| 2026-10-12 | **EXP-019 modern audit CLOSED / SUPPORTED: fresh publication-quality Windows validation.** SUPPORTED retained with workload-completeness / tested-path scope (30B, 4 GiB, coding DEV, llama-research-v2@a0063720, commit 3a0df91).
- The workload is complete (127 / 127 decode steps per request) and replay is event-exact over it.
- The termination tail is INCOMPLETE: 5 records and 1 node lost, in the flush sentinel only. FH-024 stands.
- The exact completion share is 0.634, so queue depth is an upper bound.
- The manifest records no backend. Auto resolved to vulkan, but the CPU-only binary enumerated no device, so execution was effectively CPU; backend recording is fixed prospectively.
- Overhead remains EXP-024. Independent review completed; EXP-019 audit CLOSED / SUPPORTED. | EXP-019, FH-024 |
| 2026-10-12 | **EXP-020 modern audit CLOSED / SUPPORTED** (tested Argine2 expert-store / hit-rate-semantics scope).
- The fixture mechanism evidence is preserved unchanged.
- EXP-021 confirms the real-30B gate-node semantics.
- EXP-007 confirms the miss ratio 3.000 in all 252 qualified 30B/80B runs.
- The field `expert_level_hit_rate` is defined as the workload-only gate-slice rate; the whole-expert `expert_event` metric remains distinct.
- The ≈74% 80B figure is superseded as an estimate.
- No new measurement was performed. Independent review completed; modern audit CLOSED / SUPPORTED. | EXP-020, EXP-021, EXP-007, FH-021 |
| 2026-10-12 | **EXP-021 modern audit CLOSED / SUPPORTED** (tested-path / format-v2 workload-fidelity scope).
- No new run during the audit.
- The 2026-10-04 evidence is preserved: the FAILED first Windows v2 validation, the per-thread correction on the same trace, the fresh 30B confirmation (0 / 1,474,817), and the 80B campaign (9 / 9 workload-exact, 7 / 9 full-trace exact).
- EXP-019's 2026-10-12 run is a later independent 30B confirmation.
- The v1 shard-map limitation, the wait/undo paths, equal-size pools and recorded-interleaving limits remain. | EXP-021, EXP-019, RQ-011 |
| 2026-10-12 | **EXP-024 closed: INCONCLUSIVE** (prospective protocol EXP024-P1; enabled vs disabled instrumentation on the same `llama-research-v2` binary).
- The reviewed freeze was committed in `22ef3f4`. Session `20261012T113829-5edc0f3f` ran from that clean commit: 20 / 20 launches valid, no retries, final fingerprints valid.
- 4G: mean paired overhead 1.099%, CI95 [0.605, 1.593], so < 2% is supported at 4G.
- 1.5G: 0.549%, CI95 [−1.379, 2.478], inconclusive.
- The frozen global rule gives INCONCLUSIVE. No post-hoc extra repetitions. | EXP-024, RQ-012 |
| 2026-10-12 | **EXP-023 closed: FALSIFIED for the current controller** (retrospective, analysis-only synthesis of the four EXP-007 transition sessions; no new model run).
- Verdict from the conjunctive hypothesis: its clause "adaptive allocation shortens the transient" is already falsified by FH-029, now reproduced in 12 / 12 trace × budget rows. Independent of JSD.
- [DERIVED, retrospective operationalisation] Per-layer boundary JSD is 2.84–4.02× the within-segment reference (median per trace): descriptive positive evidence of workload-dependent routing, not confirmatory.
- Signed boundary transient: adaptive signed `excess64` lower than frozen in 11 / 36 comparisons (not by itself a shorter transient); 42 negative signed `excess64` values reported with their sign.
- Not a falsification of workload dependence or of layer-aware allocation. | EXP-023, FH-029, RQ-004, RQ-006 |
<!-- EXP-022 CLOSURE-BEGIN -->
| 2026-10-15 | **EXP-022 closed: INCONCLUSIVE formal H-5 status; extreme-pressure envelope measured.**
- All 45 prospectively frozen launches completed; 30B was 25/25 VALID and completed down to 512 MiB.
- At the frozen H-5 point (30B / 768 MiB), both eligible replicas gave a mean workload gate-slice expert-level hit rate of 0.2322%, far below the historical >30% falsifier. The prediction is NOT FALSIFIED BY THIS TEST, but no positive support threshold was pre-specified, so formal status is INCONCLUSIVE.
- Mechanism and usability separate under pressure: on 30B, expert reuse fell from 16.19% near the one-token footprint (1024 MiB) to 0.232% at 768 MiB and 0% at 512 MiB, while mean generation remained 2.71 and 2.63 tok/s.
- Exploratory 80B showed the same qualitative reuse transition; 512 MiB yielded two DEGRADED completions and one causally unresolved STORE_RESIDENCY_ABORT.
- All 18 replay-fidelity diagnostics were run post-session and remain diagnostic only. | EXP-022, RQ-008 |
<!-- EXP-022 CLOSURE-END -->
<!-- EXP-026 CLOSURE-BEGIN -->
| 2026-10-16 | **EXP-026 closed: INCONCLUSIVE; historical slope difference not reproduced at matched budget/F.**
- Protocol EXP026-P1 was frozen 2026-10-15 (commit `7168b936`). Measurements 2026-10-15 → 2026-10-16: diskbench 60/60, performance 24/24 and diagnostics 12/12 VALID; no retry, no replacement.
- Primary explanandum NOT DISTINGUISHED: untraced, 4 generation threads, P1 → P2 at matched budget/F, β30 = 0.4756, β80 = 0.4948, Δβ = +0.0192, envelopes overlap. Consistent with a normalization/region difference; EXP-006 and F-004 are not retrospectively falsified.
- Traced-slope agreement gate FAIL (traced Δβ = −0.2694): candidates (a)–(d) are all NOT EVALUABLE. RQ-009 stays PARTIALLY ANSWERED, with a data note. | EXP-026, RQ-009 |
<!-- EXP-026 CLOSURE-END -->
