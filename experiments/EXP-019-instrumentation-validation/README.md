# EXP-019 — Research instrumentation: correctness and completeness

| field | value |
|---|---|
| ID | `EXP-019-instrumentation-validation` |
| Status | **SUPPORTED** |
| Period | 2026-10-03 |
| Origin | Argine2 |
| Related questions | RQ-011, RQ-012 |
| Supersedes / superseded | — |

> **Audit scope (2026-10-12).** Three evidence sources are kept apart:
> - (a) the 2026-10-03 fixture / Linux evidence: C++ sanitizer tests and the tiny-MoE end-to-end run; `results.csv` summarises this source only;
> - (b) the 2026-10-03 Windows smoke run: not publication-quality; it found the termination tail loss (FH-024);
> - (c) the fresh publication-quality Windows validation of 2026-10-12 (`raw/windows-validation-2026-10-12-r2`, `processed/windows-validation-2026-10-12-r2`).
>
> Status remains SUPPORTED, with workload-completeness / tested-path scope. Independent review completed on 2026-10-12; the modern audit is CLOSED / SUPPORTED at workload-completeness / tested-path scope.

## Research question

Does the Argine2 instrumentation (patch 0003) record every routing decision, slice access, read, eviction and node, without changing what the runtime does?

## Hypothesis

[HYPOTHESIS] Tracing is pure observation: identical hits/misses/evictions and identical model output with and without the trace; every miss has exactly one read record; the budget invariant holds.

## Setup

(1) Standalone C++ test `patches/argine-research-test.cpp` against a store built from the tiny MoE fixture, under ThreadSanitizer and ASan/UBSan; existing store tests (79 checks) and stress test re-run with tracing on. (2) End-to-end: patched llama-server built natively on Linux from the same sources (llama.cpp 254b177), tiny MoE fixture, store budget between one and two layers, two prompts x 40 tokens.

(3) **Fresh Windows validation, 2026-10-12** (publication-quality).
- **Binary:** the cross-compiled research binary `llama-research-v2` (llama-server SHA-256 `a0063720…`, i.e. `llama-research-v2@a0063720`; BUILD-INFO SHA-256 `cc76046a…`; llama.cpp 254b177; trace format v2; instrumentation `argine2-research-v2`).
- **Machine:** the reference machine (i5-1340P, Windows 11 build 26200, 16 GB).
- **Model and store:** Qwen3-30B-A3B Q4_K_M with expert store `Qwen_Qwen3-30B-A3B-Q4_K_M.psxs`, 4 GiB expert budget, asynchronous reads, 16 cache shards, PRETOUCH off.
- **Workload:** `coding` v1, DEV split (3 prompts × 128 tokens), seed 42, threads 4 / threads-batch 12, ctx 4096, with warm-up and flush-sentinel requests.
- **Repository:** Argine2 commit `3a0df91…`, clean tracked tree.
- **Fidelity check:** runtime/replay fidelity was checked offline with `tools/validate_replay_fidelity.py` (format v2, recorded shard map, per-thread pin model).

## Commands

```bat
g++ -fsanitize=thread argine-store.cpp argine-research.cpp argine-research-test.cpp -o arxtest && ./arxtest tiny.psxs 2 8 trace.raw
ARGINE_EXPERT_STORE=tiny.psxs ARGINE_EXPERT_BUDGET=275251 ARGINE_RESEARCH_TRACE=e2e.raw llama-server -m tiny.gguf
```

Fresh Windows validation (2026-10-12): the `argine research trace` command line is not archived. The manifest records the llama-server arguments, budget, workload, split, seed and read mode. The fidelity outputs come from `tools/validate_replay_fidelity.py` (usage `RUNS_DIR OUT_DIR`); its exact invocation was not archived either.

## Raw data

`raw/e2e-fixture/raw.tsv` (end-to-end raw log) and derived tables in `processed/e2e-fixture/`.

*Added 2026-10-12:*
- `raw/windows-validation-2026-10-12-r2/run-00-4G/`: `raw.tsv`, `markers.jsonl`, `manifest.json`, `summary.json`, `responses.jsonl`, derived CSVs and `trace.jsonl.gz`; per-run summary in `raw/windows-validation-2026-10-12-r2/runs.csv`. Kept unchanged.
- `processed/windows-validation-2026-10-12-r2/`: `fidelity_runs.csv`, `fidelity_runs.json`, `divergence-run-00-4G.json`.
- `raw/windows-validation-2026-10-12/` and `processed/windows-validation-2026-10-12/`: an earlier attempt on the same day. Its run directory is empty and its `fidelity_runs.json` is `[]`. It holds no data and is not evidence; it is kept unchanged.
- `results.csv` is the 2026-10-03 fixture summary only (the tiny-MoE end-to-end run). It contains no Windows rows and was not changed.
- Experiment `raw/` folders are excluded from git (`.git/info/exclude`), so the raw traces are local evidence, not version-controlled.

## Result

*Fixture / Linux evidence (2026-10-03):*

[MEASURED] All instrumentation checks pass under TSan and ASan/UBSan (stats identical with/without trace; 1:1 miss↔read; ordered timestamps; every eviction logged; 0 budget violations with 6 threads). End-to-end: 82 forward passes (2 prompts + warm-up), 164 router records, 3069 slice accesses, 71 reads (100% exact completions on the POSIX path), 38 evictions, 492 node records, 81 snapshots, 0 budget violations; runtime counters (71 misses) equal the trace. Generated text identical with no store, store, and store + trace.

[MEASURED, Windows smoke run 2026-10-03, not publication-quality: no git commit] Qwen3-30B-A3B Q4_K_M, 4 GiB budget, `coding` dev split (3 prompts × 128 tokens), binary `llama-research` ee463158: instrumentation confirmed by the runtime, 1,453,486 slice accesses, 144,508 reads, 0 failed reads, 0 malformed records, 0 budget violations; 66.6% of completions exact (the rest upper bounds), so queue depth is reported as an upper bound (time-weighted mean while I/O active ≤ 8.9, over the full span ≤ 4.8). [MEASURED] The trace lacked the last ~263 ms of the run: the final request has 126 of 127 decode steps plus one partial step (16 of 48 layers), although the file was well-formed — records buffered after the last 250 ms flush were lost when the process was terminated. Fixed in the harness (flush sentinel request + per-request completeness check), without rebuilding the binary.

*Fresh Windows validation (2026-10-12, publication-quality):*
- [MEASURED] **Provenance.**
  - Argine2 commit `3a0df9146fcac2fed8894f3cbb248ac4f586c478`, `dirty = false` (tracked files; untracked run output ignored).
  - llama-server SHA-256 `a0063720…`; BUILD-INFO SHA-256 `cc76046a…`.
  - Model `Qwen_Qwen3-30B-A3B-Q4_K_M.gguf` and store `Qwen_Qwen3-30B-A3B-Q4_K_M.psxs` fingerprinted (`sha256-fast`). The store opened by the runtime is the fingerprinted one.
  - Workload `coding.json` SHA-256 `460949e7…` (v1, DEV).
  - Read mode requested = effective = `asincrono` (trace C record; stderr agrees).
  - `missing = []`, `publication_quality = true`.
- [MEASURED] **Trace.**
  - 1,474,158 slice accesses, 149,269 reads (0 failed), 144,768 evictions, 56,920 nodes, 395 snapshots.
  - 0 malformed records, no truncated line, 0 budget violations (peak resident 4,290,637,824 B ≤ 4 GiB).
  - Workload-only hit rates: expert-level 0.70560096, runtime-style 0.90186699. All misses are in gate nodes.
  - Every workload request is complete: 127 / 127 decode steps for each of the 3 prompts, 0 partial steps.
  - Generated-text provenance complete (3 / 3).
- [MEASURED] **Completion stamps.**
  - The Windows overlapped-I/O path ran: `exact_completion_share` = 0.63404324. 100% exact completion timing is not claimed.
  - Queue depth is therefore `upper_bound`: time-weighted mean ≤ 10.27 while I/O is active, ≤ 5.43 over the span, maximum 112.
- [DERIVED] **Replay fidelity** (format v2, 16 shards, recorded shard map, per-thread pin model, threads 4 / threads-batch 12).
  - Reconstruct inconsistencies 0.
  - Over the workload (1,420,470 requests), runtime and replay are identical: misses 139,395 = 139,395; evictions 137,049 = 137,049; 0 outcome, victim or skip-count disagreements.
  - Over the full lifetime they are also identical: misses 149,269 = 149,269; evictions 144,768 = 144,768; event disagreement rate 0; replay failures 0; replay budget violations 0.
  - `workload_fidelity` PASS, `full_trace_fidelity` PASS, `exp021_closure` PASS.
- [MEASURED] **Termination tail: INCOMPLETE.**
  - 5 per-shard sequence numbers are missing in 3 shards (3, 4, 10). All are inside the flush-sentinel window, after each shard's last workload record, and are classified replay-irrelevant (0 shards with replay-relevant gaps).
  - One MoE node of the flush sentinel (graph 395, layer 13) lacks 12 slice requests. 0 incomplete nodes are in the workload.
  - 0 tail disagreements over 14,853 tail requests.
- [MEASURED, context only] **Throughput.** Generation 6.1166291 tok/s (3 × 128 tokens). This is not an overhead measurement: there is no untraced comparison (EXP-024). Minimum available RAM during the run was 747,122,688 bytes.

## Interpretation

[SUPPORTED, workload-completeness / tested-path scope] Observation-only behaviour is directly validated on the Linux fixture. On the fresh Windows validation, the instrumentation is complete and runtime/replay event-exact over the measured workload; record loss is confined to the post-workload flush-sentinel tail. Windows completion timestamps are partially exact, so queue-depth timing remains an upper bound for this run.

- **What each path shows.** Observation-only behaviour (identical statistics and identical generated text with and without the trace) was tested on the fixture and in the C++ sanitizer test (Linux). The Windows validation shows completeness and event-exact runtime/replay agreement over the workload. It has no trace-off control, so it does not itself re-test on/off identity on Windows.
- **Workload completeness, not termination completeness.** `trace_complete = True` and `publication_quality = True` mean the measured workload is complete: per-request decode-step counts match llama.cpp's `predicted_n`, all layers are present, and the warm-up and flush sentinel are not checked. They do not mean the process trace is complete through termination: the same run's termination tail is INCOMPLETE.
- **Flush sentinel.** The flush-sentinel protocol protected the measured workload in this run. It does not make the whole server lifetime record-complete, and FH-024 stands.
- **Overhead.** Tracing overhead is not measured here (EXP-024, RQ-012).

*[Historical interpretation (2026-10-03), superseded by the 2026-10-12 audit: "[DERIVED] The instrumentation is complete and observation-only on the tested paths. The Windows overlapped-I/O completion stamping (COMPLETE_EXACT via WaitForMultipleObjects) compiles but has NOT been executed: it runs only on Windows." The Windows path was in fact first exercised by the 2026-10-03 smoke run (66.6% exact completions) and again on 2026-10-12 (63.4%).]*

## Limitations

*[Historical limitation (2026-10-03), superseded: "[LIMITATION] Tiny model; Linux POSIX read path; the Windows binary in llama-research/ is cross-compiled and untested at runtime. Overhead not measured (EXP-024)." The Windows binary has since run (smoke run 2026-10-03; EXP-021 / EXP-007 v2 runs since 2026-10-04; this validation 2026-10-12).]*

1. [LIMITATION, audit 2026-10-12] Fixture evidence: tiny model, Linux POSIX synchronous read path. On/off identity of statistics and output is established there, not on Windows.
2. [LIMITATION, audit 2026-10-12] Windows validation scope: one run, one model (30B), one budget (4 GiB), one workload and split (coding DEV, 3 prompts × 128 tokens), one machine, one read mode (`asincrono`). Other Windows models, budgets and workloads are covered by the EXP-021 / EXP-007 fidelity runs, not by this validation.
3. [LIMITATION, audit 2026-10-12] Termination tail incomplete: 5 lost records and 1 incomplete node in the flush sentinel. The whole-process trace is not complete through termination, so completeness claims are workload-scoped.
4. [LIMITATION, audit 2026-10-12] Exact completion share 0.634: queue-depth values are upper bounds, and the all-reads latency figures include bounded completions (an `exact_only` subset is reported separately in `summary.json`).
5. [LIMITATION, audit 2026-10-12] No trace-off control on Windows: behaviour preservation on Windows is inferred from event-exact replay, not from an on/off comparison of outputs or statistics.
6. [LIMITATION, audit 2026-10-12] Overhead not measured (EXP-024, RQ-012). 6.1166291 tok/s is not overhead evidence.
7. [LIMITATION, audit 2026-10-12] Backend provenance. The harness resolved `auto -> vulkan (Intel Iris Xe)` and passed `--n-gpu-layers 49 --n-cpu-moe 48`. It then warned that the binary enumerated no Vulkan device, and printed `backend=vulkan`, which is the resolved backend, not the effective one. The manifest has no backend field, and the console log is not archived. [DERIVED] The effective execution was CPU: the BUILD-INFO CMake flags enable no Vulkan backend, and the static binary depends on no `vulkan-1.dll`. A prospective fix (2026-10-12) makes the runner record `runtime_flags.backend` {requested, resolved, vulkan_devices_enumerated, effective}. The recorded run is not rewritten.
8. [LIMITATION, audit 2026-10-12] `llama-research-v2/BUILD-INFO.json` still says `"runtime_tested": false` / "Not executed on Windows before delivery". That is immutable delivery-time build provenance. Runtime validation is recorded in `VALIDATIONS.jsonl`, which has entries since 2026-10-04, including `20261012T072226Z-a0063720-run00` (publication_quality true).
9. [LIMITATION, audit 2026-10-12] Machine state: available RAM fell to 747,122,688 bytes during the run. This is not a correctness issue, but it bears on the throughput figure.
10. [LIMITATION, audit 2026-10-12] The harness and fidelity command lines are not archived; the manifest and outputs are.
11. [LIMITATION, audit 2026-10-12] The raw traces are excluded from git (local evidence, not version-controlled).
12. [LIMITATION, audit 2026-10-12] The first 2026-10-12 attempt left an empty run directory; it contributes no evidence.

## Status history

2026-10-03 SUPPORTED on the fixture. 2026-10-03 (later) Windows overlapped-I/O path executed in a smoke run (see Result); tail-loss defect found and fixed in the harness; complete publication-quality validation pending a rerun after `git init`.

2026-10-12: fresh publication-quality Windows validation (`windows-validation-2026-10-12-r2`; commit 3a0df91, clean; binary llama-research-v2@a0063720). SUPPORTED retained with workload-completeness / tested-path scope: the workload is complete and replay is event-exact over it. The termination tail is INCOMPLETE (5 records and 1 node lost, flush sentinel only). The exact completion share is 0.634, so queue depth is an upper bound. Overhead remains EXP-024. Independent review completed; modern audit CLOSED / SUPPORTED.
