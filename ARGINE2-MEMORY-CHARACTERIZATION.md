# ARGINE2 — Memory characterization

> **Status note (2026-10-11).** [EXP-007](experiments/EXP-007-router-characterization/README.md) is closed: 28 / 28 cells (both models, seven workload families, DEV and EVAL), EXP-021-qualified. Every "[OPEN] awaiting EXP-007" marker below now means *the data exist and the v1 write-up of this section is pending*. The measured results are in the EXP-007 README and `experiments/EXP-007-router-characterization/analysis-v3/`. This v0 text is kept as written.

**Version 0 (2026-10-03). First deliverable of Argine2 (requirement section 59). This version contains the methodology, the platform, the instrumentation and its validation, the historical evidence re-derived from Argine's raw logs, and a pilot on the only router trace that exists (legacy, 30B, 127 tokens). Every per-model section that needs Phase-1 runs is marked **[OPEN] awaiting EXP-007** with the exact command that fills it. The adaptive controller will not be designed for the runtime before v1 of this document.

Contents: [1 Methodology](#1-methodology) · [2 Machine](#2-machine-configuration) · [3 Binaries](#3-binary-fingerprints) · [4 Models](#4-model-metadata) · [5 Routing distributions](#5-routing-distributions) · [6 Per-layer entropy](#6-per-layer-entropy-and-concentration) · [7 Expert frequency](#7-expert-frequency-distributions) · [8 Reuse](#8-reuse-distances) · [9 EWS](#9-effective-working-sets) · [10 I/O timeline](#10-io-timeline) · [11 Cache behaviour](#11-cache-behaviour) · [12 Elasticity](#12-memory-elasticity) · [13 30B vs 80B](#13-30b-vs-80b) · [14 Supported](#14-supported-hypotheses) · [15 Falsified](#15-falsified-hypotheses) · [16 Open](#16-open-questions)

---

## 1. Methodology

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). In short: router traces recorded by the runtime itself (patch 0003), event-atomic replay, expert-level hit rates, queue depth from per-request timestamps (labelled exact only when every completion is exact, otherwise upper bound), dev/eval separation, ≥3 interleaved repetitions, complete fingerprints, and every statement tagged.

Phase-1 protocol for v1 of this document, per model (30B Q4_K_M, 80B Q3_K_M):

```bat
set ARGINE_LLAMA_DIR=%CD%\llama-research
for %W in (natural-language coding reasoning-math structured-json italian multi-turn transition) do ^
  argine research trace MODEL.gguf --expert-store --budgets 4G,2G,1.5G --repeat 3 ^
     --workload %W --split dev --out experiments\EXP-007-router-characterization\raw\MODEL-%W-dev
:: then the same with --split eval
argine research analyze TRACE --budgets 4G,3G,2G,1.5G,1G,768M,512M --train DEV_TRACE --out ...
```

## 2. Machine configuration

[MEASURED] Windows 11 (build 26200); Intel Core i5-1340P, 4 P-cores + 8 E-cores (16 logical), AVX2 without AVX-512/AMX; 16 GB RAM (15.72 GiB usable); NVMe; Intel Iris Xe iGPU, no dedicated GPU; CPU backend for every result.

[MEASURED] Storage ([EXP-003](experiments/EXP-003-direct-io/README.md)): sequential 8 MiB reads 3305-3678 MB/s; 896 KiB reads at one reader: page faults 310 MB/s (2.96 ms), unbuffered direct reads 1254-1324 MB/s (0.69-0.73 ms); direct reads with 2 / 3 readers: 2899 / 3711 MB/s.

## 3. Binary fingerprints

| binary | used for | SHA-256 / label |
|---|---|---|
| `llama-noomp@3d91ef83`, `@3cac5531`, `@e007d93d`, `@78ee4096` | historical 30B/80B results (Argine v1) | first 8 hex of the binary hash, as recorded in `runs.jsonl` |
| `llama-research/llama-server.exe` | Phase 1 (Argine2) | see [`llama-research/BUILD-INFO.json`](llama-research/BUILD-INFO.json): llama.cpp `254b177307b1`, patches 0001+0002+0003, MinGW-w64 13 posix, static, `GGML_OPENMP=OFF`, AVX2/FMA/F16C/BMI2 |

[LIMITATION] The research binary is a new build: its results are not directly comparable with Argine v1 numbers (EXP-011: builds can differ by 1.74×). Phase 1 re-measures the baselines on it.

## 4. Model metadata

[MEASURED, from GGUF metadata recorded by Argine]

| | Qwen3-30B-A3B Q4_K_M | Qwen3-Next-80B-A3B-Instruct Q3_K_M |
|---|---|---|
| file | 18.63 GB | 38.30 GB |
| parameters total / active | 30.53 B / 3.35 B | 79.67 B / 3.87 B |
| experts / active | 128 / 8 | 512 / 10 (+ shared expert) |
| layers with MoE | 48 | 48 |
| expert bytes / non-expert bytes | 16.35 GiB / ~1.0 GiB | 34.31 GiB / 1.35 GiB |
| per-token expert bytes / reads (all misses) | 1046 MiB / 1152 | 684 MiB / 1440 |
| one token's expert footprint (decode) | 384 experts ≈ 1.02 GiB | 480 experts ≈ 0.67 GiB |

## 5. Routing distributions

**30B — [DERIVED, pilot]** legacy `route.log` (2026-09-27, mmap era, default benchmark prompt, greedy), request 1: 77 prompt tokens, 127 generated tokens. Files: [`data/processed/pilot-legacy-30b/`](data/processed/pilot-legacy-30b/REPORT.md).

[MEASURED, pilot] Two benchmark repetitions in the log are bit-identical (greedy decoding is deterministic): routing replays exactly for an identical request, so repeated identical requests are not independent samples.

[MEASURED, pilot] In prompt steps the last layer routes one token (llama.cpp computes only the output position there): a 77-token prompt needs the experts of 77 tokens in layers 0-46 and of one token in layer 47.

**80B — [OPEN] awaiting EXP-007.** No router trace of the 80B exists. [SUPERSEDED, 2026-10-11] The 80B router is now traced on 14 EXP-007 cells (global EWS90 fraction 0.225–0.472; see the EXP-007 README, claims C1–C4). The section is to be written in v1.

## 6. Per-layer entropy and concentration

[DERIVED, pilot, 30B, decode, 127 tokens × 8 selections per layer]

| layer | normalised entropy | effective experts (2^H) | Gini | top-1 | top-10 | distinct experts used |
|---|---|---|---|---|---|---|
| 0 | 0.773 | 42.5 | 0.722 | 12.5% | 53.9% | 95 |
| 23 | 0.809 | 50.7 | 0.694 | 8.4% | 41.1% | 79 |
| 47 | 0.582 | 16.8 | 0.901 | 12.2% | 80.7% | 41 |
| all 48 (range) | 0.58 – 0.81 (CV 0.07) | | 0.69 – 0.90 | | | |

[LIMITATION] 1016 selections over 128 experts per layer: plug-in entropy is biased low and EWS biased small. Differences between layers of this size need longer traces and confidence intervals (RQ-003).

## 7. Expert-frequency distributions

[DERIVED, pilot] Full ranked table: `data/processed/pilot-legacy-30b/expert_frequency.csv`. Co-activation (same token, same layer): most pairs near independence (median lift 1.06 among each layer's top-5 pairs), the strongest pair at lift 3.7 (`coactivation.csv`).

80B: **[OPEN] awaiting EXP-007.**

## 8. Reuse distances

[DERIVED, pilot] Pooled over layers, P(an expert is selected again within N tokens | it is selected again):

| N | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| P(reuse ≤ N) | 0.515 | 0.615 | 0.720 | 0.841 | 0.932 | 0.976 | 0.997 |

Per layer the median reuse distance goes from 1 token (layer 47; 68% reused by the next token) to 3 tokens (layer 0; 38%). Temporal locality exists at short distance and differs by layer; whether "conventional temporal-locality assumptions" hold is a question about the policies (Section 11), not only about this curve.

## 9. Effective working sets

[DERIVED, pilot]

| layer | EWS90 | EWS95 | EWS99 | ws_τ=1 | ws_τ=4 | ws_τ=16 | ws_τ=64 |
|---|---|---|---|---|---|---|---|
| 0 | 51 | 67 | 85 | 8 | 21.6 | 47.2 | 74.1 |
| 23 | 47 | 57 | 70 | 8 | 19.6 | 41.3 | 62.0 |
| 47 | 15 | 19 | 32 | 8 | 14.2 | 22.5 | 31.5 |

Global (all 6144 layer-expert keys): EWS90 = 1729 (28%), EWS95 = 2115 (34%), EWS99 = 2763 (45%) — in bytes 4.9 / 6.0 / 7.9 GB at the uniform derived size of 2.86 MB per expert. Against these, Argine's measured 30B budgets (4 GiB ≈ 1504 experts) sit below the pilot's EWS90.

[LIMITATION] Whole-trace EWS over 127 tokens mixes short- and long-term demand; the Denning windows (ws_τ) are the short-term measure. Expert sizes in the pilot are derived, not read from the GGUF.

## 10. I/O timeline

**Historical [MEASURED, Argine v1, proxy]**: wait-derived queue depth 2.09 → 2.49 (sync) and 2.58 → 3.43 (async) from 4 to 1.5 GiB on the 30B; reads in flight per wait 2.36 → 3.71. These are proxies, not Q(t).

**Argine2 [MEASURED, test model, EXP-019]**: exact Q(t) from 71 reads, 100% exact completions (Linux synchronous path: time-weighted mean depth when busy 1.04, max 2). Demonstrates the pipeline only.

**Real models: [OPEN] awaiting EXP-007** — `summary.json → io` of each run: Q(t) distribution, latency percentiles, hidden vs unhidden I/O, blocked time per thread.

## 11. Cache behaviour

[MEASURED, Argine v1] 80B at 4 GiB: runtime-style hit rate 91.2% (170,088 misses). [DERIVED, Argine2] Expert-level ≈ 74% (EXP-020: runtime miss rate ≈ expert miss rate / 3). *[Historical derivation, superseded as a number by the 2026-10-12 EXP-020 audit: EXP-007 measures the 80B gate-slice hit rate directly, 61.4–78.1% at 4 GiB across workloads, coding DEV ≈ 65.9%.]*

[DERIVED, 30B, coding dev, Windows, EXP-007/EXP-021] The runtime and the replay must be compared on the same metric. Workload only, expert-event hit (all 3 slices) over decode:

| budget | measured runtime (3 runs) | runtime reproduction `store-replay` | `sharded-lru-16-expert` (counterfactual) | global LRU | per-layer LRU uniform | layer-aware static (IN-SAMPLE) | Belady-style |
|---|---|---|---|---|---|---|---|
| 4 GiB | 73.7% | 73.8% | 75.4% | 75.7% | 75.7% | 77.4% | 89.9% |
| 2 GiB | 51.7% | 51.7% | 53.5% | 53.3% | 56.1% | 57.9% | 75.5% |
| 1.5 GiB | 43.1% | 43.1% | 45.1% | 44.1% | 49.1% | 51.3% | 68.8% |

[SUPPORTED; "Windows confirmation pending" is a historical label, resolved 2026-10-04 (EXP-021 §14) and confirmed again 2026-10-12 (EXP-019)] `store-replay` reproduces the runtime: 0 event disagreements over the measured workload on the tested v2 traces (per-thread pins, 2026-10-04), and within 0.1 pp on the nine 30B runs. The `store-replay` column uses the corrected per-thread pin model (analysis-v3); with the superseded single-pool model it read 73.9 / 51.9 / 43.2% ([write-up](docs/research/REPLAY-FIDELITY-VALIDATION.md)). Counterfactual and oracle rows are expert-unit and event-atomic. They are comparable with each other, not with the measured column. Source: `experiments/EXP-007-router-characterization/analysis-v3/run-00-4G/`.

[SUPERSEDED, 2026-10-03, EXP-021] The pilot table below used replay semantics v1-legacy. In that version prefill was not served to the cache and warm-up was counted. It also called `sharded-lru-16` the "runtime model", which it is not. Regenerated v2 values (`data/processed/pilot-legacy-30b-v2/`), same column order:

| budget | global LRU | sharded LRU-16, expert units | per-layer LRU uniform | layer-aware static* | LFU | static frequency* | Belady-style |
|---|---|---|---|---|---|---|---|
| 4 GiB | 82.4% | 81.9% | 81.9% | 83.6% | 81.6% | 85.7% | 93.5% |
| 3 GiB | 72.5% | 72.5% | 73.5% | 75.1% | 72.6% | 76.7% | 88.9% |
| 2 GiB | 60.4% | 59.9% | 62.3% | 64.1% | 60.3% | 63.8% | 80.7% |
| 1.5 GiB | 50.6% | 51.6% | 55.1% | 57.4% | 52.3% | 55.4% | 74.1% |
| 1 GiB | 0.0% | 20.1% | infeasible | infeasible | 42.5% | 45.0% | 63.9% |
| 768 MiB | 0.0% | 2.1% | infeasible | infeasible | 36.5% | 38.8% | 56.0% |

Original v1-legacy pilot table, kept as history:

| budget | global LRU | sharded LRU-16 (runtime model) | per-layer LRU uniform | layer-aware static* | LFU | static frequency* | Belady-style |
|---|---|---|---|---|---|---|---|
| 4 GiB | 81.7% | 81.1% | 81.1% | 82.7% | 80.9% | 85.7% | 90.4% |
| 3 GiB | 72.1% | 72.1% | 73.1% | 74.6% | 72.2% | 76.7% | 86.6% |
| 2 GiB | 60.2% | 59.7% | 62.1% | 63.9% | 60.1% | 63.8% | 79.2% |
| 1.5 GiB | 50.5% | 51.5% | 54.9% | 57.2% | 52.2% | 55.4% | 73.0% |
| 1 GiB | 0.0% | 20.0% | infeasible | infeasible | 42.0% | 45.0% | 63.2% |
| 768 MiB | 0.0% | 2.1% | infeasible | infeasible | 36.5% | 38.8% | 55.5% |

\* fitted in-sample: upper bounds, not results.

Readings that Phase 1 must confirm or falsify (status after EXP-007 run-00, v2): (a) the Belady gap is large (LRU ≈ 1.9× the oracle's misses at 4 GiB in the v1 pilot; 2.4× on the 30B coding trace) — [SUPPORTED, one trace]; (b) recency policies collapse below one token's footprint; (c) a uniform per-layer split beats a global LRU at 1.5-3 GiB in this trace — [SUPPORTED, one trace] +2.8 pp at 2 GiB and +5.0 pp at 1.5 GiB, tie at 4 GiB.

## 12. Memory elasticity

[DERIVED, pilot, in-sample] Exact per-layer LRU curves (`lru_curves.csv`), converted with the cost model at 3.96 misses per node (`elasticity.csv`). Misses per token and marginal value dE/dm (ms of stall per token per additional MiB) at selected slot counts:

| slots (bytes) | layer 0 | layer 23 | layer 47 |
|---|---|---|---|
| 8 (22.9 MB) | 5.28 / 0.028 | 4.63 / 0.069 | 2.78 / 0.061 |
| 16 (45.7 MB) | 4.37 / 0.047 | 3.39 / 0.047 | 1.06 / 0.036 |
| 31 (88.6 MB) ← uniform share at 4 GiB | 2.72 / 0.028 | 2.02 / 0.022 | 0.35 / **0.000** |
| 64 (183 MB) | 1.15 / 0.008 | 0.76 / 0.011 | 0.32 / 0.000 |

[DERIVED, pilot] At the uniform share, an additional MiB is worth nothing to layer 47 (only 41 distinct experts were used there) and still ~0.03 ms/token to layers 0 and 23: layer-specific elasticity exists in this trace. [SUPERSEDED, EXP-021: grid-rounded allocator, v1-legacy curves] The in-sample optimal static split uses 17-45 slots per layer at 4 GiB instead of 31 everywhere, and saves 9.0% of misses vs uniform (6.1% at 3 GiB, 5.0% at 2 GiB, 5.3% at 1.5 GiB) — `heterogeneity.json`.

[DERIVED, pilot, v2, exact allocator] 17-46 slots per layer at 4 GiB. Uniform makes 10.1% more misses than the optimal split (6.4% at 3 GiB, 5.2% at 2 GiB, 5.4% at 1.5 GiB): `pilot-legacy-30b-v2/heterogeneity.json`. [DERIVED, 30B coding dev, 389 decode tokens, in-sample] 7.1 / 4.2 / 4.5% at 4 / 2 / 1.5 GiB, with 18-71 slots per layer at 4 GiB. The earlier report of an "optimal" split worse than uniform was an allocator bug (FH-026). The curves in the table above are v1-legacy. v2 curves differ by ≤ 2.4% for layers 0 and 23 up to 31 slots. They differ more for layer 47 (45 → 38 misses at 31 slots) and at 64 slots (up to 36%), because the prompt now warms the cache (`pilot-legacy-30b-v2/elasticity.csv`).

[LIMITATION] Small-sample curves are noisy (non-monotone dE/dm); in-sample; derived expert sizes.

## 13. 30B vs 80B

**Known [MEASURED / DERIVED from Argine v1]:** at ~12% residency the 80B is 1.58× faster; speed/residency exponent 0.384 vs 0.50-0.58; per-token expert bytes 0.65× and reads 1.25× of the 30B; slices 440-704 KiB vs ~930 KiB; I/O bucket on the 80B: read 44%, decommit 31%, lock 17%, commit 8% (30B: 48 / 6 / 3 / 1%); RAM peak − budget 2.110 vs 1.520 GiB.

**[OPEN] awaiting EXP-007/EXP-026:** routing concentration, EWS, reuse, expert-level hit rates and queue depth of the 80B. [Update 2026-10-11] EXP-007 now provides the 80B's concentration, EWS, reuse and expert-level hit rates (61.4–78.1% at 4 GiB); queue depth is in each run's `summary.json`. The causal comparison remains EXP-026. [Update 2026-10-16] EXP-026 closed INCONCLUSIVE: at matched budget/F the untraced slopes were not distinguishable, the traced agreement gate failed, and no candidate cause was evaluable; the causal comparison remains open.

## 14. Supported hypotheses

- [SUPPORTED] The expert budget is a hard ceiling (F-001; 24 runs, two architectures).
- [SUPPORTED] Asynchronous reads help in proportion to miss pressure (EXP-004).
- [SUPPORTED] The access path, not the device, limited mmap streaming (EXP-003).
- [SUPPORTED, test model] The research instrumentation is complete and observation-only (EXP-019).
- [SUPPORTED, test model] The runtime hit counter includes guaranteed re-pins (EXP-020).

## 15. Falsified hypotheses

- [FALSIFIED] Residency percentage predicts scaling (EXP-006).
- [FALSIFIED, proxy-scoped] A larger compressed model knows more (EXP-012). Read as: the larger compressed 30B showed no advantage on the matched-text perplexity proxy. Equal quality or knowledge is not established (historical audit 2026-10-11).
- [FALSIFIED] Runs of the same configuration agree within 5% (EXP-018).
- [FALSIFIED] The runtime hit rate is the expert-level hit rate (EXP-020).
- Full registry: [`FAILED_HYPOTHESES.md`](FAILED_HYPOTHESES.md).

## 16. Open questions

RQ-002, RQ-003, RQ-009, RQ-010 and RQ-011 are the questions this document must answer in v1; see [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).
