# Argine2 — expert memory as a resource-allocation problem

**An open, executable investigation of sparse Mixture-of-Experts (MoE) inference under a hard DRAM budget on one consumer PC: is expert residency better modelled as an online resource-allocation problem across layers than as a cache-replacement problem?**

Argine2 is a research fork of [Argine](docs/history/ARGINE-V1-README.it.md), a local runtime that runs MoE models larger than RAM on a 16 GB Windows laptop by streaming experts from NVMe through a hard-capped expert cache built into llama.cpp. Argine answered *"how big can a usable MoE be on this machine?"* (an 80-billion-parameter model at 6.87 tokens/s). Argine2 asks *why* it behaves the way it does, and whether a better way to spend a fixed expert-memory budget exists. The repository is the research artifact: runtime, instrumentation, experiment framework, committed evidence, cryptographic metadata for external raw traces, and the narrative, including what failed.

**Artifact map.**

| resource | purpose |
|---|---|
| [`paper/main.tex`](paper/main.tex) | paper source: *Argine: Expert Streaming Under Hard Memory Constraints* |
| [`REPRODUCING.md`](REPRODUCING.md) | public reproduction and experimental-replication guide |
| [`experiments/INDEX.md`](experiments/INDEX.md) / [`experiments/index.json`](experiments/index.json) | canonical experiment registry and machine-readable status |
| [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) | research-question ledger, including what remains open |
| [`data/raw/`](data/raw), [`data/external_raw_manifest.csv`](data/external_raw_manifest.csv), and [`experiments/`](experiments) | committed historical raw evidence, cryptographic inventory of external experiment raw data, processed outputs, protocols and experiment-local figures |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | experimental methodology and evidence rules |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | third-party license notices and text-fixture provenance |
| [`tools/paper_figures.py`](tools/paper_figures.py) | reproducible generator for the six main paper figures |

**Publication snapshot.** The planned experimental campaign is closed: **27 experiments**, with **13 SUPPORTED, 7 FALSIFIED, 1 SUPERSEDED, and 6 INCONCLUSIVE; 0 RUNNING and 0 PLANNED**. Open research questions remain explicitly open rather than being converted into post-hoc experiments.

Every statement below is tagged: **[MEASURED]** read from raw data, **[DERIVED]** computed from measured data by a stated method, **[HYPOTHESIS]**, **[SUPPORTED]**, **[FALSIFIED]**, **[OPEN]**, **[LIMITATION]**. Experiments have permanent identifiers (`EXP-NNN`), open questions too (`RQ-NNN`).

<!-- indice: generato da strumenti/controlla_readme.py, non modificare a mano -->

- [Research status](#research-status)
- [Research questions](#research-questions)
- [Why this problem matters](#why-this-problem-matters)
- [System architecture](#system-architecture)
- [Experimental platform](#experimental-platform)
- [Models](#models)
- [Methodology](#methodology)
- [Key findings](#key-findings)
  - [Finding F-001 — the expert budget is a hard ceiling](#finding-f-001--the-expert-budget-is-a-hard-ceiling)
  - [Finding F-002 — an 80B MoE is interactive on 16 GB without a GPU](#finding-f-002--an-80b-moe-is-interactive-on-16-gb-without-a-gpu)
  - [Finding F-003 — residency alone does not predict speed](#finding-f-003--residency-alone-does-not-predict-speed)
  - [Finding F-004 — speed falls sub-linearly with residency, and differently per model](#finding-f-004--speed-falls-sub-linearly-with-residency-and-differently-per-model)
  - [Finding F-005 — asynchronous reads help only under miss pressure](#finding-f-005--asynchronous-reads-help-only-under-miss-pressure)
  - [Finding F-006 — the access path, not the device, limited streaming](#finding-f-006--the-access-path-not-the-device-limited-streaming)
  - [Finding F-007 — the binary is an experimental variable](#finding-f-007--the-binary-is-an-experimental-variable)
  - [Finding F-008 — at similar footprint, the larger compressed MoE showed no perplexity advantage](#finding-f-008--at-similar-footprint-the-larger-compressed-moe-showed-no-perplexity-advantage)
  - [Finding F-009 — routing concentration is not a model-agnostic predictor of allocation opportunity](#finding-f-009--routing-concentration-is-not-a-model-agnostic-predictor-of-allocation-opportunity)
  - [Finding F-010 — adaptive layer allocation helps under pressure, but mostly by partitioning, and far from the oracle](#finding-f-010--adaptive-layer-allocation-helps-under-pressure-but-mostly-by-partitioning-and-far-from-the-oracle)
- [Unexpected findings](#unexpected-findings)
- [Falsified hypotheses](#falsified-hypotheses)
- [Current hypotheses](#current-hypotheses)
- [Open research questions](#open-research-questions)
- [Experiment index](#experiment-index)
- [Reproduction](#reproduction)
  - [Without a model: re-derive every table in this repository](#without-a-model-re-derive-every-table-in-this-repository)
  - [On the reference machine: instrumented runs](#on-the-reference-machine-instrumented-runs)
  - [Building the instrumented llama.cpp](#building-the-instrumented-llamacpp)
- [Raw data](#raw-data)
- [How to contribute](#how-to-contribute)
- [Limitations](#limitations)
- [Literature positioning](#literature-positioning)
- [Roadmap](#roadmap)
- [Citation](#citation)

<!-- fine indice -->

---

## Research status

| phase | state |
|---|---|
| Argine v1 — runtime, expert store, measurements | complete; 163 raw measurement records imported ([`data/raw/argine-v1`](data/raw/argine-v1)) and 16 historical experiments reconstructed from surviving Argine v1 evidence; most are raw-rederived where provenance permits, while EXP-015 is provenance-limited and depends on the historical README for PRETOUCH attribution and per-miss counters |
| **Phase 1 — observation** | **complete (2026-10-11).** Instrumentation built (patch [`0003`](patches/argine-research.h)) and validated ([EXP-019](experiments/EXP-019-instrumentation-validation/README.md): fixture/Linux 2026-10-03; fresh publication-quality Windows run 2026-10-12, workload-complete and replay event-exact, with record loss confined to the flush-sentinel tail and queue depth an upper bound; enabled-instrumentation overhead measured in [EXP-024](experiments/EXP-024-tracing-overhead/README.md): 4G 1.099% [0.605, 1.593], 1.5G 0.549% [−1.379, 2.478], overall INCONCLUSIVE). The [EXP-007](experiments/EXP-007-router-characterization/README.md) characterization campaign is closed: **28 / 28 cells** (2 models × 7 workload families × DEV/EVAL), 9 runs each, every cell-budget with 3 runs qualified by the [EXP-021](experiments/EXP-021-replay-fidelity/README.md) replay-fidelity method. Both models are characterized. |
| Phase 2 — counterfactual replay | event-atomic replay engine, 13 policies, Belady-style oracle, exact LRU elasticity curves, adaptive controller and ablations: run on all 28 EXP-007 cells (in-sample) and in a temporal analysis of the four transition traces |
| Phase 3 — adaptive residency | controller exists **in replay only** ([EXP-010](experiments/EXP-010-adaptive-allocation/README.md)); deliberately not in the runtime. [EXP-010](experiments/EXP-010-adaptive-allocation/README.md) is **CLOSED, FALSIFIED** (2026-10-11, frozen protocol EXP010-P1a; a DEV-selected, frozen, EVAL-tested retrospective holdout whose EVAL adaptive outcomes had already been seen in EXP-007): the DEV-selected adaptive controller did not outperform the DEV-fitted, workload-family-informed static allocation on EVAL for either model (G adaptive/static 1.0037 on the 30B, 1.0015 on the 80B; small effects). Layer-aware allocation still outperforms global LRU and per-layer uniform in aggregate; the additional online adaptation does not beat the stronger DEV-fitted static comparator. |
| Publication artifact | [`paper/main.tex`](paper/main.tex) is the canonical scientific synthesis; [`ARGINE2-MEMORY-CHARACTERIZATION.md`](ARGINE2-MEMORY-CHARACTERIZATION.md) is retained as an earlier research note. Reproduction instructions are in [`REPRODUCING.md`](REPRODUCING.md). |

[SUPPORTED, EXP-007, counterfactual replay, in-sample] The evidence on the central question is materially stronger, but it is not settled:
- Layer-aware adaptive allocation beats global LRU in most constrained regimes, and the gain grows with memory pressure. Most of it under pressure comes from per-layer partitioning itself.
- The gain is model- and workload-dependent: large on the 30B, small on the 80B.
- It is not universally best: 2Q is competitive at high memory on the 80B.
- It is far from the oracle.
- [OPEN] What is still missing: an untouched holdout (EXP-010 was a retrospective one) and a runtime implementation. The retrospective DEV-selected → frozen → EVAL-tested evaluation is done (EXP-010, FALSIFIED for adaptive over the DEV-fitted static split).

[FALSIFIED for the current controller] Its gain does not come from fast reaction to workload transitions (FH-029).

## Research questions

**Central — [RQ-001](OPEN_QUESTIONS.md#rq-001--is-expert-residency-better-modelled-as-online-resource-allocation-than-as-cache-replacement).** Can a sparse-LLM runtime dynamically allocate a fixed global expert-memory budget `M = M_1 + ... + M_L` across layers and experts according to their observed runtime value, outperforming static residency and conventional cache policies, without modifying or retraining the model?

**Secondary — [RQ-002](OPEN_QUESTIONS.md#rq-002--do-routing-concentration-and-effective-working-set-predict-behaviour-better-than-size-or-residency).** Do routing concentration and the effective expert working set predict memory-tiered inference behaviour more accurately than total parameter count or resident-expert percentage?

The working hypothesis behind both: *the marginal value of one additional byte of expert memory differs significantly between layers and changes over time*. Argine2 is built to be able to prove this wrong. The full list (18 questions) with known/unknown/next experiment is in [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).

## Why this problem matters

Sparse MoE models activate a few experts per token (8 of 128, 10 of 512), so the bytes a token needs are a small fraction of the model — but which fraction changes every token. On machines whose DRAM cannot hold the model, the runtime must decide which experts stay resident. Today this is mostly treated as cache replacement (LRU-like) inside one undifferentiated pool, or as a static split (llama.cpp's `--n-cpu-moe` / tensor overrides, per-layer slot counts in several research systems). If the value of memory differs strongly between layers and over time, both are leaving performance on the table; if it does not, simple policies are enough and the research community can stop looking there. Either answer is useful. The setting — a consumer laptop, NVMe, no GPU, Windows — is where memory-tiered inference is most needed and least studied.

## System architecture

```
Argine2
 ├─ Argine runtime (Python)          planner, hardware probe, llama-server manager, OpenAI/Anthropic API
 ├─ llama.cpp + Argine patches (C++)
 │   ├─ 0002 expert store            buffer type: experts read from a .psxs file into memory we commit;
 │   │                               sharded LRU with a HARD byte budget; pin/unpin; overlapped reads
 │   ├─ 0001 route log               (legacy) router decisions to a text file
 │   └─ 0003 research trace  [new]   per-token routing, every slice access (hit/miss/waited), every read
 │                                   with request id + issue/complete/wait timestamps, evictions with
 │                                   resident time, residency phase of every MoE node, cache snapshots
 └─ argine/research (Python) [new]   trace format · characterization · exact I/O timeline · policies ·
                                     event-atomic replay · Belady-style oracle · elasticity · allocators ·
                                     adaptive controller (replay) · performance model · fingerprints ·
                                     workloads · reports        → `argine research ...`
```

The expert store owns memory, residency, the hard budget, loading and pinning; any future controller sits above it and only decides budgets per layer, pinned sets and timing. The instrumentation observes and never decides. Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Experimental platform

The machine is a controlled constant. Generalisation is studied across models, workloads, budgets and policies, **not** across hardware: no claim in this repository is about hardware in general. [LIMITATION]

| | |
|---|---|
| CPU | Intel Core i5-1340P — 4 P-cores + 8 E-cores, 16 logical, AVX2 (no AVX-512, no AMX) |
| RAM | 16 GB (15.72 GiB usable) |
| Storage | NVMe; sequential 8 MiB reads 3305-3678 MB/s, 896 KiB direct reads 1254 MB/s at 1 reader → 3711 MB/s at 3 readers [MEASURED, [EXP-003](experiments/EXP-003-direct-io/README.md)] |
| GPU | Intel Iris Xe (integrated); no dedicated GPU; all results CPU backend |
| OS | Windows 11 (build 26200) |
| Runtime | llama.cpp patched with Argine's expert store; current research binary: llama.cpp `254b177` + patches 0001-0003, cross-compiled with MinGW-w64 13 (static, no OpenMP) — [`llama-research/BUILD-INFO.json`](llama-research/BUILD-INFO.json) (trace format v1, historical provenance of the EXP-007 runs); trace format v2 (shard id + per-shard sequence, needed for exact replay validation): [`llama-research-v2/BUILD-INFO.json`](llama-research-v2/BUILD-INFO.json), used on Windows since 2026-10-04 for every EXP-007 campaign except 30B coding DEV (format v1) |

## Models

The two models are treated as two experimental regimes, not as "small" and "large". [MEASURED from GGUF metadata as recorded by Argine]

| | Qwen3-30B-A3B | Qwen3-Next-80B-A3B-Instruct |
|---|---|---|
| parameters total / active | 30.5 B / 3.35 B | 79.7 B / 3.87 B |
| experts per layer / active per token | 128 / 8 | 512 / 10 (+1 shared) |
| MoE layers | 48 | 48 (36 with Gated DeltaNet attention) |
| quantisation used | Q4_K_M | Q3_K_M (down tensors Q4_K/Q5_K) |
| expert bytes | 16.35 GiB | 34.31 GiB |
| non-expert bytes (always resident) | ~1.0 GiB | 1.35 GiB |
| expert bytes read per token (all misses) | 1046 MiB in 1152 reads | 684 MiB in 1440 reads |
| bytes per read | ~930 KiB | 440 / 576 / 704 KiB |
| residency at a 4 GiB budget | 24.5% | 11.7% |

Additional models are added only when they run on this machine and answer a specific question.

## Methodology

`observe → measure → characterize → hypothesize → simulate → implement → falsify → document`. Summary (full text: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)):

- **Observation before optimisation.** No allocation algorithm is designed from intuition; the first deliverable is a characterization document.
- **Event-atomic replay.** One event = the set of experts one forward pass needs in one layer; all members must be resident together (the runtime pins them), so no policy may evict a member of the current event. Flattened sequential replay changes policy rankings ([arXiv 2608.07911](https://arxiv.org/abs/2608.07911)).
- **Exact vs modelled vs proxy.** Hits, misses, bytes and evictions from replay are exact; stall time from replay is modelled (published cost model); queue depth from the runtime is reconstructed from per-request timestamps, and any bound or proxy is labelled as such.
- **Hit-rate semantics.** Run manifests report two rates:
  - the runtime-style rate, over all slice accesses of the gate, up and down nodes;
  - the workload gate-slice rate stored as `expert_level_hit_rate`, over the slice accesses of the first (gate) node of each layer, prefill + decode.

  Replay analyses use a third metric, the expert-event hit rate (an expert hits only if all three of its slices are resident), which is not interchangeable with gate-slice ([EXP-020](experiments/EXP-020-hit-rate-semantics/README.md), [EXP-021](experiments/EXP-021-replay-fidelity/README.md); definitions in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §4).
  - *(Earlier wording, kept as history: "Hit rates in Argine2 count the first request of each expert per layer and step; the runtime's own counter does not.")*
- **Development vs evaluation.** Workloads have disjoint `dev` and `eval` prompt sets; constants are tuned on `dev`, claims are made on `eval`. Offline policies fitted on the same trace they are evaluated on are labelled *in-sample* everywhere.
- **Repetitions and interleaving.** ≥3 runs (5 preferred), mean/median/SD/CV/95% CI, configurations interleaved (`A B C A C B C A B`) inside one session, because legacy non-interleaved / incompletely fingerprinted comparisons showed differences large enough to confound small effects ([EXP-018](experiments/EXP-018-reproducibility-audit/README.md)).
- **Fingerprints.** Every measurement records Argine2 commit, llama.cpp commit and compile flags, binary SHA-256, model and store checksums, runtime flags, workload SHA-256. Incomplete fingerprint ⇒ not publication-quality.
- **Hard memory invariant.** `resident_expert_bytes ≤ budget` at all times; checked after every replay event and counted by the runtime (`violazioni`).
- **No result rewriting.** Old conclusions are marked SUPERSEDED with a link, never edited away.

## Key findings

### Finding F-001 — the expert budget is a hard ceiling

[MEASURED] RAM peak minus expert budget is **1.520 GiB (±0.001)** for Qwen3-30B-A3B in all 20 store runs of 2026-10-03, at budgets 4 / 3 / 2 / 1.5 GiB and three read modes, and **2.110 GiB** for Qwen3-Next-80B-A3B at all four budgets.

Evidence: [EXP-005](experiments/EXP-005-30b-budget-sweep/README.md), [EXP-006](experiments/EXP-006-80b-budget-sweep/README.md); 24 records in `data/raw/argine-v1/runs.jsonl`; binaries `llama-noomp@e007d93d`, `@3d91ef83`, `@3cac5531`.

[DERIVED] The constant is the non-expert footprint (shared tensors, KV cache, runtime). With memory-mapped experts the peak reached 105-111% of the declared budget ([EXP-003](experiments/EXP-003-direct-io/README.md)).

### Finding F-002 — an 80B MoE is interactive on 16 GB without a GPU

[MEASURED] Qwen3-Next-80B-A3B Q3_K_M: **6.96 / 6.87 / 6.81 tok/s** with a 4 GiB expert budget (11.7% of 34.31 GiB resident), RAM peak 6.11 GiB, TTFT 5.0 s. Evidence: [EXP-006](experiments/EXP-006-80b-budget-sweep/README.md), 3 repetitions, binary `llama-noomp@3d91ef83`, default benchmark prompt (77 tokens) + 128 generated.

[OPEN] A second 4 GiB benchmark on another binary eight minutes after a sweep gave 5.74 tok/s (see [Unexpected findings](#unexpected-findings)).

### Finding F-003 — residency alone does not predict speed

[MEASURED] At ~12% residency the 30B runs at 4.35 tok/s and the 80B at 6.87 — the model with 2.1× the experts is **1.58× faster**. The prediction written before the run, from residency and bytes per token, was 4.0-5.0 tok/s.

[FALSIFIED] *"Throughput degradation is primarily explained by resident expert percentage."* Evidence: [EXP-006](experiments/EXP-006-80b-budget-sweep/README.md).

[DERIVED] Argine attributed the difference to router concentration (a cache holding 11.7% of the experts served 91.2% of requests). Argine2 found that 91.2% counts guaranteed re-pins; the derived expert-level figure was ≈74% *[historical derivation from Argine v1's 91.2% on another binary and prompt, superseded as a number by the EXP-007 measurements below: 80B gate-slice 61.4–78.1% at 4 GiB, coding DEV ≈ 65.9%]* — still far above 11.7%, so the explanation survives with a smaller number ([EXP-020](experiments/EXP-020-hit-rate-semantics/README.md)).

[SUPERSEDED, 2026-10-11] "Concentration was inferred from a hit counter; no router trace of the 80B exists yet." [SUPPORTED, [EXP-007](experiments/EXP-007-router-characterization/README.md)] The 80B router is now traced directly on 14 cells.
- Its routing is generally more concentrated than the 30B's: global EWS90 fraction 0.225–0.472 vs 0.327–0.554.
- Measured expert-level hit rate at 4 GiB: 61.4–78.1%, with 11.7% of the experts resident.
- So concentration is measured, not inferred. Whether it *causes* the speed difference is still open ([RQ-009](OPEN_QUESTIONS.md#rq-009--why-does-the-80b-behave-differently-from-the-30b)), and it is not a model-agnostic predictor of allocation opportunity ([F-009](#finding-f-009--routing-concentration-is-not-a-model-agnostic-predictor-of-allocation-opportunity)).

### Finding F-004 — speed falls sub-linearly with residency, and differently per model

[DERIVED] Two-endpoint exponent d log(speed) / d log(residency) over a 2.67× residency span: **0.50-0.58** on the 30B (two sessions), **0.384** on the 80B (re-derived from raw medians; Argine reported 0.384 ± 0.043). Evidence: [EXP-005](experiments/EXP-005-30b-budget-sweep/README.md), [EXP-006](experiments/EXP-006-80b-budget-sweep/README.md).

[LIMITATION] Between-session spread at the same point (up to 15%) exceeds the quoted ± on the 30B exponent. The exponents are descriptive measurements of those sessions, not universal constants; the supported claim is that degradation was sub-linear over the measured range ([historical audit](docs/research/HISTORICAL-AUDIT-EXP001-006.md)).

### Finding F-005 — asynchronous reads help only under miss pressure

[MEASURED] One binary, 3 repetitions, modes in the fixed order sync → block-2 → async within each budget (not rotated; corrected 2026-10-11, previously described as "interleaved"): emit-and-continue reads vs synchronous **−0.6%** at 4 GiB, +4.2%, +4.9%, **+10.8%** at 1.5 GiB; fixed blocks of 2 are indistinguishable from synchronous. Evidence: [EXP-004](experiments/EXP-004-async-store/README.md).

[SUPERSEDED] Argine's earlier "+7-11% uniformly" compared three binaries.

[LIMITATION] Because the order was fixed, session and thermal order effects cannot be fully excluded.

### Finding F-006 — the access path, not the device, limited streaming

[MEASURED] One reader, 896 KiB slices, cold cache: memory-mapped page faults **310 MB/s**, explicit unbuffered reads **1254-1324 MB/s** (4.0-4.3×). [FALSIFIED] *"Disk bandwidth is the bottleneck."* Evidence: [EXP-003](experiments/EXP-003-direct-io/README.md). Scope: the reference machine, the tested Windows/runtime configuration and access pattern. A third session (2026-10-01) gave 524.5 vs 1665.8 MB/s (3.2×).

### Finding F-007 — the binary is an experimental variable

[MEASURED] Same model and flags, 4 threads: 7.45 tok/s (old build, OpenMP) vs 12.97 (new build, no OpenMP, static) vs 13.53 (official b11214): **1.74×** between two builds. Evidence: [EXP-011](experiments/EXP-011-binary-rebuild/README.md). Hence binary fingerprints in every measurement.

*Historical audit (2026-10-11):* 1.74× is the observed old-patched → new-patched contrast under the matched recorded model/runtime plan. The llama.cpp source/version, OpenMP and linkage/build configuration were not isolated, so the conclusion is about binary identity as an experimental variable, not about a specific compiler flag.

### Finding F-008 — at similar footprint, the larger compressed MoE showed no perplexity advantage

[MEASURED] Dense Qwen3-14B Q4_K_M and Qwen3-30B-A3B IQ2_S were compared within the same 16 GB deployment envelope using the same perplexity text/signature. Perplexity was 13.28 ± 0.32 for the dense 14B and 13.63 ± 0.33 for the 30B IQ2_S. The larger compressed model therefore showed no advantage on this proxy. Historical generation throughput was 3.83 versus 12.97 tok/s. Evidence: [EXP-012](experiments/EXP-012-dense-14b-control/README.md).

*Historical audit (2026-10-11):* the perplexity result falsifies only the proxy-scoped prediction that the larger compressed model would show a clear quality advantage. It does not establish equal assistant quality, equal knowledge or model equivalence. The approximately 3.39x throughput contrast is historical rather than a causal sparse-vs-dense speedup because the two measurements used different runtime binaries and different sessions.

### Finding F-009 — routing concentration is not a model-agnostic predictor of allocation opportunity

[SUPPORTED, [EXP-007](experiments/EXP-007-router-characterization/README.md), 28 cells, IN-SAMPLE upper bounds] In multi-turn the 80B has the more concentrated working set:
- global EWS90 fraction: 0.267 DEV / 0.226 EVAL on the 80B vs 0.369 / 0.327 on the 30B;
- larger per-layer routing spread.

Yet a uniform per-layer split wastes far less on the 80B. Excess misses over the optimal static split:

| | 4 GiB | 2 GiB | 1.5 GiB |
|---|---|---|---|
| 80B DEV | 2.05% | 0.49% | 1.49% |
| 80B EVAL | 3.85% | 0.96% | 1.49% |
| 30B DEV | 6.95% | 4.15% | 4.50% |
| 30B EVAL | 7.40% | 4.10% | 4.53% |

Across all workloads: 0.5–3.9% on the 80B vs 3.5–10.4% on the 30B.

[SUPPORTED] Routing concentration is not a model-agnostic predictor of exploitable inter-layer allocation opportunity.
- [DERIVED, exploratory, post hoc] **Within a model:** more concentrated workloads tend to have a larger in-sample uniform excess at 4 GiB (Spearman ρ = −0.767 on the 30B, −0.934 on the 80B; n = 14 cells per model, unequal trace lengths, no formal predictive test).
- **Across models:** the relationship does not transfer. Pooled, its sign reverses, and the more concentrated model has the smaller opportunity.
- The formal predictor comparison is EXP-008.

### Finding F-010 — adaptive layer allocation helps under pressure, but mostly by partitioning, and far from the oracle

[SUPPORTED, [EXP-007](experiments/EXP-007-router-characterization/README.md), counterfactual replay] adaptive-e64 vs global LRU:
- 30B: +0.8 … +1.5 pp at 4 GiB, +5.7 … +8.5 pp at 1.5 GiB;
- 80B: −0.3 … +0.9 pp at 4 GiB, +1.4 … +3.2 pp at 1.5 GiB.

What drives the gain, and what does not:
- [DERIVED] At 1.5 GiB a fixed uniform per-layer partition already gives most of it; the adaptive split adds +0.4 … +1.8 pp.
- 2Q matches or beats adaptive at six 80B points, mostly at 4 GiB.
- Epoch size (e32/e64/e128), hysteresis and cost weighting each change the hit rate by < 1 pp; hysteresis and epoch mainly change churn.
- Pinning helps consistently only in multi-turn (+0.7 … +1.7 pp, both models and splits). Elsewhere it is mostly neutral or harmful: on the 80B negative at every point, on the 30B small gains only in coding DEV.
- The future-informed Belady-style reference is still +8.5 … +20.8 pp above adaptive. How much of that headroom an online policy can realise is open.
- [FALSIFIED for the current controller] The transition gain is not fast reaction ([FH-029](FAILED_HYPOTHESES.md)).

## Unexpected findings

These came from re-reading Argine's raw data and code with Argine2's tools.

**U-001 — the runtime's hit rate is not the expert-level hit rate.** [MEASURED, test model] For every layer the gate, up and down nodes each request the slices of all three kinds; the gate node loads them and the other two re-pin them. On the end-to-end test, all 71 misses were in gate nodes: runtime-style hit rate 97.7%, gate-slice (`expert_level_hit_rate`) 93.1%. [MEASURED, first Windows smoke run, 30B at 4 GiB, not publication-quality] 90.06% runtime-style vs 70.17% gate-slice (`expert_level_hit_rate`): miss ratio 3.00. [DERIVED] runtime miss rate ≈ expert miss rate / 3. [EXP-020](experiments/EXP-020-hit-rate-semantics/README.md).
- [Update 2026-10-12] Confirmed beyond the smoke run:
  - EXP-021 attributed real 30B runs record for record, and all misses are in gate nodes;
  - EXP-007's 252 qualified 30B/80B runs give a gate-slice (`expert_level_hit_rate`) / runtime-style miss ratio of exactly 3.000 in every run.
- Scope: the tested Argine2 expert store and instrumented binaries, not MoE runtimes in general. The replay's expert-event hit rate is a separate metric.

**U-002 — legacy nominally matched runs can differ far more than 5%.**
- [MEASURED] Within one invocation, the repetition CV reached ≈6.43%.
- [MEASURED, LEGACY RECORDED-CONFIGURATION] 30B records matching the historical grouping key showed min-to-max spreads of ≈15.4% (4 GiB, same binary, five records over about ten hours: 5.34-6.16 tok/s) and ≈16.0% (6 GiB: 8.29-9.62).
- [MEASURED, SUPPLEMENTARY] A raw 80B 4 GiB pair on the same binary, eight minutes apart, measured 7.10 vs 5.74 tok/s. It is not a builder-matched group and its read-mode provenance is asymmetric.
- Hidden or unrecorded flags (`PRETOUCH` throughout, read mode before 2026-10-03) prevent interpreting these 15-19% contrasts as a pure session noise floor.
- The modern protocol uses complete fingerprints and interleaving. [EXP-018](experiments/EXP-018-reproducibility-audit/README.md).
- *[Historical heading, narrowed by the 2026-10-11 audit: "between-session dispersion is 15-19%, not 5%"]*

**U-003 — "one thread per P-core" was measured on the slow build.** [MEASURED] The +40-46% held for the MoE on the old OpenMP build, later measured 1.74× slower than a newer build (EXP-011; the cause, whether llama.cpp version, OpenMP or linkage, was not isolated); on the official build the dense 8B was faster with 8-12 threads than with 4. The rule is downgraded to INCONCLUSIVE. [EXP-001](experiments/EXP-001-thread-sweep/README.md). [Update 2026-10-11, [EXP-027](experiments/EXP-027-current-thread-sweep/README.md)] On the current CPU-only research binary at 4 GiB the optimum is model-dependent: the 30B is fastest with 12 threads (+3.5% over 4), the 80B with 4 threads (+2.0% over 12). That is why the old rule should not have been generalised. EXP-001's status is unchanged.

**U-004 — prompt steps touch one token in the last layer.** [MEASURED, legacy trace] llama.cpp computes only the output token in the last layer, so a 77-token prompt needs 77 tokens' experts in layers 0-46 and one token's in layer 47. Any prefill working-set model must account for it. [EXP-007](experiments/EXP-007-router-characterization/README.md).

**U-005 — pilot: LRU may collapse below one token's expert footprint.** [DERIVED, pilot, in-sample] On a 127-token legacy 30B trace, global LRU gets 0% expert hits at 1 GiB, 768 MiB and 512 MiB (one token sweeps 384 experts ≈ 1.02 GiB through the cache in layer order, the classic cyclic worst case), while LFU keeps 29-42% and Belady-style 43-63%. [HYPOTHESIS] The runtime's sharded LRU will collapse the same way: written down before measuring, as [EXP-022](experiments/EXP-022-extreme-pressure/README.md).

## Falsified hypotheses

Kept, not deleted; each with origin, reason, experiment, evidence and conclusion in [`FAILED_HYPOTHESES.md`](FAILED_HYPOTHESES.md). The most instructive:

| hypothesis | evidence | |
|---|---|---|
| Disk bandwidth is the primary bottleneck | page faults 310 MB/s vs direct reads 1254-1324 MB/s; in generation the store used ~half of the device curve | [FALSIFIED] EXP-003 |
| Larger reads beat more numerous reads | at fixed bytes per node, queue depth wins (3522 vs 2351 MB/s) | [FALSIFIED] EXP-003 |
| Asynchronous I/O helps every memory regime equally | −0.6% at 4 GiB, +10.8% at 1.5 GiB | [FALSIFIED] EXP-004 |
| Resident percentage predicts scaling | 80B 1.58× faster than 30B at equal residency | [FALSIFIED] EXP-006 |
| Larger compressed model shows a clear quality advantage | no PPL advantage on the matched-text proxy: 13.63 ± 0.33 vs 13.28 ± 0.32 | [FALSIFIED, proxy-scoped] EXP-012 |
| n-gram speculative decoding helps | no gain | [FALSIFIED] (Argine v1, no raw record) |
| The tested historical co-activation GGUF reorder improves throughput | 4.69 vs 5.17 tok/s in the historical attributed comparison (attribution from Argine's README) | [FALSIFIED, tested layout only] EXP-016 |
| Runs of the same configuration agree within 5% | within-invocation CV up to ≈6.43%; legacy nominally matched 30B groups ≈15-16% min-max spread (recorded-configuration scope) | [FALSIFIED] EXP-018 |
| One thread per P-core is not slower on the current 30B runtime; 12 threads is slowest | 30B: 12 threads CLEAR BEST (+3.5% over 4); 80B: 4 best, 12 second | [FALSIFIED] EXP-027 (FH-030) |
| The runtime hit rate is the expert-level hit rate | re-pins triple the denominator | [FALSIFIED] EXP-020 |
| The adaptive controller wins by reacting to workload transitions | a fixed split at its mean allocation gains 113–240% of its gain on all four transition traces | [FALSIFIED for the current controller] FH-029, EXP-007 |

## Current hypotheses

- **H-1** [HYPOTHESIS → partly SUPPORTED, model-dependent] Per-layer marginal value of memory is heterogeneous enough that a non-uniform split reduces misses materially.
  - *Earlier evidence: in-sample, exact allocator, uniform makes 7.1 / 4.2 / 4.5% more misses than the optimal static split on the 30B coding trace at 4 / 2 / 1.5 GiB (pilot: 5-10%). That is small next to the 2.4× gap between LRU and the oracle ([EXP-009](experiments/EXP-009-policy-replay/README.md)). The earlier pilot figures are [SUPERSEDED] by [EXP-021](experiments/EXP-021-replay-fidelity/README.md).*
  - *2026-10-11, [EXP-007](experiments/EXP-007-router-characterization/README.md), in-sample upper bounds: 3.5–10.4% on the 30B and 0.5–3.9% on the 80B, across 7 workload families × DEV/EVAL. "Material" holds for the 30B, hardly for the 80B.*
  - *2026-10-11, [EXP-010](experiments/EXP-010-adaptive-allocation/README.md): EXP-010 now provides a DEV-fitted → EVAL-tested retrospective-holdout result (leakage B: the EVAL traces had already been inspected in EXP-007). The workload-family-informed static split has about 4.1% (30B) and 1.0% (80B) fewer EVAL misses than uniform, as geometric means over 21 cells per model. An untouched holdout is still missing.*
- **H-2** [HYPOTHESIS] EWS / total experts predicts throughput better than resident / total ([EXP-008](experiments/EXP-008-working-set/README.md)).
  - *2026-10-11, [EXP-008](experiments/EXP-008-working-set/README.md): INCONCLUSIVE for throughput (80B favours the working-set predictor, 30B favours resident fraction). For expert-level hit rate the length-matched working set beats resident fraction in both models (supported within the traced EXP-007 dataset).*
  - *EXP-007 shows that EWS is not a model-agnostic predictor of allocation opportunity: it does not transfer across models, though there is an exploratory within-model association (F-009). That is a different question from throughput.*
- **H-3** [HYPOTHESIS] An online controller can track drifting marginal value and beat static splits on evaluation traces ([EXP-010](experiments/EXP-010-adaptive-allocation/README.md), [EXP-023](experiments/EXP-023-workload-drift/README.md)).
  - *Evidence so far: four transition traces (30B and 80B, DEV and EVAL). The current controller's gain is a better general split, not tracking.*
  - *It is slower than the segments, and its time variation costs misses against a fixed split ([analysis](docs/research/TRANSITION-ADAPTIVE-GAIN.md); FH-029).*
  - *2026-10-11, [EXP-010](experiments/EXP-010-adaptive-allocation/README.md) (retrospective holdout): the DEV-selected controller (e32) does not beat the DEV-fitted, workload-family-informed static split on EVAL in either model (FALSIFIED for the surviving sub-question; effects +0.37% / +0.15% misses).*
  - *2026-10-12, [EXP-023](experiments/EXP-023-workload-drift/README.md) (retrospective, existing EXP-007 traces): FALSIFIED for the current controller. Per-layer routing drift at workload boundaries is strong (descriptive JSD), and `adaptive-e64` has a lower signed `excess64` than its frozen split in 11 / 36 comparisons; the temporal-reaction falsification rests on FH-029. This does not falsify workload dependence.*
  - *H-3 stays open for a differently designed controller.*
- **H-4** [HYPOTHESIS] The 80B/30B difference has three candidate causes — concentration, smaller slices (more queue depth per byte), bookkeeping-bound I/O — to be discriminated ([EXP-026](experiments/EXP-026-30b-vs-80b-regimes/README.md)).
  - *Concentration is now measured directly (EXP-007: the 80B is more concentrated). Its causal share is not established.*
  - *2026-10-16, EXP-026 (prospective, frozen): INCONCLUSIVE.*
    - At matched budget/F and 4 generation threads, the untraced pressure-response slopes are NOT DISTINGUISHED (β30 = 0.476, β80 = 0.495; envelopes overlap). The historical flatter-80B slope did not reproduce in that region; this is consistent with a normalization/region difference and does not falsify EXP-006 or F-004.
    - The traced-slope agreement gate failed, so all candidates, including RQ-009's (d), are NOT EVALUABLE.
- **H-5** [INCONCLUSIVE] EXP-022 did not falsify the historical >30% condition at 30B / 768 MiB (mean workload gate-slice hit rate 0.2322%), but no positive support threshold was pre-specified; mechanism-side reuse nevertheless collapsed descriptively below the one-token footprint ([EXP-022](experiments/EXP-022-extreme-pressure/README.md)).

[SUPERSEDED, 2026-10-11] "The pilot already suggests a possible negative answer to the central question: what LRU loses is mostly *which* experts it keeps, not *how* memory is split across layers. That reading comes from 127 tokens and is not a result."

[SUPPORTED, [EXP-007](experiments/EXP-007-router-characterization/README.md), 28 cells, counterfactual] Within the tested replay abstraction, the remaining headroom is substantially larger for expert retention/selection than for layer-budget reallocation:
- the gap to the future-informed Belady-style reference (+8.5 … +20.8 pp over adaptive-e64) is several times larger than any layer re-split gain;
- whether an online prediction-aware policy can realise a material fraction of it is [OPEN];
- under pressure, most of the re-split gain comes from partitioning itself.

It answers the central question only in part. Layer-aware allocation does help in the constrained regimes. EXP-010 adds a retrospective DEV → EVAL test (leakage B), but no untouched holdout and no runtime controller test exists.

## Open research questions

Each question has a status, evidence, what is known and unknown, and a suggested next experiment in [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md). The ones a contributor can move today:

| ID | question | first useful contribution |
|---|---|---|
| RQ-001 | central question | partially answered: EXP-010 (retrospective holdout) found layer isolation plus a DEV-fitted static split at least as good as the online controller; open: an untouched holdout and a runtime test |
| RQ-002 | does EWS predict behaviour better than residency? | partially answered; EXP-008 completed the traced-data predictor comparison (INCONCLUSIVE for throughput, SUPPORTED for hit rate); tracing overhead is now measured (EXP-024: 95% CI entirely below 2% at 4G, inconclusive at 1.5G); a clean throughput follow-up still needs matched, untraced, interleaved measurements |
| RQ-005 | how much does uniform allocation lose? | partially answered in-sample (30B 3.5–10.4%, 80B 0.5–3.9%); EXP-010 (retrospective holdout): the DEV-fitted, workload-family-informed static split has 4.1% (30B) and 1.0% (80B) fewer misses than uniform on EVAL (geometric means) |
| RQ-009 | why is the 80B different? | partially answered: the 80B router is traced (EXP-007); EXP-026 (INCONCLUSIVE) found the historical slope difference not reproduced at matched budget/F, with all candidates not evaluable |
| RQ-010 | true expert-level hit rate on real models | answered for both models and seven workload families (EXP-007): 61–80% at 4 GiB; the miss ratio to runtime-style is exactly 3 |
| RQ-011 | does replay reproduce the runtime? | answered: exact over the workload on v2 traces (Linux; 30B and 80B on Windows) with per-thread pins — [write-up](docs/research/REPLAY-FIDELITY-VALIDATION.md) |
| RQ-012 | what does the instrumentation cost? | answered for the tested configuration ([EXP-024](experiments/EXP-024-tracing-overhead/README.md)): 4G 1.099% [0.605, 1.593], 1.5G 0.549% [−1.379, 2.478]. The preregistered < 2% hypothesis is INCONCLUSIVE overall (supported at 4G only) |
| RQ-014 | thread count on the current build | answered for the tested configuration ([EXP-027](experiments/EXP-027-current-thread-sweep/README.md)): model-dependent, 30B 12 threads, 80B 4 threads (CPU, 4 GiB, current binary) |
| RQ-017 | what exactly is Least-Stale? | read SpecMD and fix the replay variant |

## Experiment index

Generated, machine-readable: [`experiments/INDEX.md`](experiments/INDEX.md), [`experiments/index.json`](experiments/index.json). Status counts: SUPPORTED 13, FALSIFIED 7, SUPERSEDED 1, INCONCLUSIVE 6, RUNNING 0, PLANNED 0.

| ID | short title | status |
|---|---|---|
| [EXP-001](experiments/EXP-001-thread-sweep/README.md) | thread sweep | INCONCLUSIVE |
| [EXP-002](experiments/EXP-002-repack/README.md) | repack buffer | SUPPORTED |
| [EXP-003](experiments/EXP-003-direct-io/README.md) | direct I/O vs mmap | SUPPORTED |
| [EXP-004](experiments/EXP-004-async-store/README.md) | three read layers, one binary | SUPPORTED |
| [EXP-005](experiments/EXP-005-30b-budget-sweep/README.md) | 30B budget sweep | SUPPORTED |
| [EXP-006](experiments/EXP-006-80b-budget-sweep/README.md) | 80B prediction and sweep | FALSIFIED |
| [EXP-007](experiments/EXP-007-router-characterization/README.md) | router characterization | SUPPORTED (collection and analysis closed) |
| [EXP-008](experiments/EXP-008-working-set/README.md) | EWS as predictor | INCONCLUSIVE (throughput: models disagree; hit rate: supported within traced data) |
| [EXP-009](experiments/EXP-009-policy-replay/README.md) | counterfactual policy replay | INCONCLUSIVE (closed: "significant" had no pre-specified criterion) |
| [EXP-010](experiments/EXP-010-adaptive-allocation/README.md) | adaptive allocation (replay) | FALSIFIED (retrospective holdout: DEV-selected adaptive did not beat the DEV-fitted static split on EVAL) |
| [EXP-011](experiments/EXP-011-binary-rebuild/README.md) | binary identity (1.74× observed) | SUPPORTED |
| [EXP-012](experiments/EXP-012-dense-14b-control/README.md) | dense 14B control | FALSIFIED |
| [EXP-013](experiments/EXP-013-quality-per-bit/README.md) | quality per bit | SUPPORTED |
| [EXP-014](experiments/EXP-014-conversation-prefix-reuse/README.md) | conversation TTFT | SUPPORTED |
| [EXP-015](experiments/EXP-015-pretouch/README.md) | PRETOUCH | SUPPORTED |
| [EXP-016](experiments/EXP-016-coactivation-reorder/README.md) | co-activation reorder | FALSIFIED |
| [EXP-017](experiments/EXP-017-calibration-text/README.md) | calibration text | SUPERSEDED |
| [EXP-018](experiments/EXP-018-reproducibility-audit/README.md) | reproducibility audit | FALSIFIED |
| [EXP-019](experiments/EXP-019-instrumentation-validation/README.md) | instrumentation validation | SUPPORTED |
| [EXP-020](experiments/EXP-020-hit-rate-semantics/README.md) | hit-rate semantics | SUPPORTED |
| [EXP-021](experiments/EXP-021-replay-fidelity/README.md) | replay fidelity | SUPPORTED |
| [EXP-022](experiments/EXP-022-extreme-pressure/README.md) | extreme memory pressure | INCONCLUSIVE |
| [EXP-023](experiments/EXP-023-workload-drift/README.md) | workload drift | FALSIFIED (current controller) |
| [EXP-024](experiments/EXP-024-tracing-overhead/README.md) | tracing overhead | INCONCLUSIVE |
| [EXP-025](experiments/EXP-025-performance-model/README.md) | performance model | SUPPORTED |
| [EXP-026](experiments/EXP-026-30b-vs-80b-regimes/README.md) | 30B vs 80B regimes | INCONCLUSIVE (explanandum not distinguished at matched budget/F; traced gate failed; candidates not evaluable) |
| [EXP-027](experiments/EXP-027-current-thread-sweep/README.md) | current-runtime thread sweep (RQ-014) | FALSIFIED (pre-registered predictions; finding: model-dependent optimum) |

## Reproduction

The public reproduction entry point is [`REPRODUCING.md`](REPRODUCING.md). The commands below are retained as the detailed reference-machine workflow.

### Without a model: re-derive every table in this repository

```bash
# from the Argine2 folder; no install needed (python -m argine runs this folder's code)
python tools/build_historical_data.py          # EXP-001..006, 011..018 from data/raw/argine-v1
python -m argine research analyze data/processed/legacy-30b-route-request1.jsonl.gz \
       --budgets 4G,3G,2G,1.5G,1G,768M,512M --out /tmp/pilot   # EXP-007/009/010 pilot
python tools/index_experiments.py --check       # registry consistency
python -m unittest discover -s tests            # full test suite; no network, no model
```

Outputs are deterministic (sorted keys, fixed float precision); replay timings go to separate `overhead.csv` files.

### On the reference machine: instrumented runs

From the Argine2 folder, in an Anaconda Prompt. `argine.bat` runs this folder's code without installing it, so an existing `pip install -e` of the original Argine keeps working untouched (do not run `install.bat` here unless you want Argine2 to replace it; an installed Argine2 provides the command `argine2`).

```bat
set ARGINE_LLAMA_DIR=%CD%\llama-research
argine.bat doctor
argine.bat store build C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf
argine.bat research trace C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf --expert-store ^
   --budgets 4G,2G,1.5G --repeat 3 --workload coding --split dev ^
   --out experiments\EXP-007-router-characterization\raw\30b-coding-dev
argine.bat research analyze experiments\EXP-007-router-characterization\raw\30b-coding-dev\run-00-4G\trace.jsonl.gz ^
   --budgets 4G,3G,2G,1.5G,1G --out experiments\EXP-007-router-characterization\processed\30b-coding-dev
```

The `.psxs` store already built by Argine for the same GGUF is valid (same format); `--expert-store` finds it next to the model and the manifest records the path the runtime actually opened, with its checksum.

Publication-quality runs need a committed repository (the manifest records the Argine2 commit and whether the tree was dirty). Smoke tests may go in a non-`EXP-` folder such as `experiments\SMOKE-30B` (not indexed, no claims). `argine.bat research reprocess RUN_DIR` re-derives a run's tables and `summary.json` from its raw log.

`research trace` refuses to run if the binary does not confirm the instrumentation, runs budgets in interleaved order, and writes per run: `raw.tsv` (runtime log), `trace.jsonl.gz`, `io.csv`, `nodes.csv`, `evictions.csv`, `graph_timing.csv`, `summary.json` (queue depth from per-request timestamps with its status — `exact` only when every completion is exact, otherwise `upper_bound` — both hit rates, budget check) and `manifest.json` (fingerprint).

### Building the instrumented llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 254b177307b14d055bbe4c050c9c441a1e06d8a4     # last commit before the buffer-type API change
git apply ../patches/0002-argine-expert-store.patch
cp ../patches/argine-{store,experts,research}.{h,cpp} ggml/src/ggml-cpu/
git apply ../patches/0001-argine-route-log.patch
git apply ../patches/0003-argine-research-trace.patch
bash ../tools/build_llama_research.sh          # MinGW cross-build, static, no OpenMP
```

The C++ side is tested without llama.cpp: `patches/argine-research-test.cpp` (TSan / ASan clean) and the existing store tests.

## Raw data

```
data/raw/argine-v1/      runs.jsonl (163 records, 2026-09-27 → 10-03), perplexity.jsonl (8) — Argine's original logs,
                         sanitised only for host and user name (IMPORT.json holds both hashes)
data/raw/legacy/         route.log — the only router trace of the Argine v1 era (30B, 2026-09-27)
data/processed/          argine-v1-runs.csv (353 speed measurements), the legacy trace in Argine2 format,
                         pilot-legacy-30b-v2/ (pilot analysis, current replay semantics),
                         pilot-legacy-30b/ (original v1-legacy pilot analysis, superseded, kept as history)
experiments/EXP-*/raw/   external raw records used by experiments; intentionally excluded from Git and inventoried by data/external_raw_manifest.csv

The publication-candidate external raw manifest enumerates **3,280 files** (117,746,227,384 bytes; **109.660 GiB**) and has SHA-256 `da1d920dea3065bdde7800e644550647804fa64817c79e03229ec1d48a0b9b57`. The manifest records repository-relative paths, file sizes, and per-file SHA-256 digests; it contains no local absolute paths. The raw corpus itself is not stored in Git. Its archival download location will be added to the release metadata when the external archive is published.
experiments/EXP-*/       results.csv, processed/, figures/
```

Processed data can be re-derived from raw data with the commands above when the corresponding raw input is available. Repository-only checks and paper-figure regeneration use committed evidence; full reprocessing of experiments whose run artifacts live under `experiments/EXP-*/raw/` requires the external raw corpus matching [`data/external_raw_manifest.csv`](data/external_raw_manifest.csv). Figures always sit next to their CSV source.

## How to contribute

Work against a research question, not a feature. Good contributions: reproducing an experiment, adding a workload, testing an open hypothesis, implementing a baseline policy, improving instrumentation, analysing raw traces, reproducing a result on other hardware (labelled as such), or disproving an interpretation. Cite `RQ-` and `EXP-` IDs; negative results are welcome. Template and rules: [`CONTRIBUTING.md`](CONTRIBUTING.md); ready-to-file research issues: [`docs/ISSUES.md`](docs/ISSUES.md).

## Limitations

- [LIMITATION] One machine, one OS, one NVMe, CPU backend only. No hardware generalisation is claimed.
- [SUPERSEDED, 2026-10-11] ~~The only router trace analysed so far is a 127-token legacy trace of one prompt; every Argine2 pilot number is method validation, not evidence.~~ EXP-007 now provides instrumented traces of both models on 28 cells. Pilot numbers remain method validation only.
- [LIMITATION] Every EXP-007 policy analysis is **in-sample** (fitted and evaluated on the same trace). The static-frequency, static-hot-per-layer, layer-aware-static and optimal-static results are upper bounds, not deployable competitors. Single-turn cells have only 320–389 decode tokens.
- [LIMITATION] No EXP-007 run is publication-quality by the repository's own fingerprint rule: the Argine2 tree was not a git checkout, so `argine2 commit` is missing in every manifest. Validity rests on the EXP-021 qualification of each run.
- [LIMITATION] Historical results come from Argine's logs, which did not record cache counters, `PRETOUCH`, or (before 2026-10-03) the read mode; some numbers exist only in Argine's README and are labelled so.
- [LIMITATION] Counterfactual replay models residency per expert (gate+up+down together, event-atomic). The runtime caches slices independently, in address-hashed shards. Only `store-replay` reproduces the runtime: exactly over the measured workload on the tested v2 traces (with per-thread pins, since 2026-10-04); in the tested campaign, strict full-trace differences occurred only in the flush-sentinel tail. On the v1 30B runs it is statistical and within 0.1 pp. The counterfactual rows are comparable to each other, never to the measured runtime number ([EXP-021](experiments/EXP-021-replay-fidelity/README.md)).
- [LIMITATION] Replay stall times are modelled from a measured device curve; they are secondary to exact miss and byte counts.
- [LIMITATION] On Windows only ~67% of read completions are timestamped exactly (first smoke run); queue depth is then reported as an upper bound, never as exact.
- [LIMITATION] The Belady-style oracle is optimal for single-key requests and uniform sizes; for multi-key events and mixed sizes it is a reference, not a proven bound.
- [LIMITATION] Perplexity is the only quality measure; it is a weak proxy across different models.

## Literature positioning

Argine2 does not claim novelty for generic MoE caching, expert offload, prefetching, or cache-allocation mechanisms. The canonical literature review and comparison to prior systems are in Section 11 of the paper ([`paper/sections/11_related_work.tex`](paper/sections/11_related_work.tex)) and [`paper/references.bib`](paper/references.bib). Extended research notes are retained in [`docs/LITERATURE.md`](docs/LITERATURE.md). The contribution evaluated here is the empirical decomposition of expert residency under a hard global DRAM budget on the reference CPU+NVMe system, including validated replay, negative tests, and explicitly scoped limitations.

## Roadmap

The publication experiment campaign is complete: the registry contains no RUNNING or PLANNED experiments. Remaining work is deliberately separated from the evidence already reported.

Future directions include an untouched prospective holdout, runtime implementation and end-to-end validation of a controller above the expert store, replication on other hardware, and a broader failure-envelope map. These are future research directions, not missing prerequisites for reproducing the current artifact. Public release metadata, repository URL, release tag, and archival/DOI information are handled separately from the scientific experiment ledger.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a DOI exists, cite the repository and the commit hash of the results you use, and the `EXP-` identifiers.
