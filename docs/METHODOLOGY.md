# Methodology

## 1. Research method

`observe → measure → characterize → hypothesize → simulate → implement → falsify → document`

Every optimisation must start from a measured bottleneck or an observed phenomenon. The project is required to be able to prove its main hypothesis wrong; performance improvement is desirable, not required for scientific success.

## 2. Statement taxonomy

| tag | meaning |
|---|---|
| `[MEASURED]` | read from raw data (cite the record or file) |
| `[DERIVED]` | computed from measured data by a stated method |
| `[HYPOTHESIS]` | a claim to be tested |
| `[SUPPORTED]` | a hypothesis that survived a test designed to falsify it |
| `[FALSIFIED]` | a hypothesis contradicted by evidence |
| `[OPEN]` | not known yet |
| `[LIMITATION]` | what the data cannot support |

An inference is never presented as a measurement. "Pilot" results (too small, in-sample, or legacy data) are tagged `[DERIVED, pilot]` and are not evidence.

## 3. Experiments

- Permanent IDs `EXP-NNN-short-name`; never reused.
- Directory: `README.md` (question, hypothesis, setup, commands, raw data, result, interpretation, limitations, status history), `config.json`, `raw/`, `processed/`, `results.csv`, `figures/`.
- Lifecycle: PLANNED → RUNNING → SUPPORTED | FALSIFIED | INCONCLUSIVE; SUPERSEDED when a later experiment replaces the conclusion (with a link, without deleting).
- `tools/index_experiments.py --check` enforces structure and vocabulary.

## 4. Definitions

- **Step.** One forward pass (one llama.cpp graph = one ubatch). Decode steps have one token; prompt steps many, except in the last layer.
- **Event.** (step, layer, set of experts needed): the unit of residency.
- **Hit-rate metrics** (three, not interchangeable; EXP-020 / EXP-021, clarified 2026-10-12):
  - **Runtime-style slice hit rate** (`runtime_style_hit_rate`): hits over all slice accesses in the gate, up and down nodes of each layer. In the tested Argine2 store the up/down accesses are guaranteed re-pins, so this rate over-states first-request hits. It is reported, never compared with the other two.
  - **Runtime gate-slice hit rate** (stored as `expert_level_hit_rate`; field name kept): slice hits over the slice accesses of the gate (first) node of each layer, prefill + decode, workload only (warm-up and flush sentinel excluded). On the tested store, gate-slice miss rate = 3 × runtime-style miss rate, exactly 3.000 in all 252 qualified EXP-007 runs; implementation-specific.
  - **Expert-event hit rate** (`expert_event`; replay and EXP-021 analyses): the fraction of an event's experts already resident, with all three slices, when the event starts. It is stated per phase. It is separate from gate-slice: on 30B coding DEV at 4 GiB, decode gate-slice is 75.4% and decode expert-event is 73.7%.
  - Always state the metric and the phase. An unqualified "expert-level hit rate" taken from a run manifest is the gate-slice field; replay results name `expert_event` explicitly.
  - *(Earlier wording, kept as history: "**Expert-level hit rate.** Fraction of event members already resident when the event starts. The runtime's counter (slice requests in all three nodes of a layer) is reported as *runtime-style* and is not comparable.")*
- **EWS_q.** Minimum number of experts covering a fraction q of selections in a layer (or globally over all layer-expert keys).
- **Denning working set ws_τ.** Mean number of distinct experts selected in the last τ tokens.
- **Stack distance.** Event-atomic LRU stack distance (see `metrics.lru_stack_distances`); per-layer LRU with c slots hits iff distance < c.
- **Elasticity E_L(m).** Stall avoided by m bytes in layer L relative to its minimum feasible allotment; **SCB** = marginal stall avoided per byte per token.
- **Queue depth Q(t).** Number of reads outstanding at time t from per-request issue/complete timestamps. The wait-derived figure printed by Argine v1 is a *proxy* and is labelled so.
- **Hidden / unhidden I/O.** Part of a read's lifetime before / after its thread started waiting.

## 5. Replay

Three categories, never mixed in one table or conclusion (EXP-021, [`research/REPLAY-FIDELITY-VALIDATION.md`](research/REPLAY-FIDELITY-VALIDATION.md)):
- **runtime reproduction**: `store-replay`, a transcription of the store's state machine. It works on slices, per shard, with per-thread pins, and is non-atomic, and is validated event by event against the runtime;
- **counterfactual**: implementable policies;
- **oracle**: uses future knowledge, so it gives bounds.

In-sample fitted policies carry the label IN-SAMPLE / UPPER BOUND.

Counterfactual and oracle replay is event-atomic, deterministic (members in ascending id, fixed hash seeds) and at expert granularity. Semantics v2:
- every recorded step is served to the cache state;
- only steps of the analysed phase are counted (decode by default; `--phase all` counts prompt steps too);
- warm-up and flush-sentinel steps are never counted.

v1-legacy semantics (only the analysed phase served, warm-up counted) remain selectable solely to reproduce superseded numbers. Compare a replay with the measured runtime only through `store-replay` and the runtime's own metric definition (EXP-020). Exact: hits, misses, bytes, evictions, overflow, peak residency, per-layer misses. Modelled: stall (cost model with its parameters stored in `analysis_manifest.json`). The budget invariant is checked after every event.

Oracles: Belady-style farthest-next-use (optimal only for single-key requests with uniform sizes); in-sample optimal static allocation (DP over exact LRU curves). Both are upper references, not achievable results.

## 6. Development and evaluation separation

Workloads (`workloads/v1`) have disjoint `dev` and `eval` prompt sets (the loader rejects overlap). Controller constants and offline policies are fitted on `dev` traces; claims use `eval` traces. Online adaptation during an evaluation run is allowed. In-sample results are always labelled `train=in-sample`.

Contamination guard: identical prompts produce identical routing under greedy decoding (observed: the two benchmark repetitions in the legacy route.log are bit-identical), so repeated identical requests are not independent samples and are not pooled.

## 7. End-to-end measurements

- ≥3 runs, 5 preferred; report mean, median, SD, CV and 95% CI (Student t).
- Interleave configurations inside one session (`argine research schedule`). Legacy Argine v1 records matching incomplete historical configuration keys showed cross-run differences of approximately 15-16%, and one supplementary 80B comparison differed by approximately 19%. EXP-018 cannot separate hidden configuration from stochastic cross-time variation. Accordingly, Argine2 interleaves/rotates configurations and uses complete fingerprints; the historical ~15% figure is a conservative legacy guardrail, not a universal significance threshold.
- Cool-down between runs (default 20 s) and a warm-up request whose steps are labelled `warmup`.
- Primary metrics: generation tok/s, ms/token, TTFT, unhidden I/O stall per token, bytes read per token, useful overlap, RAM peak, resident expert RAM. Secondary: hit/miss rates (expert-level), evictions, entropy, reuse, EWS, queue depth, controller overhead, adaptation time. A higher hit rate with lower throughput is not an improvement.

## 8. Fingerprints

Every run's `manifest.json` records: Argine2 git commit (and dirty flag), binary SHA-256 and `BUILD-INFO.json` (llama.cpp commit, patches, compiler, flags), model and store checksums (full SHA-256 or the labelled fast method), workload SHA-256, runtime flags (budget, read mode, threads, context, llama-server arguments), Python and platform, hardware summary. `publication_quality` is false if anything is missing.

## 8b. Run provenance (added after the first Windows trace)

Every `argine research trace` run records, in `manifest.json`:

- **Expert Store:** requested value (`auto` or a path), the resolved absolute path, the path the runtime actually opened (trace `C store` record), and a fingerprint (path, size, checksum, method) of the opened file. Runs with a store cannot be publication-quality without it.
- **Read mode:** `requested` (the CLI's normalised value) and `effective` (trace `C read_mode` record, i.e. runtime confirmation); a disagreement is reported, and the effective mode is what was measured.
- **Backend (added 2026-10-12):** `runtime_flags.backend` records the requested value, Argine's resolved backend, the Vulkan devices the binary enumerated, and the effective backend (CPU when a resolved `vulkan` finds no device). Runs before 2026-10-12 record no backend field; their effective backend must be derived from the binary's BUILD-INFO (EXP-019 audit).
- **Machine:** CPU model name with its source (Windows registry `ProcessorNameString`, PowerShell `Win32_Processor`, `/proc/cpuinfo`, `sysctl`; never hard-coded, never Windows' family/model/stepping string), physical and logical cores, RAM, OS.
- **Machine state:** system-available RAM immediately before launch, minimum during the run, after shutdown. Below `--min-start-available` (default 8 GiB) the run gets a warning, never an invalidation: the aim is to detect different starting states (EXP-018), and historical runs are not re-judged on RAM levels.
- **Final Expert Store statistics:** derived from structured trace records (last `S` snapshot + totals of `A`/`I`/`E` records), never from the bounded stderr buffer. Cited hit rates are **workload-only**: the warm-up and the flush-sentinel request are excluded by their marker windows.
- **Trace completeness:** for each request, decode steps in the trace must equal llama.cpp's `predicted_n − 1` and every step must cover all MoE layers. A flush-sentinel request after the workload pushes every buffered workload record to the file before the process is terminated. The sentinel's own records can still be lost at termination (observed in EXP-019, 2026-10-12: 5 records, 1 node). `trace_complete` is therefore a workload-scope verdict (warm-up and sentinel are not checked); termination-tail completeness is reported separately by `tools/validate_replay_fidelity.py` (`termination_tail_completeness`).
- **Validation:** platform, instrumentation and store confirmation (trace and stderr sources), effective read mode, binary and BUILD-INFO hashes, trace completeness, and a validation run ID, also appended to `VALIDATIONS.jsonl` next to the binary. `BUILD-INFO.json` is historical build provenance and is never modified (a test pins its hash).
- **Publication check:** `publication_quality` is true only if the fingerprint is complete (Argine2 commit, binary, llama.cpp commit, model, store, workload, CPU) and the run-level checks pass (instrumentation confirmed, effective read mode, effective store fingerprinted, final statistics present, trace complete over the workload requests, zero budget violations). It is not a statement that the trace is complete through process termination. `missing` lists what failed; `warnings` lists machine-state notes.

Queue depth is reported with `status` = `exact` only when every completion timestamp is exact; otherwise `upper_bound`, with `exact_completion_share`, the time-weighted mean over the full span and while I/O is active.

## 9. Correctness

Residency policies must not change model semantics. With identical model, prompt, seed and greedy sampling, outputs must be identical across policies and with tracing on or off (verified on the test model in EXP-019). The store's byte-level verification against the GGUF (`argine store verify`) and the C++ tests (TSan/ASan) protect expert weights.

## 10. No result rewriting

Conclusions are never edited away. A changed interpretation is added with `SUPERSEDED by EXP-NNN`, and both stay visible (see EXP-004 and EXP-017).
