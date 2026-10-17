# EXP-027 — Generation thread count on the current research runtime

| field | value |
|---|---|
| ID | `EXP-027-current-thread-sweep` |
| Status | **FALSIFIED** |
| Period | 2026-10-11 |
| Origin | Argine2 |
| Related questions | RQ-014 |
| Supersedes / superseded | — (new evidence; does **not** rewrite EXP-001) |

## Research question

Which generation thread count gives the highest generation throughput on the **current** Argine2 research runtime (`llama-research-v2`, CPU backend), separately for Qwen3-30B-A3B Q4_K_M and Qwen3-Next-80B-A3B-Instruct Q3_K_M, with the Expert Store at a 4 GiB budget?

**Relation to EXP-001.** This is the modern follow-up to RQ-014.
- [EXP-001](../EXP-001-thread-sweep/README.md) remains historically **CLOSED as INCONCLUSIVE** ([historical audit](../../docs/research/HISTORICAL-AUDIT-EXP001-006.md)). Its model and build were confounded, and it never ran on this binary.
- EXP-027 is new evidence. Whatever it finds, EXP-001's record and status stay unchanged.

## Hypothesis

[HYPOTHESIS, written before any run]
- **P1.** On the 30B, 4 threads (one per P-core) is not slower than 6, 8 or 12 threads by more than the between-launch variation.
- **P2.** 12 threads (P-cores plus E-cores) gives the lowest generation tok/s on both models.

Rationale: ggml synchronises all threads at every graph node, so a slow E-core adds a participant to every barrier. That argument does not depend on OpenMP, but it was only ever measured on the OpenMP build (EXP-001).

**Falsification.**
- P1 is falsified if 6, 8 or 12 threads is CLEAR BEST on the 30B.
- P2 is falsified if 12 threads is not the lowest-mean configuration on a model where the verdict is CLEAR BEST or the configurations are separated.

For the 80B there is no prior measurement, so no directional prediction is made beyond P2.

## Setup

**Independent variable:** generation threads ∈ {4, 6, 8, 12}.

**Controlled variables**, fixed and enforced by the runner:

| variable | value |
|---|---|
| models | Qwen3-30B-A3B Q4_K_M; Qwen3-Next-80B-A3B-Instruct Q3_K_M (one at a time, no alternation) |
| runtime | `llama-research-v2/llama-server.exe` (llama.cpp `254b177`, patches 0001–0003, no OpenMP, **no Vulkan**); full SHA-256 recorded in `raw/session.json` |
| backend | `--backend cpu`, explicit (not `auto`) |
| expert store | on (`--expert-store`, the `.psxs` next to each model); budget `--expert-budget 4G` (4 GiB) |
| read mode | `--read-mode asincrono`, verified from the store's own report after every launch |
| batch (prompt) threads | 12 in every configuration. Not a CLI option: the planner sets max(threads, min(logical, physical)) = 12 on the i5-1340P. The runner verifies it in a dry-run preflight and after every launch, and refuses to continue if it differs. |
| workload | the unchanged `argine benchmark` prompt (`argine.bench.DEFAULT_PROMPT`, SHA-256 recorded), `--n-predict 128`, temperature 0, prompt cache off, default 8-token warm-up |
| repetitions per launch | `--repeat 1` |
| retries | `--retries 0`: a failed start is a failed launch, never a silently reduced configuration |
| store statistics | `--expert-stats` in every launch (logging only, identical across configurations) |
| research trace | off |
| cooldown | 60 s between launches |
| machine | reference machine (i5-1340P, 16 GB, Windows 11), mains power, same power plan, no other heavy processes |

**Dependent metrics** (from the benchmark record, per launch):
- primary: generation tok/s;
- also: prompt tok/s, TTFT, warm-prefix TTFT, RAM peak, minimum free RAM, CPU mean and peak, process and system disk read, page faults, store counters (when reported), plan, errors.

**Replication unit:** one complete benchmark launch, meaning a fresh `llama-server` process with an empty Expert Store and one measured request. Requests inside one process share the populated store and process state, so they are **not** independent replicates. Each model × thread configuration gets **3 independent launches**, giving 2 × 4 × 3 = **24 launches**.

**Interleaving schedule.** The whole 30B block runs first, then the whole 80B block, because switching models changes the OS file-cache state. Thread order is rotated within each model:

| model | round 1 | round 2 | round 3 |
|---|---|---|---|
| 30B (launches 1–12) | 4, 6, 12, 8 | 6, 8, 4, 12 | 8, 12, 6, 4 |
| 80B (launches 13–24) | 12, 4, 8, 6 | 4, 6, 12, 8 | 6, 8, 4, 12 |

**Conditional fourth round (not part of this run).** A fourth, balanced round per model is run later only if:
- the verdict for that model is PRACTICAL TIE or INCONCLUSIVE, meaning the best configurations are too close relative to between-launch variation; or
- an order effect remains plausible.

It would complete the Williams square: 30B 12, 4, 8, 6; 80B 8, 12, 6, 4.

## Commands

```bat
python tools\exp027_run.py plan        &REM print and check the 24-launch plan (no model)
python tools\exp027_run.py preflight   &REM dry-run of the 8 configurations (no model load, no inference)
python tools\exp027_run.py run         &REM the experiment: preflight, fingerprints, 24 launches
python tools\exp027_run.py run --resume &REM continue an interrupted session (same session id)
python tools\exp027_run.py finalize    &REM post-run fingerprint finalisation only (automatic at the end of run)
python tools\exp027_run.py analyze     &REM raw\records.jsonl -> results.csv, processed\summary.json
```

Each launch is exactly:

```bat
python -m argine benchmark MODEL.gguf --backend cpu --expert-store --expert-budget 4G --read-mode asincrono --threads T --repeat 1 --n-predict 128 --retries 0 --expert-stats
```

The runner runs it with `ARGINE_LLAMA_DIR` = `llama-research-v2`. It refuses to start if the variable already points elsewhere.

## Raw data

[MEASURED] Session `20261011T124307-1a3efd64`, 2026-10-11 10:43–11:20 UTC, on mains power with the Balanced power scheme. 24/24 launches valid at the first attempt, no resume. The session wrote:
- `raw/session.json`: session id; protocol; the 24-launch plan with every exact command line; preflight results; best-effort Windows power scheme and AC/battery state (optional fields). Fingerprints:
  - full SHA-256 of the research binary and its BUILD-INFO, computed before the run (small file);
  - Argine2 git state (the commit is recorded as **missing** if the tree is not a git checkout; never invented);
  - machine identity and benchmark workload identity;
  - **models and Expert Stores (~107 GB) are not read in full before or during the 24 launches.** A full read just before the measurements would perturb the Windows file cache, NVMe state, thermal state and first-round timing. Before the run the runner records, per file:
    - canonical path, size and mtime;
    - the Argine2 fast fingerprint (first and last 16 MiB; the method used in EXP-007 manifests);
    - a full SHA-256 only if one is already known without reading the file: from the Argine2 checksum cache, keyed on path, size and mtime, or from an earlier run manifest with method `sha256` for the same path and size, written after the file's mtime. (None of the EXP-007 manifests has one; they used the fast method.)
  - **After launch 24** the runner computes the missing full SHA-256 values (`finalize`, automatic at the end of `run`). It checks that size and mtime are unchanged against the pre-run snapshot, before and after hashing. If any file changed, the session is marked **invalid**. Re-finalising or resuming never recomputes an already finalised hash.
- `raw/records.jsonl`: one line per launch attempt. Each line is the unchanged benchmark record copied from the shared `runs.jsonl`, plus an `exp027` block:
  - experiment and session id, model, round, position, order index, attempt;
  - threads, batch threads, budget, backend, read mode;
  - launch time (UTC), free RAM immediately before launch, exit status;
  - validity verdict with reasons, exact command, launch log, source shared-log identity (path, byte offset, timestamp).
- `raw/logs/launch-NN-aK.log`: complete stdout/stderr of each launch attempt.

After the run, `analyze` writes `results.csv` (one row per model × thread count) and `processed/summary.json` (verdicts).

**Fingerprints of this session** (`raw/session.json`):

| item | full SHA-256 | how |
|---|---|---|
| `llama-research-v2/llama-server.exe` (llama.cpp `254b177`) | `a0063720ec4bb5d1ba3e4f7c0387dd1bc6a239b0ce038ba3d755351dab3f5cf3` | before the run |
| `Qwen_Qwen3-30B-A3B-Q4_K_M.gguf` (18,632,184,480 B) | `a015794bfb1d69cb03dbb86b185fb2b9b339f757df5f8f9dd9ebdab8f6ed5d32` | after launch 24 |
| `Qwen_Qwen3-30B-A3B-Q4_K_M.psxs` (17,553,387,520 B) | `fef943ae0bc017864a47d41c1fc90d422016045f7e8705b1d138b7defd8c5146` | after launch 24 |
| `Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf` (38,299,874,080 B) | `7705e294047a8c285b5f86556bef81e12bc9309988a9ad7e806e81cb730a86ee` | after launch 24 |
| `Qwen3-Next-80B-A3B-Instruct-Q3_K_M.psxs` (36,843,655,168 B) | `40ef8ccadf04dafabcd918033d3c3bd4bfb7e4d1aa333868cd43fd3cbf5de29e` | after launch 24 |

- All four large files had unchanged size and mtime between the pre-run snapshot and the post-run hash (`post_run_check.unchanged = true`).
- The Argine2 commit is recorded as missing: the tree is not a git checkout.
- Every record carries `threads_batch = 12`, backend cpu, a 4 GiB budget, store read mode `asincrono`, build `llama-research-v2@a0063720`, 128 generated tokens and no errors or retries.

## Result

[MEASURED] Generation tok/s, one value per independent launch, in round order 1, 2, 3 (`results.csv`; verdicts in `processed/summary.json`, from `python tools/exp027_run.py analyze`):

| model | threads | launches (r1, r2, r3) | mean | median | SD | CV | min–max | prompt tok/s (median) | TTFT s (median) | RAM peak (max) |
|---|---|---|---|---|---|---|---|---|---|---|
| 30B | 4 | 6.50, 6.48, 6.55 | 6.51 | 6.50 | 0.0361 | 0.55% | 6.48–6.55 | 23.04 | 3.35 | 5.52 GiB |
| 30B | 6 | 6.45, 6.43, 6.44 | 6.44 | 6.44 | 0.0100 | 0.16% | 6.43–6.45 | 23.14 | 3.36 | 5.52 GiB |
| 30B | 8 | 6.64, 6.64, 6.61 | 6.63 | 6.64 | 0.0173 | 0.26% | 6.61–6.64 | 23.09 | 3.34 | 5.52 GiB |
| 30B | **12** | 6.72, 6.75, 6.75 | **6.74** | 6.75 | 0.0173 | 0.26% | 6.72–6.75 | 23.11 | 3.33 | 5.52 GiB |
| 80B | **4** | 6.92, 6.91, 6.89 | **6.9067** | 6.91 | 0.0153 | 0.22% | 6.89–6.92 | 16.58 | 4.65 | 6.11 GiB |
| 80B | 6 | 6.64, 6.62, 6.60 | 6.62 | 6.62 | 0.0200 | 0.30% | 6.60–6.64 | 16.70 | 4.63 | 6.11 GiB |
| 80B | 8 | 6.46, 6.73, 6.72 | 6.6367 | 6.72 | 0.1531 | 2.31% | 6.46–6.73 | 16.66 | 4.63 | 6.11 GiB |
| 80B | 12 | 6.76, 6.77, 6.78 | 6.77 | 6.77 | 0.0100 | 0.15% | 6.76–6.78 | 16.68 | 4.62 | 6.11 GiB |

[DERIVED] Closure criterion applied as pre-registered:

| model | best | second | gap | pooled SD (top 2) | max CV | round effect | verdict |
|---|---|---|---|---|---|---|---|
| 30B | 12 threads | 8 threads | 0.110 tok/s (1.63%) | 0.0173 | 0.55% | 0.19% | **CLEAR BEST** |
| 80B | 4 threads | 12 threads | 0.137 tok/s (1.98%) | 0.0129 | 2.31% | 0.94% | **CLEAR BEST** |

For both models:
- the winner's lowest launch is above the second's highest;
- the gap is ≥ 2 pooled SD;
- the round (order/session) effect is smaller than the gap;
- every CV is below the 5% ambiguity threshold.

**The conditional fourth round is therefore not required**, and it was not run.

[DERIVED] Effect sizes (ratio of means):
- **30B:** 12 vs 8 threads +1.7%; 12 vs 4 +3.5%; 12 vs 6 +4.7%.
- **80B:** 4 vs 12 threads +2.0%; 4 vs 8 +4.1%; 4 vs 6 +4.3%.
- Neither curve is monotone in the thread count: 30B 12 > 8 > 4 > 6; 80B 4 > 12 > 8 ≈ 6.

[MEASURED] **The 80B 8-thread round-1 launch (order index 15, 6.46 tok/s) is visibly lower** than the other two 8-thread launches (6.73, 6.72). It is kept, not removed as an outlier.
- In that launch the store recorded about 16% more accesses than any other 80B launch: 656,145 hits and 66,459 misses, against 557–581k hits and 56–59k misses in the other eleven.
- Its RAM peak was about 79 MB lower (6,483,001,344 B vs about 6,561–6,563 MB in the others).
- The cause is not known. Its CV (2.31%) is below the pre-registered 5% threshold, and it does not change the winner: 4 threads also beats the 8-thread median (6.72).

Prompt tok/s, TTFT and RAM peak do not vary meaningfully with the generation thread count. That is expected: prompt processing uses 12 threads in every configuration.

## Interpretation

**Verdict on the pre-registered hypotheses.**
- **P1 [FALSIFIED].** "On the 30B, 4 threads is not slower than 6, 8 or 12 by more than the between-launch variation." On the 30B, 12 threads is CLEAR BEST, 3.5% above 4 threads, which is about 20× the 4-thread SD.
- **P2 [FALSIFIED].** "12 threads gives the lowest generation tok/s on both models." 12 threads is the *best* configuration on the 30B and the second best on the 80B.
- The experiment's registry status is therefore **FALSIFIED**, following the convention for experiments whose pre-registered prediction failed (cf. EXP-006). The measurements themselves are complete and valid.

**Finding.** [SUPPORTED] On the current CPU-only Argine2 research runtime at a 4 GiB expert-memory budget, the throughput-optimal generation thread count is **model-dependent**:
- Qwen3-30B-A3B Q4_K_M: **12 generation threads** were best in this experiment.
- Qwen3-Next-80B-A3B-Instruct Q3_K_M: **4 generation threads** were best in this experiment.

Neither value is claimed to be universally optimal. The scope is: the reference i5-1340P machine; the `llama-research-v2` binary (SHA-256 `a0063720…`); CPU-only backend; asynchronous Expert Store; 4 GiB expert budget; the unchanged deterministic `argine benchmark` workload (one 128-token request after a cold start); these two models. No conclusion is made about lower expert budgets.

**Relation to EXP-001.**
- EXP-001 remains historically CLOSED as INCONCLUSIVE; its measurements and status are unchanged.
- EXP-027 is the modern follow-up and shows why its rule ("one thread per P-core", +40–46% for 4 over 8 threads on the 30B, OpenMP build) should not have been generalised. On the same current runtime, the two models have different optima, and for the 30B the current build reverses the old ordering: 12 > 8 > 4.
- The barrier rationale behind P1 and P2 is not supported for the 30B on this build.

**Recommended settings for future performance-sensitive runs on this exact runtime** (machine, binary, CPU backend, 4 GiB):
- **30B:** `--threads 12`, with prompt threads 12 (the planner computes this on this CPU).
- **80B:** `--threads 4`, with prompt threads 12.
- Future runs must pass `--threads` explicitly: the planner's default (one thread per P-core = 4) is unchanged, because the runtime was deliberately not modified.
- Protocols of completed experiments are not altered: EXP-007's research traces used `--threads 4 --threads-batch 12` for both models and stay as recorded.
- A future experiment that must match an earlier protocol may keep its earlier thread setting, and should say so.

**Lower budgets** (2 GiB, 1.5 GiB) remain untested. They would be justified if thread behaviour is suspected to interact with miss pressure.

**Closure criterion (pre-registered, unchanged).** For each model separately, using valid launches only:
- **INCONCLUSIVE** if any of the following holds:
  - a configuration has fewer than 3 valid launches;
  - the between-launch CV of any configuration exceeds 5%;
  - the best and second configurations are separated, but the spread of round means (each launch normalised by its configuration mean) is at least as large as the relative best–second gap, so order or session drift could explain it.
- **CLEAR BEST** if the best configuration's lowest launch is above the second's highest, the mean gap is at least 2 × the pooled SD of the two, and the round effect is smaller than the gap.
- **PRACTICAL TIE** otherwise. The tied set is every configuration within 2 pooled SD of the best, or whose launch range overlaps the best's.

n = 3 per configuration does not allow a significance test, and none is claimed. PRACTICAL TIE or INCONCLUSIVE triggers the conditional fourth round instead of naming a winner. EXP-027 can be closed SUPPORTED or FALSIFIED (with respect to P1 and P2) only once each model has a CLEAR BEST or a stable PRACTICAL TIE. Otherwise it closes INCONCLUSIVE.

**Scope of any claim:** this machine, these two models, this binary, CPU backend, 4 GiB expert budget, this benchmark workload. No universal optimal thread count is claimed.

**Lower budgets** (2 GiB, 1.5 GiB) are not part of EXP-027. They are justified only if the 4 GiB result suggests that thread-count behaviour interacts materially with miss pressure, for example a different optimum between models that tracks their miss rates.

## Limitations

- [LIMITATION] n = 3 launches per configuration, so verdicts are descriptive; there is no significance test.
- [LIMITATION] Every launch starts with an empty Expert Store. The single measured request includes cache-fill misses after an 8-token warm-up. This is identical for every configuration, but it means the result describes a short request after a cold start, not sustained generation.
- [LIMITATION] Only generation threads vary. Prompt processing uses 12 threads throughout, so prompt tok/s and TTFT are not a thread effect here.
- [LIMITATION] Windows schedules threads across P-cores and E-cores; no affinity is set, as in every earlier measurement.
- [LIMITATION] Model blocks are not counterbalanced: the 30B always runs first. Cross-model comparison is not a goal.
- [LIMITATION] Within-session thermal and order drift are mitigated by rotation and cooldown, not eliminated; the round-effect check is the guard. Between-session variation is not measured.
- [LIMITATION] The repository was not a git checkout when the experiment ran, so the Argine2 commit is missing (same limitation as EXP-007).
- [LIMITATION] One session. The rotation guards against order effects within it (round effect 0.19% / 0.94%, smaller than the gaps). Between-session variation was not measured here; EXP-018 previously observed ≈15–19% cross-run differences among legacy Argine v1 records with incomplete runtime provenance (not a pure session noise floor; narrowed by the 2026-10-11 EXP-018 audit).
- [LIMITATION] Free RAM before launch rose steadily during the session, from 9.53 GB (launch 1) to 11.85 GB (launch 24). The rotation spreads this drift across thread counts, and the round effect stays below the gaps.
- [LIMITATION] The low 80B 8-thread launch (index 15) is unexplained. Its store counters differ from every other 80B launch (see Result).
- [LIMITATION] The full model and store hashes are taken after the measurements. Identity during the run rests on the unchanged size and mtime plus the pre-run fast fingerprint. A content change that preserves both size and mtime would not be detected; this is not expected for read-only model files.

## Status history

2026-10-11 PLANNED: protocol, runner (`tools/exp027_run.py`) and folder prepared; nothing run. 2026-10-11 (before execution): full model/store hashing moved after the 24 launches; pre-run snapshot of path, size, mtime and fast fingerprint. Design unchanged. 2026-10-11 run: session `20261011T124307-1a3efd64`, 24/24 valid launches, full post-run fingerprints, files unchanged. **2026-10-11 FALSIFIED** (pre-registered P1 and P2). CLEAR BEST for both models, so no fourth round. Finding [SUPPORTED]: the optimum is model-dependent (30B: 12 threads; 80B: 4 threads), scoped to this machine, binary, CPU backend, 4 GiB and workload.
