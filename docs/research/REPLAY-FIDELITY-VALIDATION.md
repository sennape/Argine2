# Replay fidelity validation (EXP-021)

**Modern audit (2026-10-12).** EXP-021 is CLOSED / SUPPORTED within the tested-path / format-v2 workload-fidelity scope ([EXP-021 README](../../experiments/EXP-021-replay-fidelity/README.md)). The research narrative below is kept as written.
- **Historical "pending" statements.** The top status, §10, §12 and §13.8 say a fresh Windows run is pending; they are historical. The fresh 30B confirmation and the 80B campaign (§14) resolved them on 2026-10-04.
- **Later confirmation.** A publication-quality 30B Windows run ([EXP-019](../../experiments/EXP-019-instrumentation-validation/README.md), 2026-10-12, not part of EXP-021) agrees independently: `exp021_closure` PASS, 0 tail disagreements.
- **Scope reminders.** The nine format-v1 30B traces remain statistical. On the 80B: 9 / 9 workload-exact, 7 / 9 full-trace exact.

**Status (2026-10-04, final):** EXP-021 SUPPORTED. The per-thread pin model passed a fresh 30B Windows run and the 80B coding-dev campaign, exact over the workload (§14). Earlier status text is kept below.

**Status (2026-10-04):** Question A is answered and Question B is answered. The first Windows format-v2 validation failed the closure criterion (5.29% event disagreement, 3 shards with sequence gaps). Both findings are explained in §13:
- **The divergence** came from a wrong pin model in the replay. Pins are per thread, and on Windows prompt and decode graphs run on different thread pools.
- **The gaps** are 5 records lost when the process was terminated, all inside the flush sentinel, so replay-irrelevant.

With the corrected model the same failed trace replays with **0 disagreements** in 1,474,909 decisions. One fresh Windows run is pending before EXP-021 closes. *[Historical (2026-10-04), SUPERSEDED: the fresh run passed (§14.1); see the 2026-10-12 audit note above.]* History of this section: the 2026-10-03 text below is kept; where §13 corrects it, the text says so.

- **Question A, why replay predicted higher hit rates than the runtime.** [SUPPORTED] The runtime-reproduction replay (`store-replay`) was shown equivalent to the real Expert Store event by event in three settings:
  - the C++ store driven deterministically, 0 disagreements;
  - a native Linux end-to-end run with 16 shards, 0 of 89,892 decisions different;
  - a high-pressure variant of that run, 0 of 89,928 different.

  On the nine Windows 30B runs it reproduces the runtime's hit rate within +0.10…+0.28 pp. The residual ~0.7% event disagreement has a single known cause: the shard mapping depends on memory addresses and is not recorded in format-v1 traces.
- **Question B, why "optimal static" lost to uniform.** [SUPPORTED] The allocator had a discretization bug and is now exact. An invariant check makes it fail loudly if the result is ever worse than uniform.

**Methodological limit.** Event-level equivalence on the *Windows* 30B runs needs one short trace recorded with the v2 instrumentation. The command is in §12. *[Historical (2026-10-03/04), SUPERSEDED for format v2: the v2 traces were recorded and replay exactly over the workload (§13, §14.1). The nine v1 traces remain statistical.]*

Taxonomy as in [`METHODOLOGY.md`](../METHODOLOGY.md): [MEASURED] [DERIVED] [HYPOTHESIS] [SUPPORTED] [FALSIFIED] [OPEN] [LIMITATION] [SUPERSEDED].

Code:
- [`argine/research/storereplay.py`](../../argine/research/storereplay.py), the runtime reproduction;
- [`argine/research/fidelity.py`](../../argine/research/fidelity.py), the event-level validator;
- [`tools/validate_replay_fidelity.py`](../../tools/validate_replay_fidelity.py), the 9-run table;
- [`argine/research/allocator.py`](../../argine/research/allocator.py), the exact allocator;
- [`patches/argine-store-fidelity-test.cpp`](../../patches/argine-store-fidelity-test.cpp), the deterministic driver of the real store;
- [`tests/test_fidelity.py`](../../tests/test_fidelity.py).

Data:
- [`experiments/EXP-021-replay-fidelity/`](../../experiments/EXP-021-replay-fidelity/README.md).

---

## 1. Problem

Every policy conclusion in Argine2 comes from offline replay of recorded router traces: LRU vs LFU vs 2Q, static vs adaptive, oracle gaps. Those conclusions mean something for Argine only if replay of the runtime's own policy reproduces the runtime. A replay that is "conceptually similar" can be off by more than the effects being measured. A 1–3 pp adaptive gain is meaningless next to a 4.7 pp fidelity error.

## 2. Observed discrepancy

[MEASURED] EXP-007, 30B Q4_K_M, coding dev, three repetitions per budget:

| budget | runtime "expert hit" (workload only) | replay `sharded-lru-16` (v1) | gap |
|---|---:|---:|---:|
| 4 GiB | 70.60% | 75.32% | +4.72 pp |
| 2 GiB | 49.75% | 53.30% | +3.55 pp |
| 1.5 GiB | 41.46% | 45.05% | +3.59 pp |

[MEASURED] Second anomaly at 4 GiB (`analysis/run-00-4G/heterogeneity.json`): uniform per-layer LRU 36,473 misses; "DP-optimal static split" 38,206 misses.

[MEASURED] Side observation that bounds the problem: the router trace (`R` records) is **bit-identical** across all nine runs and all three budgets.
- The model computes the same thing whatever the cache does, which is the expected correctness invariant.
- So every run at one budget replays the same input, and replay variance across repetitions is zero by construction.
- The 0.1–0.6 pp spread between repetitions is runtime-side.

## 3. Runtime semantics (as implemented in `patches/argine-store.cpp`, `argine-experts.cpp`)

The cache is a **set of independent per-shard LRUs over slices**. One slice is one kind (gate / up / down) of one expert of one layer.

| property | runtime behaviour |
|---|---|
| unit of residency | the slice `(layer, expert, kind)`; an expert can be partly resident |
| bytes charged | `need = round_up(slice_bytes, 4096)` (`psx_slice_read_bytes`). On the 30B all slice sizes (884,736 B; 1,290,240 B for Q6_K `down`) are already multiples of 4096, so alignment adds 0 B. No metadata is charged to the budget |
| expert sizes | vary by layer: 2,654,208 B (Q4_K down) or 3,059,712 B (Q6_K down) per expert, 3 slices each |
| number of shards | `n = clamp(budget / 32 MiB, 1, 16)`, or `ARGINE_EXPERT_SHARDS`. All EXP-007 budgets give 16 |
| per-shard budget | `budget / n`; the last shard gets the remainder. 4 GiB → 16 × 256 MiB |
| key → shard | `mix64(region_ptr · φ + offset) mod n`, where `region_ptr` is the **virtual address** of the tensor's reserved region. It differs per process (ASLR), so it is **not reproducible offline** from a v1 trace |
| hit | slot present and ready → pin, move to the front of the shard LRU, record `A … h` |
| miss | evict from the **tail** of the shard LRU, skipping pinned or not-ready slots, until `resident + need ≤ shard budget`. Each victim is recorded as an `E` record. Then insert at the **front** with `pins = 1, ready = false`, record `A … m`, issue the read. `completa` sets `ready = true`; on a read error it removes the slot (format v2 records `U`) |
| in-flight slot | present but not ready. Another requester waits on the condition variable, then re-runs the lookup; it is recorded `w` and counted as a hit. Not-ready slots are never evicted |
| nothing evictable | wait 1 ms, up to 2000 times, then fail. A failure is fatal for the node (`GGML_ABORT`) |
| node request order | in the **gate** node the layer's whole working set is requested: all three kinds, kind-major, experts in ascending id order, split across threads by an atomic counter. The **up** and **down** nodes request the same slices again and always hit |
| pin lifetime | **per thread** (`thread_local` list in argine-experts.cpp): a thread drops its pins at ITS next MoE node entry, before the node's barrier. [SUPERSEDED, 2026-10-04] The pinned set equals the slices requested in the current node only when one thread pool computes every graph. With `--threads ≠ --threads-batch` (Windows reference runs: 4 / 12), the threads of the idle pool keep the pins of the last node they ran (§13) |
| event atomicity | **none**: requests are served one slice at a time; earlier slices of the same node are protected only by their pins |
| eviction timing | at the miss, under the shard mutex, before insertion |
| insertion timing | at reservation (before the read completes) |
| initial state | empty; warm-up request and flush-sentinel request run through the same cache (they change state) |
| failed / duplicate accesses | a failed read is undone (no residual state); a duplicate request of a ready slot is an ordinary hit |
| runtime "expert hit" figure | gate-node slice hits / gate-node slice requests, workload only, prefill **and** decode (`summary.json` `expert_level_hit_rate`) |

[DERIVED] Consequences used below:
- [FALSIFIED on Windows, 2026-10-04, §13] **Pins never create skips in practice.** Pinned slices are the ones requested in the current node, which form the MRU prefix of the shard LRU, so the eviction scan reaches unpinned victims first.
  - [MEASURED] 0 skipped victims in the Linux v2 runs, including the 14 MiB / 16-shard run that evicts 2,339 times on ~875 KiB shards. This holds there because one pool runs everything.
  - Pins matter only for failures, of which there are none in EXP-007. This holds only for a single pool: on the Windows v2 run, 78,394 of 144,689 evictions passed over 1–4 pinned slots.
- **The `w` path (cause G) does not occur in the EXP-007 traces.** [MEASURED] run-00-4G has 1,325,036 `h`, 148,990 `m`, 0 `w`.

## 4. Replay semantics

**Before (v1-legacy, the `sharded-lru-16` row):**
- unit = expert, 3 slices as one;
- 16 shards by a Python hash of (layer, expert);
- event-atomic insertion;
- only decode steps served to the cache, plus warm-up decode counted;
- metric = expert hit over decode.

**After:**

`store-replay` (category **runtime reproduction**, [`storereplay.py`](../../argine/research/storereplay.py)) is a transcription of §3:
- slice units with exact runtime byte sizes;
- per-shard budgets computed exactly as in C++;
- per-shard LRU with tail eviction that skips pinned or not-ready slots;
- pins dropped at node entry;
- requests in the runtime's node order (gate node: 3 kinds × experts ascending; up and down nodes: same slices);
- every recorded step served (warm-up, prompt, decode, sentinel);
- metrics computed on workload steps only.

Shard map:
- **recorded**: format v2, exact;
- or **hash model**: v1, statistical. Same `mix64`, with a random per-(layer, kind) base standing in for the unknown address. Several seeds give the spread.

Metrics, all workload-only, each split by decode / prefill / all:
- `slice_gate`, the runtime's figure;
- `slice_all`, the runtime-style figure;
- `expert_event`, all three slices hit.

**Counterfactuals** keep expert units and event atomicity, because that is what makes them comparable with each other and with stack-distance curves. They now:
- serve every step;
- count only the requested phase;
- exclude warm-up and sentinel.

v1-legacy semantics remain selectable (`semantics="v1-legacy"`) only to reproduce superseded numbers.

## 5. First divergence

The validator ([`fidelity.compare`](../../argine/research/fidelity.py)) works as follows:
- It rebuilds the runtime's request stream from `A` records, attributing each to its `N` node, in `(shard, seq)` order (v2) or timestamp order (v1).
- It replays the stream request by request.
- It compares the outcome and the eviction victims at every request.

Each divergence is emitted with:
- the previous requests in the shard;
- the LRU tail with pin flags;
- resident bytes before and after;
- the shard budget;
- both decisions.

The full lists are in `experiments/EXP-021-replay-fidelity/processed/*/divergence-*.json`.

- **Linux v2 runs** (recorded shard map, 16 shards, mid-size MoE fixture, 24 MiB and 14 MiB): [MEASURED] **no divergence**. Every outcome and every victim is identical: 89,892 and 89,928 requests; 794 and 2,339 evictions; misses 1,171 / 2,563 equal on both sides.
- **C++ store driver** (`argine-store-fidelity-test.cpp`, 1–4 shards, unequal slice sizes, prompt steps): [MEASURED] no divergence (`TestRealStoreEquivalence`).
- **Windows v1 runs, run-00-4G**, first divergence at request 18,963 of 1,474,026 (the warm-up prompt):
  - layer 29, expert 12, gate slice, prompt step;
  - the runtime hit, while the replay with hash seed 0 missed and evicted (29, 33, gate);
  - in the replay the slice had been placed in a different shard, whose LRU had already pushed it out.

  The same request index (18,963 at 4 GiB; 6,390 / 6,474 / 6,621 / 6,534 at the smaller budgets) recurs across repetitions because the routing is identical and the seed-0 map is the same. That points to the **map**, not to the runtime.

[MEASURED] Control experiment proving the attribution:
- Replay the Linux v2 runs with the *hash model* instead of the recorded map. Disagreement goes from 0 to **0.64–0.75 %** (24 MiB) and 0.94–1.53 % (14 MiB), and the hit rate becomes slightly optimistic (+0.14…+0.23 pp at 24 MiB).
- That matches the 0.68–0.79 % and +0.10…+0.28 pp seen on the Windows v1 runs.
- Mapping-free consistency check of every v1 log (`fidelity.reconstruct`): 0 inconsistencies in 8 of 9 runs. In run-03-4G there is 1 in 1,617,888 records: a miss of (1, 119, down) with no recorded eviction since the previous miss of the same slice. v1 records cannot disambiguate it (no seq, no undo); v2 makes the same check exact.

## 6. Root cause — Question A

Each candidate cause from the task, with its verdict:

| cause | verdict | effect at 4 GiB (run-00) |
|---|---|---|
| **A. metric semantics** | **[SUPPORTED] main cause.** The runtime figure counts gate-node slice hits over prefill + decode; the replay figure counted expert hits over decode only. Prefill gate-slice hit rate is ≈ 9 % (thousands of first-touch experts per prompt), which pulls the all-phase figure down | 75.32 → 70.62 (−4.7 pp, almost the whole gap) |
| A′. warm-up counted | [SUPPORTED, small] the v1 replay counted the warm-up decode tokens | included above |
| **B. event stream** | [SUPPORTED, small] v1 replay never served prefill events to the state, so decode started from a cache the runtime never had | 75.32 → 75.43 (+0.1 pp) |
| C. sharding | [SUPPORTED, small] the old row hashed experts, not slices, with a different hash; the runtime map is address-dependent. Shard count and per-shard budget were already right | part of the residual 1.5 pp at matched metric |
| D. byte accounting | [FALSIFIED as a cause] exact sizes; 4096 alignment adds 0 B on this model; no metadata charged; effective capacity = 16 × 256 MiB minus < 1 slice of fragmentation per shard | 0 |
| E. event atomicity | [SUPPORTED, small] runtime is slice-at-a-time and non-atomic; an expert can be partially resident. [MEASURED] 4,563 decode expert events at 4 GiB (3.1 %) and 5,205 at 2 GiB found 1–2 of 3 slices resident | with C: 75.43 (expert-unit) vs 73.88 runtime, expert-event decode |
| F. initial / warm state | [SUPPORTED, folded into B] the runtime's workload starts after a warm-up request that filled the cache; v1 replay started empty at the first decode step | included in B |
| G. in-flight async state | [FALSIFIED as a cause here] modelled anyway (not-ready slots are not evictable; `w` = hit); 0 `w` records in EXP-007 | 0 |

Decomposition, run-00, [DERIVED]:
- `sharded-lru-16` v1 0.7532;
- + prefill served 0.7543;
- + runtime metric (gate, all phases, workload) 0.7062, runtime 0.7062.

The match of the expert-unit model on this one metric is partly coincidental: it is +1.5 pp too high on decode and too low on prefill. At matched expert-event decode it stays 1.5–2.1 pp above the runtime on all nine runs (table §9b). That residual is the slice granularity + non-atomicity + sharding of C/E. It is why the expert-unit row is **not** a runtime reproduction and has been renamed.

**Classification:** **metric** (dominant) + **replay** (event stream and state, granularity, sharding). Not a runtime bug and not an instrumentation bug: the runtime's records are internally consistent (§5), and the runtime does what its code says. The v1 instrumentation was insufficient for exact replay because it lacked shard id, sequence and undo records, which is now fixed in v2.

## 7. Root cause — Question B (optimal static vs uniform)

Audit of `allocator.dp_optimal` (old) and `elasticity.heterogeneity`:

| item | old implementation | now |
|---|---|---|
| objective | Σ_L misses_L(c_L), per-layer LRU stack-distance curves | same |
| decision variables | slots per layer c_L | same |
| budget constraint | Σ_L c_L · **round_up(size_L, unit)** ≤ budget, unit = budget / 8192 = **512 KiB** | Σ_L c_L · size_L ≤ budget, **exact bytes** |
| minimum per layer | `min_slots` = largest event in any layer, including the prompt steps the analysis does not count (up to 128) | largest **counted** event (8 at decode); larger served events overflow and are streamed |
| maximum | n_experts | same |
| curves | v1-legacy (decode served only) | v2 (all served, counted phase only), exact incl. overflowing prompt events |
| capacity | slot counts from rounded sizes | exact |

[DERIVED] Mechanism:
- 2,654,208-byte experts (24 of 48 layers) were charged 3 MiB (+18.5 %).
- The uniform allocation (29 or 33 slots per layer, 4,231,692,288 B exact) costs **4,680,843,264 B** under that charging, which is more than the 4,294,967,296-byte budget. So uniform was *outside the DP's feasible set*.
- The DP returned the best of a smaller set: 38,206 misses, using only 3.93 GB of the 4 GiB.
- It was a correct optimum of a **different, mis-specified constraint**, mislabelled as optimal for the real one.

[DERIVED] Fix verification on the identical v1-legacy curves (so the semantics change is not mixed in): the exact DP gives **34,091 misses vs uniform 36,473** (old: 38,206). With v2 curves: 33,131 vs 35,496 (4 GiB), 61,657 vs 64,268 (2 GiB), 71,234 vs 74,429 (1.5 GiB).

Method: per size class, an exact min-plus convolution of the per-layer curves; the size classes are then combined exactly. The checks:
- `dp_optimal_checked` raises `AllocationInvariantError` if the result exceeds any feasible baseline (uniform here).
- The analysis cannot emit an "optimal" worse than uniform.
- A brute-force test on unequal sizes pins exactness.

The row keeps the name `layer-aware-static` / `oracle-static-lru`, now that it is optimal, and stays labelled IN-SAMPLE / UPPER BOUND.

## 8. Fix

| change | file |
|---|---|
| runtime-reproduction replay (slice store, shards, pins, node order, metrics) | `argine/research/storereplay.py` (new) |
| event-level validator, first divergence with context, hash-map ensemble, mapping-free reconstruction | `argine/research/fidelity.py` (new), `tools/validate_replay_fidelity.py` (new), `argine research fidelity` |
| replay semantics v2: all steps served, counted phase on workload steps only; v1-legacy kept, selectable | `replay.py`, `oracle.py`, `elasticity.py`, `policies.py` (overflow streaming), `analysis.py` |
| exact allocator + invariant | `allocator.py` |
| categories never mixed: measured / runtime reproduction / counterfactual / counterfactual IN-SAMPLE / oracle; `sharded-lru-16` → `sharded-lru-16-expert` (counterfactual); oracle gaps computed for counterfactuals only | `analysis.py`, `report.py`, `cli.py` |
| measured runtime rows (mean ± sd over repetitions) in the policy table; StoreReplay row with hash-map spread | `analysis.compare_to`, `cli --runtime-runs`, `--slice-bytes-from` |
| instrumentation **format v2**: `A` and `E` carry shard id and a per-shard sequence number assigned under the shard mutex; `E` carries the skipped count; new `U` (undo) record; id `argine2-research-v2`. Parser reads v1 and v2 | `patches/argine-research.{h,cpp}`, `argine-store.cpp`, `argine-experts.cpp`, `rawlog.py`, `runner.py`, `fingerprint.py` |
| deterministic C++ driver of the real store for equivalence tests | `patches/argine-store-fidelity-test.cpp` (new) |
| Windows binaries with v2 instrumentation, in a **new** folder with its own BUILD-INFO (v1 folder untouched) | `llama-research-v2/` |

History preserved:
- raw runs untouched;
- `analysis/run-00-4G/` untouched plus `SUPERSEDED.md`, with the regenerated version in `analysis-v2/run-00-4G/`;
- pilot regenerated in `data/processed/pilot-legacy-30b-v2/`, next to the original.

## 9. Validation (all nine runs)

[SUPERSEDED in part, 2026-10-04] The AFTER columns below use the single-pool pin model. The per-thread table is §13.6: same disagreement, bias −0.07…+0.06 pp instead of +0.10…+0.28 pp.

[MEASURED] runtime, [DERIVED] replay. Windows v1 traces, hash-model shard map, 3 seeds. "Runtime hit" = runtime gate-slice hit, all phases, workload only, recomputed from `raw.tsv`. Columns:
- "BEFORE" = `sharded-lru-16` v1 as originally reported;
- "AFTER" = `store-replay`;
- "first divergence" = request index with hash seed 0;
- misses / bytes / evictions = whole-run slice totals.

`experiments/EXP-021-replay-fidelity/processed/30b-coding-dev/fidelity_runs.csv`.

| run | budget | runtime hit | BEFORE replay hit | BEFORE abs Δ | AFTER replay hit (± sd, 3 shard maps) | AFTER abs Δ | event disagreement | misses rt / rp | bytes read rt / rp (GB) | evictions rt / rp | effective capacity | first divergence (request #) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run-00-4G | 4 GiB | 70.62% | 75.32% | +4.71 pp | 70.72% ± 0.01 | +0.10 pp | 0.70% | 148,990 / 148,507 | 142.6 / 142.1 | 144,490 / 144,007 | 16 × 256 MiB | 18963 of 1,474,026 |
| run-01-2G | 2 GiB | 49.74% | 53.31% | +3.57 pp | 49.91% ± 0.05 | +0.17 pp | 0.75% | 251,438 / 250,634 | 240.8 / 240.0 | 249,209 / 248,400 | 16 × 128 MiB | 6390 of 1,474,884 |
| run-02-1.5G | 1.5 GiB | 41.55% | 45.06% | +3.51 pp | 41.66% ± 0.09 | +0.11 pp | 0.78% | 291,003 / 290,488 | 278.5 / 278.0 | 289,349 / 288,834 | 16 × 96 MiB | 6474 of 1,474,879 |
| run-03-4G | 4 GiB | 70.58% | 75.36% | +4.78 pp | 70.72% ± 0.01 | +0.14 pp | 0.68% | 149,093 / 148,407 | 142.7 / 142.0 | 144,594 / 143,910 | 16 × 256 MiB | 18963 of 1,473,294 |
| run-04-1.5G | 1.5 GiB | 41.44% | 45.05% | +3.61 pp | 41.66% ± 0.09 | +0.22 pp | 0.76% | 291,591 / 290,570 | 279.0 / 278.0 | 289,938 / 288,918 | 16 × 96 MiB | 6621 of 1,475,130 |
| run-05-2G | 2 GiB | 49.75% | 53.29% | +3.54 pp | 49.91% ± 0.05 | +0.16 pp | 0.73% | 251,563 / 250,787 | 240.9 / 240.2 | 249,329 / 248,555 | 16 × 128 MiB | 6390 of 1,475,394 |
| run-06-1.5G | 1.5 GiB | 41.38% | 45.04% | +3.66 pp | 41.66% ± 0.09 | +0.28 pp | 0.75% | 291,950 / 290,662 | 279.4 / 278.1 | 290,292 / 289,006 | 16 × 96 MiB | 6534 of 1,475,388 |
| run-07-4G | 4 GiB | 70.60% | 75.32% | +4.72 pp | 70.72% ± 0.01 | +0.12 pp | 0.79% | 149,100 / 148,507 | 142.7 / 142.1 | 144,599 / 144,008 | 16 × 256 MiB | 18963 of 1,474,023 |
| run-08-2G | 2 GiB | 49.75% | 53.29% | +3.55 pp | 49.91% ± 0.05 | +0.17 pp | 0.74% | 251,571 / 250,788 | 240.9 / 240.2 | 249,339 / 248,556 | 16 × 128 MiB | 6390 of 1,475,381 |

**(b) Matched metric, expert-event hit, decode, workload only.**

| run | runtime | store-replay | Δ | `sharded-lru-16-expert` (v2, counterfactual) | Δ vs runtime |
|---|---|---|---|---|---|
| run-00-4G | 73.88% | 73.98% | +0.10 pp | 75.43% | +1.55 pp |
| run-01-2G | 51.64% | 51.93% | +0.29 pp | 53.49% | +1.85 pp |
| run-02-1.5G | 43.13% | 43.18% | +0.06 pp | 45.13% | +2.01 pp |
| run-03-4G | 73.96% | 73.98% | +0.02 pp | 75.43% | +1.47 pp |
| run-04-1.5G | 43.11% | 43.18% | +0.07 pp | 45.13% | +2.03 pp |
| run-05-2G | 51.76% | 51.93% | +0.17 pp | 53.49% | +1.72 pp |
| run-06-1.5G | 43.07% | 43.18% | +0.11 pp | 45.13% | +2.06 pp |
| run-07-4G | 73.40% | 73.98% | +0.57 pp | 75.43% | +2.03 pp |
| run-08-2G | 51.68% | 51.93% | +0.25 pp | 53.49% | +1.80 pp |

**(c) Exact validation with format v2** (native Linux, mid-size MoE fixture, 6 layers × 32 experts, 16 shards, recorded map). In `experiments/EXP-021-replay-fidelity/raw/linux-mid-moe-v2/`, processed results next to it:

| run | budget | runtime hit | replay hit | abs Δ | event disagreement | misses rt / rp | bytes rt / rp | evictions rt / rp | skipped victims | first divergence |
|---|---|---|---|---|---|---|---|---|---|---|
| run-00-24M | 24 MiB (16 × 1.5 MiB) | 97.11% | 97.11% | 0 | **0 / 89,892** | 1,171 / 1,171 | 76,742,656 / 76,742,656 | 794 / 794 | 0 | none |
| run-01-14M | 14 MiB (16 × 0.875 MiB) | 93.20% | 93.20% | 0 | **0 / 89,928** | 2,563 / 2,563 | 167,968,768 / 167,968,768 | 2,339 / 2,339 | 0 | none |

The v1 `sharded-lru-16` on the same two runs: +1.37 pp and **−12.09 pp**. With 16 shards of < 1 MiB the expert-unit model is badly wrong, in either direction.

[DERIVED] Across budgets the runtime-reproduction replay predicts:
- the runtime's hit rate within +0.10…+0.28 pp;
- misses, bytes and evictions within 0.2–0.5 %.

The replay is systematically very slightly optimistic, exactly like the hash-model control on v2 data. On recorded maps it is exact.

## 10. Remaining limitations

[SUPERSEDED in part, 2026-10-04] The current list is §13.5 and §13.8. In particular, the Linux exactness did not transfer to Windows until pins were modelled per thread.

- [LIMITATION] **v1 traces cannot be replayed exactly.** The key→shard map depends on virtual addresses, which v1 does not record. Residual 0.7 % event disagreement, +0.1–0.3 pp hit rate. Format v2 records the shard id and is exact. The Windows v2 binary is built but **not yet run on Windows** (§12). *[Historical, SUPERSEDED: it ran on 2026-10-04 (§13, §14).]*
- [LIMITATION] Concurrency is replayed through the per-shard order the runtime recorded, not simulated. A *different* interleaving, for example with another thread count, cannot be predicted, only replayed. The cross-shard order inside a node does not affect decisions, because shards are independent.
- [LIMITATION] Read failures (`U`) and wait-on-in-flight (`w`) are modelled but were never exercised by EXP-007, which has 0 of each. They are tested on synthetic streams only.
- [LIMITATION] Counterfactual policies remain expert-unit and event-atomic by design: they answer "what would a different policy do". They are **not** comparable to the measured runtime number. Compare them to each other, to `sharded-lru-16-expert` (the runtime's policy in the same model) and to the oracles.
- [LIMITATION] One workload (coding dev), one model (30B). *[Superseded in part: the 80B coding-dev campaign was added in §14.2; EXP-021 itself still covers one workload family.]* The fidelity result is about the replay machinery and should transfer. The policy numbers do not.
- [OPEN] One unexplained record in run-03-4G (§5). It is 1 in 1.6 M, below any effect discussed, and v2 makes it checkable.

## 11. Research consequence

| earlier claim | verdict |
|---|---|
| "`sharded-lru-16` replay models the runtime" | [SUPERSEDED / FALSIFIED as a runtime reproduction] off by 3.5–4.8 pp, mostly metric. Renamed `sharded-lru-16-expert`, counterfactual |
| "replay predicts runtime misses within 10 %" (EXP-021 hypothesis) | [SUPPORTED, strengthened] `store-replay`: within 0.5 % (v1 traces), exact (v2) |
| pilot and EXP-007 "DP-optimal static saves X % vs uniform" | [SUPERSEDED] 4 GiB 30B coding: −4.5 % (worse!) → **uniform excess 7.1 %** (2 GiB 0.9 % → 4.2 %, 1.5 GiB 3.0 % → 4.5 %). Pilot: 9.0 / 6.1 / 5.0 / 5.3 % → 10.1 / 6.4 / 5.2 / 5.4 %. Layer heterogeneity of memory value is **strengthened** |
| layer-aware static is worse than uniform at 4 GiB | [SUPERSEDED] was the allocator bug; in-sample it is the best static policy at 2 and 1.5 GiB and second at 4 GiB (after static-hot-per-layer) |
| adaptive-e64 beats every baseline | [WEAKENED, unchanged in sign] vs expert-unit LRU baselines (same trace semantics): +1.2 pp vs global LRU and +1.2 pp vs per-layer uniform at 4 GiB; +1.1 pp vs uniform at 2 GiB; +1.4 pp at 1.5 GiB. It is **below** the in-sample layer-aware static (0.4–0.8 pp), now correctly computed. No claim against the measured runtime is made |
| hit rates of all counterfactuals (v1 semantics) | [SUPERSEDED, small] ±0.1–0.6 pp from serving prefill and excluding warm-up; rankings unchanged except layer-aware static |
| routing characterization (entropy, EWS, reuse, drift, co-activation) | [SUPPORTED, unchanged] does not depend on the cache; regenerated values identical |
| runtime expert-level hit ≈ 1 − 3 × runtime-style miss (EXP-020) | [SUPPORTED] gate-slice semantics confirmed record-for-record |
| [NEW, DERIVED] the runtime's slice-sharded design costs about 1.5–2 pp of expert-event hit vs the same LRU on whole experts, and ~1.8 pp vs a global expert LRU at 4 GiB | from `store-replay` vs `sharded-lru-16-expert` and `global-lru` on identical semantics |

What may now be claimed, and how:
- An algorithmic improvement is the difference between two replays in the same category on the same trace semantics.
- An improvement *over the runtime* is measured against `store-replay`, which is validated against the runtime.
- Not against the raw measured number.

## 12. Next experiment (exact)

*[Historical plan (2026-10-03), executed 2026-10-04: the trace is `raw/30b-coding-dev-v2`. Its first validation failed and is explained in §13; the fresh confirmation passed (§14.1).]*

One short Windows trace with the v2 instrumentation, to make the 30B equivalence exact instead of statistical:
- one run, 4 GiB, coding dev;
- nothing else changes.

```bat
cd C:\path\to\Argine2
set ARGINE_LLAMA_DIR=%CD%\llama-research-v2
argine.bat doctor
argine.bat research trace C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf --expert-store --budgets 4G --repeat 1 ^
    --workload coding --split dev --out experiments\EXP-021-replay-fidelity\raw\30b-coding-dev-v2
argine.bat research fidelity experiments\EXP-021-replay-fidelity\raw\30b-coding-dev-v2 ^
    --out experiments\EXP-021-replay-fidelity\processed\30b-coding-dev-v2
```

Acceptance:
- `shard_map = recorded (format v2)`;
- `event_disagreement_rate = 0`;
- `reconstruct_inconsistencies = 0`;
- identical misses, bytes and evictions;
- routing identical to the nine v1 runs.

The model path is the one EXP-007 used (`summary.json` → `expert_store.effective`: `C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.psxs`, resolved automatically next to the GGUF). If the result is 0, the 30B Windows equivalence becomes exact rather than statistical, and the policy campaign can resume on `store-replay` as the runtime baseline. If it is not 0, the first-divergence report says where, and the campaign stays blocked.

---

## 13. Windows v2 follow-up (2026-10-04)

### 13.1 What the first Windows v2 run showed

[MEASURED] `raw/30b-coding-dev-v2/run-00-4G`: 30B, 4 GiB, coding dev, format v2, 16 shards, recorded shard placement.

| check | 2026-10-03 replay | after the fix |
|---|---|---|
| reconstruct inconsistencies | 0 | 0 |
| shards with sequence gaps / with **replay-relevant** gaps | 3 / not classified | 3 / **0** |
| incomplete MoE nodes (all / in workload) | not checked | 1 / **0** |
| event disagreement | **5.29%** | **0** (0 of 1,474,909) |
| outcome / victim disagreements | 739 / 78,039 | 0 / 0 |
| evictions that passed over pinned slots, runtime / replay | 78,394 / 0 | 78,394 / 78,394 (identical skip count on every eviction) |
| misses runtime / replay | 149,191 / 148,742 | 149,191 / 149,191 |
| miss bytes runtime / replay | 142,739,288,064 / 142,301,896,704 | 142,739,288,064 / 142,739,288,064 |
| evictions runtime / replay | 144,689 / 144,240 | 144,689 / 144,689 |
| gate-slice hit, all phases, workload | 0.70581 / 0.70676 | 0.70581 / 0.70581 |
| budget violations, failures | 0, 0 | 0, 0 |

Storage:
- The failed result is kept unchanged in `processed/30b-coding-dev-v2/`, with `FAILED-VALIDATION.md` added.
- The re-analysis is in `processed/30b-coding-dev-v2-per-thread-pins/`.
- `processed/30b-coding-dev-v2-single-pool-reproduction/` reruns the old model and reproduces the failed 0.052928011 exactly. This confirms the pin model is the only difference.

### 13.2 The first divergence, explained

At request 60,351 (graph 10 = prompt of `code-dev-1`, layer 18 gate node, shard 2), thread 2 missed on (18, 38, down).
- The replay's LRU tail was (47, 85, down), then (0, 36, gate).
- The runtime evicted (0, 36, gate) with `skipped = 1` in its `E` record: it passed over (47, 85, down) because that slot was **pinned**.
- [MEASURED] The pin holder was **thread 12**. It requested (47, 85, down) at seq 3475, in the layer-47 down node of **graph 9**, the last decode graph of the warm-up request (N record `nth = 4`).
- Between that node and the divergence, the trace contains 55 nodes, all of graph 10, all with `nth = 12`. Thread 12 entered none of them, so it never dropped its pin.

Why: the store's pins live in a `thread_local` list (`argine-experts.cpp`, `Contesto::miei_pin`). Each thread releases its own list when it enters its next MoE node, before the barrier. llama-server was started with `--threads 4 --threads-batch 12`.

[MEASURED] From the records:
- prompt graphs (`nth = 12`) are computed by threads {0, …, 11};
- decode graphs (`nth = 4`) by threads {4, 12, 13, 14}. Thread 4, the main thread, belongs to both.

So:
- threads 12–14 hold the slices of the last decode node through every prompt;
- threads 0–3 and 5–11 hold the slices of the last prompt node through every decode step.

The replay modelled "every thread drops its pins at every node", which is true only when one pool computes every graph.

[MEASURED] The effect is large on the runtime's decisions and small on its hit rate:
- 66,829 of 110,407 decode evictions and 11,565 of 34,282 prompt evictions passed over 1–4 slots pinned by the idle pool;
- yet the wrong model moved the hit rate by only +0.09 pp.

A wrong victim changes recency order for a long time, which is why 5.29% of decisions differed while the aggregate barely moved.

### 13.3 Why Linux reached 0 and Windows did not

| | Linux EXP-021 runs | Windows reference runs |
|---|---|---|
| `--threads` / `--threads-batch` | 4 / 4 | 4 / 12 (the planner gives prompt processing every logical core) |
| thread pools seen in the trace | one pool computes every graph | two pools, sharing only the main thread |
| per-thread pins = "pins of the current node" | yes | no |
| C++ driver (tests) | one thread | one thread (now also two: `pools = 2`) |

It is a configuration difference, not an OS difference. The Linux validation could not see it, because with one pool the two pin models make identical decisions. The single-thread driver could not see it either.

### 13.4 The sequence gaps, record by record

[MEASURED] Missing numbers:
- shard 1, seq 97,557;
- shard 4, seq 101,206;
- shard 8, seq 109,501.

There are 0 duplicates. All three lie between records at t = …656.19–656.20 s and t = …657.82 s of steady time, inside the **flush-sentinel** request. Graph 395 is its first decode step, and the trace ends at its layer 23.

Independent check: compare every MoE node's requests with its router decision (`fidelity.node_completeness`). Exactly **1 of 56,951 nodes** is incomplete: the **up** node of graph 395, layer 23. It is missing 5 re-pin requests: (23, 3, gate), (23, 3, down), (23, 30, up), (23, 30, down), (23, 56, gate).
- All five slices were requested in the gate node of the same layer, so they were resident, and all five were hits.
- The node shows records from threads 4, 12 and 14 but **none from thread 13**. Thread 13 also issued gate-node requests of that layer.
- Three of the five sit between surviving records of their shard: these are the three gaps. The other two would have been the last numbers of shards 1 and 3, which no gap check can detect.

Mechanism:
- Every thread writes its buffered records into the trace file at the end of each node.
- The file is `fflush`ed by thread 0 at most every 250 ms.
- llama-server is terminated, not exited, after the flush-sentinel response.
- Thread 13's last write was still in the unflushed tail of the file buffer. This is the tail loss the sentinel exists to absorb (FH-024).

The user's eight hypotheses:

| # | hypothesis | verdict |
|---|---|---|
| 1 | sequence numbers incremented for records that are not emitted | **no**. Every `seq++` is a hit, an eviction, an insertion or an undo, and each emits its record while the trace is active. Checked at every call site of `prenota`/`completa` in `argine-store.cpp` |
| 2 | records dropped by the buffering path | **yes, at termination only**: the stdio tail after the last `fflush` (above). 5 records, all in the sentinel |
| 3 | concurrent Windows writes reorder or lose records | no loss while running. Whole thread buffers are written under one mutex (no interleaved lines), 0 malformed lines, 0 duplicates. Cross-thread order in the file is irrelevant because replay orders by (shard, seq) |
| 4 | a state transition intentionally untraced | **yes, pin releases (unpin).** They do not take a sequence number, so they cause no gap. They are what the replay got wrong (§13.2), and they are derivable exactly from what is recorded (§13.5) |
| 5 | numbering covers replay-irrelevant events | no. Every numbered event changes the shard LRU |
| 6 | asynchronous completion / rollback missing | no. Read completions do not change LRU state; rollbacks would be `U` records: 0 here, 0 failed reads |
| 7 | pin/unpin or LRU touches not represented | touches are the `A` records. Unpins: see 4 |
| 8 | replay eviction eligibility differs from the runtime | **yes, this was the divergence**. Eligibility = not pinned by *any* thread; the replay ignored pins held by threads outside the current pool |

**Classification:** the gaps are 3 replay-irrelevant record losses at process termination, after the last workload record of each affected shard (`shards_with_replay_relevant_seq_gaps = 0`). The first victim divergence is explained independently: it lies 1.4 M requests earlier and has nothing to do with the gaps.

### 13.5 Fix

No new instrumentation is needed. The minimal information is already recorded:
- every `A` record carries the requesting thread;
- every `N` record carries the node's thread count (`nth`), which identifies the pool computing the graph.

A pool's members are the threads seen requesting in nodes of that pool. A pool member that never requests holds no pins, so it cannot be missed. All of a pool's releases happen at node entry, before the barrier that precedes the node's first request. So the replay knows, at every request, exactly which slices are pinned by whom.

Changes:
- `StoreReplay` keeps pin counts per slice and pin lists per thread. `begin_node(threads)` releases only the entering threads' pins. `begin_node()` with no argument keeps the single-pool behaviour.
- `fidelity.load_runtime_stream` infers the pools; `fidelity.compare(pin_model="per-thread" | "single-pool")`.
- The validator also checks the **skip count** of every eviction against the runtime's `E … skipped`.
- Record integrity checks:
  - `fidelity.seq_gaps` classifies every missing number as replay-relevant or not;
  - `fidelity.node_completeness` finds lost requests that no gap check can see.
- The validator reports the EXP-021 closure verdict per format-v2 run.
- **Synthetic streams** (`analysis --slice-bytes-from`) have no thread information.
  - With split pools (read from the run manifest's `--threads` / `--threads-batch`), each pool is modelled as one thread.
  - On the Windows v2 trace this approximation leaves 1.25% event disagreement and −0.01 pp, against 5.29% and +0.09 pp for one pool. The `store-replay` row carries its `pin_model`.
- The C++ driver gains `pools = 2`: prompt steps on one persistent thread and decode steps on another, serial and deterministic. The test requires 0 disagreements, identical skip counts, more than 50 evictions past pinned slots, and a failing single-pool model (so the test is sensitive).

[LIMITATION] Pools are told apart by thread count. Two different pools with the same thread count would be merged. The validator would then show disagreements; it would not hide them.

### 13.6 Effect on the nine v1 runs

[DERIVED] Shard map modelled, 3 seeds, per-thread pins (`processed/30b-coding-dev-per-thread-pins/`; the 2026-10-03 table in §9 is kept). Format-v1 `E` records have no skip count, so only the replay's count is shown.

| run | budget | runtime hit | replay, single-pool pins (03 Oct) | Δ | replay, per-thread pins | Δ | event disagreement | misses rt / rp | evictions rt / rp | replay evictions past pinned |
|---|---|---|---|---|---|---|---|---|---|---|
| run-00-4G | 4 GiB | 70.62% | 70.72% | +0.10 pp | 70.62% ± 0.01 | +0.01 pp | 0.70% | 148,990 / 148,964 | 144,490 / 144,464 | 74,245 |
| run-01-2G | 2 GiB | 49.74% | 49.91% | +0.17 pp | 49.72% ± 0.04 | −0.02 pp | 0.73% | 251,438 / 251,573 | 249,209 / 249,340 | 173,551 |
| run-02-1.5G | 1.5 GiB | 41.55% | 41.66% | +0.11 pp | 41.48% ± 0.08 | −0.07 pp | 0.73% | 291,003 / 291,325 | 289,349 / 289,671 | 190,856 |
| run-03-4G | 4 GiB | 70.58% | 70.72% | +0.14 pp | 70.61% ± 0.01 | +0.03 pp | 0.67% | 149,093 / 148,930 | 144,594 / 144,433 | 82,040 |
| run-04-1.5G | 1.5 GiB | 41.44% | 41.66% | +0.22 pp | 41.43% ± 0.08 | −0.01 pp | 0.71% | 291,591 / 291,661 | 289,938 / 290,009 | 212,569 |
| run-05-2G | 2 GiB | 49.75% | 49.91% | +0.16 pp | 49.72% ± 0.04 | −0.03 pp | 0.70% | 251,563 / 251,724 | 249,329 / 249,492 | 171,368 |
| run-06-1.5G | 1.5 GiB | 41.38% | 41.66% | +0.28 pp | 41.44% ± 0.09 | +0.06 pp | 0.70% | 291,950 / 291,707 | 290,292 / 290,051 | 204,649 |
| run-07-4G | 4 GiB | 70.60% | 70.72% | +0.12 pp | 70.61% ± 0.01 | +0.01 pp | 0.78% | 149,100 / 149,027 | 144,599 / 144,528 | 81,138 |
| run-08-2G | 2 GiB | 49.75% | 49.91% | +0.17 pp | 49.75% ± 0.04 | +0.00 pp | 0.72% | 251,571 / 251,559 | 249,339 / 249,327 | 164,173 |

[DERIVED]
- The systematic optimism of +0.10…+0.28 pp is gone (−0.07…+0.06 pp).
- The remaining ~0.7% disagreement is the unrecorded shard map, as before.

Correction to §5: the 0.7% of the v1 Windows runs contained both effects, the pin model and the shard map. With the pin model corrected it is still ≈ 0.7%: the modelled shard map dominates the decision-level disagreement, while the pin model was what biased the hit rate.

### 13.7 A runtime observation (recorded, not acted on)

[MEASURED] With two pools, the idle pool's threads keep up to one node's slices pinned for a whole phase. The prompt pool keeps the last prompt node's slices (the layer-47 slices of 1 token, up to 24 slices) through all of decode. The decode pool keeps its last node's slices through the next prompt.
- [DERIVED] About 25 MB of 4 GiB cannot be evicted, and 60% of decode evictions scan past 1–4 such slots.
- [OPEN] Whether this matters for speed is not investigated. This phase changes no runtime behaviour.

### 13.8 Closure status

On the failed trace itself, with the corrected replay, every criterion holds:
- reconstruct inconsistencies = 0;
- replay-relevant sequence gaps = 0;
- incomplete workload nodes = 0;
- event disagreement = 0;
- misses, miss bytes and evictions identical;
- budget violations = 0, failures = 0.

The model was corrected on this same trace, though, so it is not an independent test. **EXP-021 stays RUNNING** until one fresh short Windows run passes: *[Historical (2026-10-04); resolved in §14.1: the fresh run passed and EXP-021 closed.]*

```bat
cd C:\path\to\Argine2
set ARGINE_LLAMA_DIR=%CD%\llama-research-v2
argine.bat research trace C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf --expert-store --budgets 4G --repeat 1 ^
    --workload coding --split dev --out experiments\EXP-021-replay-fidelity\raw\30b-coding-dev-v2-confirm
argine.bat research fidelity experiments\EXP-021-replay-fidelity\raw\30b-coding-dev-v2-confirm ^
    --out experiments\EXP-021-replay-fidelity\processed\30b-coding-dev-v2-confirm
```

Pass = the row's `exp021_closure` column reads `PASS`. That column checks:
- reconstruct inconsistencies 0;
- replay-relevant sequence gaps 0;
- incomplete workload nodes 0;
- event disagreement 0;
- identical misses, miss bytes and evictions;
- budget violations 0, failures 0.

The binary is unchanged, so the run tests the replay on unseen data.

---

## 14. Independent confirmation and the 80B campaign (2026-10-04)

### 14.1 30B confirmation run

[MEASURED] `raw/30b-coding-dev-v2-confirm/run-00-4G` was a fresh run, not used to build the model.

- **0 of 1,474,817 decisions differ.**
- Misses 149,586 = 149,586; miss bytes 143,188,107,264 = 143,188,107,264; evictions 145,081 = 145,081.
- 71,863 evictions passed over pinned slots, all with identical skip counts.
- One record was lost at termination, inside the flush sentinel; it was not followed by any workload record.
- Closure PASS.

[SUPPORTED] The per-thread pin model holds out of sample.

### 14.2 80B campaign: the two strict failures

Campaign: Qwen3-Next-80B-A3B Q3_K_M, coding dev, 4 / 2 / 1.5 GiB × 3, format v2, `--threads 4 --threads-batch 12`, 16 shards. Seven runs pass the strict criterion exactly; run-04-1.5G and run-07-4G fail it.

Correction to the summary that came with the campaign: the validator's own files report **0 outcome disagreements and 2 / 4 victim disagreements**, not the reverse. Every disagreement is a miss whose eviction victim differs; no hit/miss decision differs.

Each disagreement, identified by request:

| run | request # (of total) | graph | node (layer, kind) | phase / pool | thread | shard / seq | segment | runtime | replay | record sequence of this shard before the request |
|---|---|---|---|---|---|---|---|---|---|---|
| run-04-1.5G | 1,917,982 (1,917,998) | 400 | layer 7, gate | decode / nth 4 | 14 | 6 / 143,615 | **flush sentinel** | miss (7, 247, up), no eviction | miss, evicts (0, 494, down) | **incomplete**: seq 143,611–143,613 lost |
| run-04-1.5G | 1,917,995 | 400 | layer 7, gate | decode / nth 4 | 1 | 14 / 142,352 | **flush sentinel** | miss (7, 247, down), evicts (2, 350, gate) | miss, evicts (1, 182, down) | **incomplete**: seq 142,347–142,348 lost |
| run-07-4G | 1,917,083 (1,917,104) | 399 | layer 47, gate | prompt / nth 12 | 5 | 1 / 134,689 | **flush sentinel** | miss (47, 284, up), evicts (5, 281, down) | miss, evicts (5, 398, up) | **incomplete**: 5 seqs lost (134,681 … 134,687) |
| run-07-4G | 1,917,093 | 399 | layer 47, gate | prompt / nth 12 | 1 | 11 / 136,746 | **flush sentinel** | miss (47, 467, gate), evicts (8, 85, down) | miss, evicts (4, 216, down) | **incomplete**: 7 seqs lost (136,737 … 136,744) |
| run-07-4G | 1,917,094 | 399 | layer 47, gate | prompt / nth 12 | 1 | 11 / 136,748 | **flush sentinel** | miss (47, 33, down), evicts (9, 249, gate) | miss, evicts (5, 398, gate) | **incomplete** (same shard) |
| run-07-4G | 1,917,096 | 399 | layer 47, gate | prompt / nth 12 | 1 | 12 / 125,831 | **flush sentinel** | miss (47, 234, down), evicts (3, 104, gate) and (5, 281, gate) | miss, evicts (2, 22, gate) | **incomplete**: 7 seqs lost (125,820 … 125,828) |

[MEASURED] Facts common to all six:
- **Position:** each is in the run's **only** incomplete MoE node: graph 400, layer 7, gate in run-04 (4 requests missing), and graph 399, layer 47, gate in run-07 (64 requests missing). Both nodes are inside the flush sentinel.
- **Timing:** each comes **after** the last workload request. The last workload request is #1,894,544 in both runs, 22,500+ requests earlier. No workload record exists after the lost records in the same shard (`shards_with_replay_relevant_seq_gaps = 0`).
- **Record loss:** each is in a shard whose own record sequence had already lost numbers before that request. The lost records are 5 (run-04) and 47 (run-07): the requests and evictions of the threads whose last buffers were still unflushed when llama-server was terminated (the same mechanism as §13.4). Run-07 was cut during the sentinel's **prompt** graph, which is computed by 12 threads, so more buffers were lost.
- **Mechanism:** a lost `A … m` record means a slice the runtime inserted that the replay never saw, together with its own evictions (`E` records lost too). That shard's LRU content then differs, and the next miss in that shard picks a different victim. Shards are independent, so nothing propagates beyond the shard that lost records.
- **Counts:** this also explains the ±1 eviction totals. In run-04 the replay evicts once more than the records show; in run-07 once less, because the runtime's evictions belonging to the lost requests are themselves lost.

**Root cause:** record loss at process termination, inside the flush sentinel, after the measured workload. It is not a replay error and not a runtime error.

### 14.3 Closure semantics: three verdicts instead of one

The strict criterion asked for equality over **every** record of the server's lifetime. That includes the flush sentinel, the artificial request issued only to absorb the termination tail (FH-024). A trace cut by termination cannot be complete there by construction, so the strict verdict mixes two questions: whether the replay is right, and whether the last records survived the kill.

The validator now answers three questions separately. Nothing is hidden, and every verdict is in every row.

| verdict | what it requires | covers |
|---|---|---|
| `full_trace_fidelity` | the original strict criterion, unchanged | every record (diagnostic) |
| `workload_fidelity` | 0 disagreements (outcome, victim, skip count) over the workload **and over everything before its last request** (warm-up, between requests); identical workload misses, miss bytes and evictions; 0 replay-relevant sequence gaps; 0 incomplete workload nodes; 0 budget violations; 0 replay failures in workload / pre-workload | the scientific measurement |
| `termination_tail_completeness` | records after the last workload request: lost records, incomplete nodes, tail disagreements, each classified as **explained** (its shard had lost records with lower sequence numbers) or **unexplained** | the termination tail |

`exp021_closure = workload_fidelity PASS and 0 unexplained tail disagreements`. Why this does not weaken the test:
- The pre-workload region (warm-up) is held to the same strictness as the workload, because a difference there would carry into the measured state.
- Only records after the last workload request can be exempt, and only one by one, and only when that same shard had already lost records.
- An unexplained difference in the tail still fails closure: it would be a replay error.
- The strict verdict stays in every row.

Tests:
- an exact stream has no disagreement in any region;
- a tail disagreement after record loss is explained;
- a tail disagreement without loss, or with loss only after it, is unexplained;
- a warm-up disagreement fails the workload verdict;
- the workload verdict fails on any eviction or node-completeness mismatch.

### 14.4 Regenerated results

[DERIVED] The two runs, regenerated with the region-aware validator (`fidelity/80b-coding-dev-v2-regions/` on the reference machine; the original `fidelity/80b-coding-dev-v2/` is kept):

| run | workload requests | workload disagreements | pre-workload disagreements | workload misses rt / rp | workload miss bytes rt / rp | workload evictions rt / rp | tail | full trace | closure |
|---|---|---|---|---|---|---|---|---|---|
| run-04-1.5G | 1,841,310 | 0 | 0 (53,235 requests) | 326,277 / 326,277 | 163,561,070,592 / 163,561,070,592 | 323,060 / 323,060 | 5 lost records, 1 incomplete node, 2 disagreements (2 explained) | FAIL (diagnostic) | **PASS** |
| run-07-4G | 1,841,310 | 0 | 0 (53,235 requests) | 209,188 / 209,188 | 104,892,129,280 / 104,892,129,280 | 205,476 / 205,476 | 47 lost records, 1 incomplete node, 4 disagreements (4 explained) | FAIL (diagnostic) | **PASS** |

The full 9-run table is in `fidelity/80b-coding-dev-v2-regions/fidelity_runs.csv`. The per-disagreement audit is in `tail-audit-run-04-1.5G.json` and `tail-audit-run-07-4G.json`.

[MEASURED, 80B, runtime] Gate-slice hit, all phases, workload (3 runs each):
- 4 GiB: 65.97 / 65.91 / 65.92%
- 2 GiB: 52.49 / 52.47 / 52.40%
- 1.5 GiB: 46.84 / 46.84 / 46.85%

On every run the replay equals the runtime to the last request of the workload.

### 14.5 Consequences

- [SUPPORTED] **EXP-021 closes.** It is exact on Linux; exact out of sample on the 30B Windows confirmation run; and exact over the workload of all nine 80B runs. Lost records are confined to the termination tail and always detected.
- **The 80B coding-dev slice of EXP-007 is scientifically valid.** No run needs to be rerun: rerunning would reproduce the same termination tail, which by design holds no measurement.
- [LIMITATION] The flush sentinel absorbs the termination tail only if the kill happens during it. A run in which the last workload request itself loses records would show replay-relevant gaps or incomplete workload nodes, and would fail `workload_fidelity`. The validator checks this on every run.
