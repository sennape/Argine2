# Architecture

## 1. Layers

```
                ┌──────────────────────────────────────────────────────────┐
  user / agent  │ argine (Python)                                          │
  ───────────►  │  planner · hardware probe · runtime manager · REST API   │
                │  argine research ... (argine/research)                   │
                └───────────────┬──────────────────────────────────────────┘
                                │ spawns, sets ARGINE_* environment
                ┌───────────────▼──────────────────────────────────────────┐
                │ llama-server (llama.cpp + Argine patches)                │
                │   MUL_MAT_ID on expert tensors ──► expert buffer type    │
                │        argine-experts.cpp  (compute_forward hook)        │
                │        argine-store.cpp    (residency, hard budget, I/O) │
                │        argine-research.cpp (observation only)  [0003]    │
                └───────────────┬──────────────────────────────────────────┘
                                │ unbuffered reads (FILE_FLAG_NO_BUFFERING, overlapped)
                         ┌──────▼──────┐
                         │ .psxs store │  expert slices, each 4096-aligned
                         └─────────────┘
```

## 2. The expert store (Argine v1, patch 0002)

- **Format (PSXS v2).** 256-byte header, layer table, then `n_layer × n_expert` entries of three (offset, size) pairs — gate, up, down. Every slice is 4096-aligned on its own, so each read lands directly in the tensor's memory without staging.
- **Residency.** The address space of each expert tensor is reserved in full; physical memory is committed per slice on load and decommitted on eviction. The pointer ggml sees stays valid.
- **Hard budget.** A cache of slices keyed by (region, offset), split into up to 16 shards by hash (fmix64), each an LRU with `budget / n` bytes; admission evicts unpinned, ready slices of the same shard until the new slice fits. The sum of shard budgets is exactly the budget. Slices in use are pinned; slices being read are in a third state ("loading") and cannot be evicted.
- **Hook.** `compute_forward` of the buffer type does not compute: it reads the router's chosen ids, makes the slices resident for **all three kinds of the layer** (so the up and down nodes find them), waits at a barrier, and returns `false` so ggml runs its standard kernel on the filled memory.
- **Read modes.** `sincrono` (one slice at a time), `blocco:n` (fixed blocks, all-or-nothing), `asincrono` (emit-and-continue: issue the read, keep taking slices, wait for all own reads before the barrier).

Consequences that matter for research:

1. No matrix compute overlaps the reads of the same node: the barrier precedes the kernel. "Hidden I/O" can only hide behind other residency work of the node.
2. The up and down nodes re-request slices loaded by the gate node; they are guaranteed hits. The runtime-style hit counter (all three nodes) therefore over-states the gate-slice hit rate stored as `expert_level_hit_rate`. On this store the miss ratio is exactly 3 (EXP-020, confirmed in 252 qualified EXP-007 runs). The replay's expert-event hit rate is a third, separate metric (METHODOLOGY §4).
3. The policy is hash-sharded LRU over slices, not a global LRU over experts.

## 3. Research instrumentation (Argine2, patch 0003)

Activated by `ARGINE_RESEARCH_TRACE=<path>` (`%p` expands to the process id). Without it every hook is one relaxed atomic load.

| record | written by | content |
|---|---|---|
| `H` | open | format, version, paired steady/system clocks, pid |
| `C` | init | store, budget, read mode, direct I/O, shards, layers, experts, pretouch |
| `R` | thread 0, once per (forward pass, layer) | graph index, time, layer, tokens in the ubatch, experts per token, ids per token in router order |
| `A` | any thread, per slice request | time, thread, layer, expert, kind, outcome (h hit, m miss issued, w waited for another thread's read), bytes, request id |
| `I` | the issuing thread, per read | request id, layer, expert, kind, bytes, issue, completion, wait start, wait end, ok, flags |
| `E` | the evicting thread | time, layer, expert, kind, bytes, loaded, last use, uses |
| `N` | thread 0, per MoE node | graph, layer, kind, entry time, time after the post-read barrier, experts requested, threads |
| `S` | thread 0, at each forward-pass boundary | resident bytes, budget, hits, misses, evictions, bytes read, budget violations |

**Exactness.** On Windows, while tracing, the wait for a thread's overlapped reads uses `WaitForMultipleObjects` one completion at a time, stamping the signalled read (`COMPLETE_EXACT`) and polling the others (`COMPLETE_BOUND`, upper bound); reads are also polled with `HasOverlappedIoCompleted` each time the thread issues a new read, so reads finished before the wait get a tight bound. Synchronous and fallback reads are flagged. The parser and the I/O timeline label every result built on bounds. The path has run on Windows: 66.6% exact completions in the 2026-10-03 smoke run and 63.4% in the 2026-10-12 EXP-019 validation, so Windows queue depth is in practice an upper bound.

**Flushing.** Records go to a per-thread 64 KiB buffer, appended to the file under one lock at the end of every node, and thread 0 flushes the stdio buffer at most every 250 ms. Argine terminates llama-server instead of letting it exit, so nothing may depend on destructors.

**Flush sentinel.** Thread 0 flushes the stdio buffer at a node boundary only when ≥250 ms have passed, and llama-server is terminated, not exited: the records of the last ≤250 ms would be lost (observed on the first Windows trace). The harness therefore sends one tiny request (segment `flush-sentinel`) >250 ms after the workload; its first node flushes everything before it, and its own records are excluded from statistics. Those own records can still be lost at termination: in EXP-019 (2026-10-12), 5 per-shard sequence numbers and one node of the sentinel were lost, none in the workload. The replay-fidelity validator reports this as `termination_tail_completeness`. Per-request completeness is checked against llama.cpp's `predicted_n`.

**Graph boundaries.** A new forward pass is detected when the layer index decreases. Prompt steps carry many tokens in every layer except the last, where llama.cpp computes only the output token(s): step size is the maximum over layers.

**Invariant monitor.** `violazioni` counts admissions after which a shard's resident bytes exceeded its budget. It must be 0; it is exported so each run proves it.

**Verification.** `patches/argine-research-test.cpp` (TSan, ASan/UBSan): identical statistics with and without tracing, one read record per miss, ordered timestamps, all evictions logged, no violations with 6 threads. End-to-end with a natively built llama-server (Linux fixture) and, on 2026-10-12, with the cross-compiled research binary on Windows: EXP-019.

## 4. Python research layer (`argine/research`)

```
raw.tsv ──rawlog──► RawLog ──► trace.jsonl.gz (trace.py)  ──► metrics  (characterization)
                         │                               ├─► replay + policies + oracle
                         │                               ├─► elasticity + allocator
                         │                               └─► controller (replay prototype)
                         ├─► iotimeline (Q(t) exact or upper bound, latency, hidden/unhidden, blocked time)
                         ├─► node timing (residency phase vs remainder per forward pass)
                         └─► hit-rate semantics (runtime-style vs expert-level)
runs ──► predictor (T_token model) ;  analysis ──► report (REPORT.md + figures)
```

- **Runtime reproduction (not event-atomic).** `storereplay.StoreReplay` follows the store's real contract: slices, per-shard LRU, per-thread pins (a thread drops its own pins when it enters a node of its pool; prompt and decode graphs may run on different pools), tail eviction skipping pinned or loading slots, and the gate node requesting all kinds in k order. Validated against runtime records by `fidelity.compare` (EXP-021).
- **Counterfactual replay contract (event-atomic).** An event = (step, layer, set of experts). All members must be resident at once; no member can be evicted for another; members processed in ascending id. Overflow (event larger than the capacity a policy may use) is streamed and counted, never hidden.
- **Policies.** global LRU, sharded LRU (runtime model), per-layer LRU (any allocation), LFU (optional decay), 2Q, Least-Stale (two variants), static pinned / hot set (fitted offline), layer-aware static (DP on LRU curves), AdaptiveResidencyController, Belady-style global and per-layer.
- **Exact curves.** Event-atomic LRU stack distances (Fenwick tree) give per-layer LRU misses for every slot count in one pass; a test asserts equality with actual replay.
- **Allocation.** Uniform, lookahead greedy (utility-based partitioning), exact DP (multiple-choice knapsack on a byte grid).

## 5. Where a runtime controller would go (not implemented)

Above the store, never inside it: the store keeps memory ownership, the hard budget, loading, async reads, slots and pinning. The controller would set per-layer budgets (the store would need per-layer partitions instead of hash shards — a change to `Shard` selection), an optional pinned set, and the timing of reallocation (epochs of 32/64/128 tokens with hysteresis). This waits for `ARGINE2-MEMORY-CHARACTERIZATION.md` (requirement section 59).
