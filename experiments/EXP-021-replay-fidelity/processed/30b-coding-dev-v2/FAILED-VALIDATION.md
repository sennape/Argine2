# First Windows format-v2 validation — FAILED closure (kept as evidence)

[MEASURED, 2026-10-04] Produced by `argine.bat research fidelity` on `raw/30b-coding-dev-v2/run-00-4G`, with the replay as of 2026-10-03. The files in this directory are left exactly as that command wrote them.

Result:
- format 2, 16 shards, recorded shard placement;
- reconstruct inconsistencies 0, replay failures 0, budget violations 0;
- runtime 0.70581427 vs replay 0.70676255 (+0.095 pp);
- misses 149,191 vs 148,742; evictions 144,689 vs 144,240;
- **event disagreement 5.29%**;
- first divergence at request 60,351 (graph 10, node 1494, prefill, shard 2, key (18, 38, down)): runtime victim (0, 36, gate), replay victim (47, 85, down);
- `shards_with_seq_gaps = 3`.

The explanation is in [`docs/research/REPLAY-FIDELITY-VALIDATION.md` §13](../../../../docs/research/REPLAY-FIDELITY-VALIDATION.md#13-windows-v2-follow-up-2026-10-04):

1. **[FALSIFIED] The replay's pin model.** It released every thread's pins at every node. The runtime's pins are per thread, and this run used `--threads 4 --threads-batch 12`: prompt graphs run on threads {0..11}, decode graphs on {4, 12, 13, 14}. Thread 12 still pinned (47, 85, down) from the last warm-up decode graph, and the runtime passed over it (`E … skipped = 1`).
2. **The 3 sequence gaps are replay-irrelevant.** They are 5 up-node re-pin hits of thread 13 in graph 395, layer 23, lost from the stdio buffer when llama-server was terminated during the flush sentinel. No workload record follows them.

Re-analysis of the same raw trace with the corrected (per-thread) pin model: `../30b-coding-dev-v2-per-thread-pins/`, which gives 0 disagreements in 1,474,909 requests. `../30b-coding-dev-v2-single-pool-reproduction/` reruns the old model and reproduces this directory's 0.052928011 exactly.
