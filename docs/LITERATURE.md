# Literature positioning

Argine2 claims no novelty without comparison. For each related work: the problem it addresses, what Argine2 shares, and what Argine2 investigates differently. Entries marked **[OPEN] not read** are known to exist and look relevant but have not been read; nothing in this repository should be read as "new" relative to them until someone does. Details of works summarised from memory are marked **[verify]**. Corrections are welcome contributions.

## Classic caching and partitioning

| work | problem | shared | different in Argine2 |
|---|---|---|---|
| Belady, "A study of replacement algorithms for a virtual-storage computer", IBM Systems Journal 1966 (MIN) | optimal offline replacement | the offline oracle (farthest next use) | multi-key events and mixed sizes: used as a reference, not a proven bound |
| Mattson, Gecsei, Slutz, Traiger, "Evaluation techniques for storage hierarchies", IBM Systems Journal 1970 | stack algorithms, all cache sizes in one pass | exact LRU curves from stack distances | event-atomic variant (members of one event never count against each other) |
| Denning, "The working set model for program behavior", CACM 1968 | working set over a window τ | `ws_tau` metric | applied per MoE layer, alongside whole-trace EWS |
| Johnson & Shasha, "2Q", VLDB 1994 | scan-resistant replacement | 2Q baseline | byte budgets, event atomicity |
| Megiddo & Modha, "ARC", FAST 2003 | adaptive recency/frequency balance | — | not yet implemented (good contribution) |
| Qureshi & Patt, "Utility-Based Cache Partitioning", MICRO 2006 | split a shared cache among cores by marginal utility (UMON shadow tags, lookahead) | the controller's utility monitors and lookahead allocation, with layers in place of cores | the "cores" are layers visited cyclically within each token; hard byte budget; NVMe miss cost with queue-depth effects |

## MoE expert offloading and caching

| work | problem | shared | different in Argine2 |
|---|---|---|---|
| Eliseev & Mazur, "Fast Inference of Mixture-of-Experts Language Models with Offloading", arXiv 2312.17238 (2023) | MoE on consumer GPUs with host-memory offloading; LRU expert cache + speculative loading [verify] | LRU expert cache baseline | CPU + NVMe, hard DRAM budget, layer allocation as the object of study |
| MoE-Infinity, Xue et al., arXiv 2401.14361 (2024) | activation-aware expert prefetching/caching for offloaded serving [verify] | request-level routing traces | single-user CPU inference; focus on allocation across layers |
| Pre-gated MoE, Hwang et al., ISCA 2024 | change the model to predict next-layer experts | — | Argine2 does not modify or retrain models |
| EdgeMoE, Yi et al., arXiv 2308.14352 (2023) | on-device MoE with expert storage and preloading [verify] | edge setting | measurement-first; global budget split |
| Fiddler, Kamahori et al., arXiv 2402.07033 (2024) | compute cold experts on CPU instead of transferring them [verify] | CPU role in MoE | everything is on CPU here; the question is DRAM residency vs NVMe |
| AdapMoE, Zhong et al., arXiv 2408.10284 (ICCAD 2024) | adaptive gating, prefetching, and **dynamic-programming cache allocation across layers** [verify] | layer-aware allocation by DP — the closest precedent for `layer-aware-static` | online re-allocation by measured marginal value under a hard budget; falsification on real runtime traces; NVMe-backed CPU |
| HOBBIT (2024) | mixed-precision expert offloading [verify] | — | precision is fixed per experiment here (quality curve in EXP-013) |
| ProMoE (2024) | expert prefetching for consumer hardware [verify] | — | prefetching not studied yet |
| FlashMoE, Kim et al., arXiv 2601.17063 (2026) | SSD-offloaded MoE on edge devices; ML-based replacement approximating Belady; **fixed per-layer expert slots** | SSD offloading, per-layer partitions, Belady as target | interpretable (no learned model) controller; per-layer slot counts are the variable, not a constant |
| SpecMD, Hoang et al., arXiv 2602.03921 (2026) | speculative expert prefetching study; **Least-Stale** eviction; expert access "not consistent with temporal locality assumptions" | Least-Stale baseline; the temporal-locality question | exact definition of Least-Stale not yet checked ([RQ-017](../OPEN_QUESTIONS.md#rq-017--what-exactly-is-least-stale-and-does-it-help-here)) |
| "Reproducible Evaluation of MoE Expert Caching", Y. Zhang, arXiv 2608.07911 (2026) | measurement fragility: **event-atomic replay**, workload contamination, per-layer quotas, union-to-capacity ratio, Belady gap 44-46% | event-atomic replay contract; per-layer quotas; contamination guard; oracle-gap metric | runtime-recorded traces on a consumer machine; adaptive allocation; end-to-end validation of replay |
| MoE-CORE, arXiv 2610.01950 (2026) — "Coordinated Expert Offloading and Residency for Memory-Constrained MoE Inference" | **[OPEN] not read** (fetch rate-limited) | possibly the central question | unknown until read |
| Mira, arXiv 2609.38090 (2026) — "Memory-Efficient MoE Inference Using Adaptive Caching and Predictive Expert Staging" | **[OPEN] not read** | possibly adaptive cache sizing | unknown until read |

## Related systems

- **llama.cpp** — the inference engine; `--n-cpu-moe` / tensor overrides give a static split of expert tensors between devices; mmap residency is decided by the OS.
- **KTransformers**, **PowerInfer** (hot/cold neuron locality on consumer GPUs), **"LLM in a Flash"** (Alizadeh et al., 2023: flash-resident weights, windowing, row-column bundling) — memory-tiered inference outside MoE expert caching; relevant to NVMe-backed residency [verify].

## What Argine2 would add if its hypotheses survive

A measurement of whether layer-level marginal value of memory differs enough on real routing traces to matter under a *global hard* budget on CPU+NVMe; a validated replay against the runtime that produced the traces; and a documented failure envelope. If they do not survive, a clean negative result with the apparatus to check it.
