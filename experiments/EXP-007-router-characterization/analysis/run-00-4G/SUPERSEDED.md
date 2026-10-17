# SUPERSEDED — kept unchanged as research history

[SUPERSEDED, 2026-10-03, EXP-021] This analysis was produced with replay semantics **v1-legacy** and the grid-rounded allocator. Every file in this directory is left exactly as it was generated. The regenerated analysis is in [`../../analysis-v2/run-00-4G/`](../../analysis-v2/run-00-4G/). The investigation is in [`docs/research/REPLAY-FIDELITY-VALIDATION.md`](../../../../docs/research/REPLAY-FIDELITY-VALIDATION.md).

The analysis has three problems.

1. **The "sharded-lru-16" row is not a runtime reproduction.** It replays expert units (one unit is the 3 slices), decode steps only. The runtime figure it was compared against (70.60% at 4G) counts gate-node slice hits over all workload steps, prefill included.
   - The row's 75.32% is a different metric, not a 4.7-point fidelity error.
   - The row has been renamed `sharded-lru-16-expert` and moved to the *counterfactual* category.
   - The runtime reproduction is now `store-replay`, which reproduces 70.72% (runtime 70.62%).
2. **Prefill events were never served to the cache state.** In v1-legacy, steps outside the counted phase did not touch the cache, so decode began from a state the runtime never had.
   - v2 serves every recorded step and counts only the phase asked for.
   - Warm-up and flush-sentinel steps are still served but never counted.
3. **`layer-aware-static` / `oracle-static-lru` (38,206 misses) is not optimal.** The DP allocated on a 512 KiB grid, which charged each 2,654,208-byte expert as 3 MiB (+18.5%). This made the uniform split (36,473 misses) infeasible inside the DP.
   - The allocator is now exact and checked against every feasible baseline (`dp_optimal_checked`).
   - The anomaly is explained in section 7 of the validation document.

Still valid from this directory: the routing characterization, which does not depend on the cache. The regenerated `characterization.json` has identical values. `expert_frequency.csv`, `coactivation.csv`, `reuse_cdf.csv`, `drift.csv` and `per_layer.csv` are byte-identical. Every policy, curve, heterogeneity, adaptive and REPORT number is superseded by `analysis-v2`.
