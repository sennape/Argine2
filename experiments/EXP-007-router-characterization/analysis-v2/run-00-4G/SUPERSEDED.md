# SUPERSEDED in part — kept unchanged

[SUPERSEDED, 2026-10-04, EXP-021 Windows v2 follow-up] Only the `store-replay` rows of this analysis are superseded. They used the single-pool pin model: every thread drops its pins at every node. The runtime's pins are per thread, and the reference runs use `--threads 4 --threads-batch 12`.

The regenerated analysis is [`../../analysis-v3/run-00-4G/`](../../analysis-v3/run-00-4G/). There, `store-replay` models each thread pool's pins separately. Expert-event hit, decode:

| budget | measured runtime | store-replay here (single pool) | store-replay in analysis-v3 (per pool) |
|---|---|---|---|
| 4 GiB | 73.75% | 73.91% | 73.79% |
| 2 GiB | 51.69% | 51.89% | 51.66% |
| 1.5 GiB | 43.10% | 43.17% | 43.09% |

Every other row and file is unchanged in analysis-v3; `overhead.csv` and `adaptive_overhead.csv` differ only in wall-clock timings.
