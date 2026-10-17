# Historical audit of EXP-011 … EXP-018 (completed and closed 2026-10-11)

EXP-011 to EXP-017 come from Argine v1. EXP-018 is an Argine2 analysis of Argine v1 raw logs. Argine v1 evidence was imported into Argine2; where preserved raw provenance permits, historical measurements were re-derived with `tools/build_historical_data.py`. EXP-015 is an explicit exception: its PRETOUCH attribution and per-miss counters survive in the historical Argine v1 README rather than in independently reconstructable raw records.
- No experiment is rerun, and no raw record or measurement is changed.
- Claims that were reasonable at the time but too broad are **narrowed** or marked **[SUPERSEDED]** in place. They are not rewritten.
- The historical Argine v1 text (`docs/history/`) is source material. It is not edited; any overstatement there is corrected by this audit layer.
- A closed historical experiment keeps the registry status its own data can support. Open follow-up questions go to new experiments.

## Audit table

| EXP | registry status | audit verdict | evidence quality | main limitation | rerun required for closure? | future follow-up |
|---|---|---|---|---|---|---|
| [EXP-011](../../experiments/EXP-011-binary-rebuild/README.md) different binaries, matched plan | SUPPORTED | **closed, SUPPORTED with narrowed causal scope**: under the matched recorded model/runtime plan, the old and new patched binaries differ by 1.74× in median generation throughput (7.45 vs 12.97 tok/s; official b11214 13.53). Binary identity is an experimental variable and must be fingerprinted. | moderate as an observation (one invocation per binary, 3 within-invocation repetitions, CV 2.1–4.5%); none as a causal attribution | llama.cpp version, OpenMP and linkage are confounded; the old binary is not fingerprinted; sequential, not interleaved; predates the Argine2 reproducibility protocol | no | none required. Isolating the build factor would need a new experiment with fingerprinted binaries differing in one factor and rotated launches. |
| [EXP-012](../../experiments/EXP-012-dense-14b-control/README.md) dense 14B control | FALSIFIED | **closed, FALSIFIED with narrowed proxy scope**: on the matched-text perplexity proxy (`ctx=512;chunk=60;testo=da315163291ee5c9`), the larger compressed 30B IQ2_S did not show the predicted advantage (13.6283 ± 0.3290 vs 13.2766 ± 0.3176 for the dense 14B Q4_K_M) | moderate for the matched-text PPL observation; weak for broad assistant-quality/knowledge claims; historical/confounded for causal throughput attribution | one PPL corpus/signature; no equivalence/non-inferiority test; similar but unequal model footprints (≈8.38 vs 9.65 GiB); different runtime binaries for throughput (`llama-noomp@757a17a4` vs `llama-noomp@ecd06508`); different sessions; non-interleaved, non-rotated benchmark design | no | Any modern assistant-capability comparison or same-binary rotated throughput comparison would be a new experiment, not required for historical closure. |
| [EXP-013](../../experiments/EXP-013-quality-per-bit/README.md) perplexity vs expert bpw | SUPPORTED | **closed, SUPPORTED with narrowed descriptive scope**: IQ2_S at 2.56 expert bpw is a descriptive elbow among the four tested 30B points under the recorded PPL signature (`ctx=512;chunk=60;testo=da315163291ee5c9`) | moderate for the four-point descriptive curve; weak for universal knee/optimality claims | four discrete points; one corpus/signature; perplexity-only quality proxy; no formal knee criterion; expert bpw not total-model bpw; duplicate IQ2_XXS calibration identity not encoded in PPL records; over-inclusive derived artifact | no | A formal knee comparison across models/corpora or BF16-origin quantizations would be a new experiment, not required for historical closure. |
| [EXP-014](../../experiments/EXP-014-conversation-prefix-reuse/README.md) TTFT in a growing conversation | SUPPORTED | **closed, SUPPORTED with narrowed operational scope**: in the single historical six-turn Qwen3-30B benchmark, in-RAM prefix reuse kept observed TTFT within 4.78–5.62 s while cumulative prompt footprint grew from 188 to 1467 tokens; without reuse TTFT rose to 40.12 s. Turn-6 observed contrast ≈7.8x. | strong descriptive contrast within the archived invocation; weak for population/generalized effect size | one conversation; one invocation; one model; no replication; no rotated/randomized order; order not encoded in archived record; historical memory pressure; no disk-KV implementation tested | no | Persistent/restart/cross-conversation KV caching or a replicated rotated prefix-reuse benchmark would be a new experiment, not required for historical closure. |
| [EXP-015](../../experiments/EXP-015-pretouch/README.md) PRETOUCH | SUPPORTED | **closed, SUPPORTED with provenance-limited mechanism scope**: the preserved Argine v1 counters report approximately 0.30 ms lower read-per-miss timing with PRETOUCH at both tested budgets, compared with approximately 0.37–0.40 ms explicit new-page cost; this is quantitatively consistent with substantial read-path work displacement. PRETOUCH was slower at both reported throughput endpoints, and the 2.67x-span exponents 0.54 ± 0.03 vs 0.58 ± 0.06 do not reproduce the earlier large short-span slope difference. | moderate as internally consistent historical-source evidence; weak for independent reproducibility/causal attribution | PRETOUCH flag absent from raw logs; per-miss counters not archived; no experiment-local raw file; results table sourced from historical README; no raw reconstruction by builder; one model/machine; no formal slope equivalence; historical configuration/order incomplete | no | A modern fingerprinted, rotated PRETOUCH on/off mechanism experiment could directly test the causal decomposition, but it is a new experiment and not required for historical closure. |
| [EXP-016](../../experiments/EXP-016-coactivation-reorder/README.md) co-activation reorder | FALSIFIED | **closed, FALSIFIED with historical-attribution and tested-layout scope**: the specific co-activation-based GGUF reorder from the historical `argine reorder` workflow did not show the predicted throughput improvement; the contemporaneous Argine v1 README reports 4.69 (reordered) vs 5.17 tok/s (original) | moderate for preservation of reorder event and selected benchmark values; weaker for independent original/reordered condition attribution | raw benchmark metadata do not independently label the reordered condition; historical README supplies attribution; one model, trace and layout objective; no rotated/interleaved condition design | no | A modern fingerprinted physical-layout experiment across multiple traces/access paths would be a new experiment, not required for historical closure. |
| [EXP-017](../../experiments/EXP-017-calibration-text/README.md) calibration text | SUPERSEDED | **closed, SUPERSEDED with measurement-design scope**: the +2.06% synthetic-signature IQ2_XXS measurement remains valid as recorded; its general negligible-damage verdict is superseded by the real-prose measurements (+36.3% old IQ2_XXS, +26.0% after real-prose recalibration) | strong for the recorded within-signature PPL observations and real-prose recalibration lineage; moderate for causal attribution of the first misleading verdict to test/calibration corpus choices | synthetic and real evaluations differ in both corpus and chunk count; no full calibration × evaluation factorial; one model/quantization; PPL-only quality proxy; over-inclusive builder extraction | no | A controlled factorial calibration-corpus × evaluation-corpus experiment with fixed signature and fingerprinted quantization binaries would be a new experiment, not required for historical closure. |
| [EXP-018](../../experiments/EXP-018-reproducibility-audit/README.md) run-to-run dispersion | FALSIFIED | **closed, FALSIFIED with recorded-configuration / legacy-provenance scope**: the historical ≤5% reproducibility assumption is operationally falsified. The maximum within-invocation CV is ≈6.43% (0.06433166). 30B records matching the historical grouping key show ≈15.36% / ≈16.04% min-max spreads. A supplementary 80B raw pair measured 7.10 vs 5.74 tok/s. No universal 15% noise floor is established. | strong for archived within-invocation repetition variability and observed numeric cross-run contrasts; moderate-to-weak for estimating intrinsic fully controlled session variance | incomplete historical runtime fingerprints; grouping key omits read mode and other plan distinctions; no explicit session IDs; 80B pair not a builder-matched results row; small groups; environmental state uncontrolled | no | Modern Argine2's interleaved, fingerprinted replication protocol supersedes the historical measurement design. A dedicated variance-components study would be a new experiment, not required for historical closure. |

## Narrowed or superseded claims

- **EXP-011.**
  - The title "Same source, different build: the 1.74x factor" is replaced by "Matched model and runtime plan, different binaries: the observed 1.74x factor". The patched binaries differ in llama.cpp version, OpenMP and linkage, so "same source" was not established.
  - The 1.74× is **not** attributed to OpenMP, static linking, a compiler option or recompilation alone. The Argine v1 text already said that OpenMP vs llama.cpp version was not isolated.
  - 1.74× is an observed historical contrast on this machine and configuration. EXP-018 later measured meaningful between-session variation, so 1.74× is not a universal build-effect coefficient.
  - Updated with the same scope: the global README (F-007, U-003), FH-009 and the research-history entry of 2026-10-01.
- **EXP-012.**
  - "Quality is indistinguishable" is narrowed to "no PPL advantage detected for the larger compressed model on the tested proxy". A failure to detect the predicted advantage is not an equivalence test.
  - "At equal quality" is unsupported and removed from current conclusions.
  - The approximately 3.39x speed ratio (12.97 vs 3.83 tok/s) is an observed historical contrast, not a causal architecture effect. The binaries and sessions differ, and EXP-011 shows binary identity is an experimental variable.
  - "Equal RAM" is narrowed to similar footprint within the same 16 GB deployment envelope: ≈8.38 GiB vs 9.65 GiB, the latter the Argine v1 README value; the 30B benchmark record gives 9.58 GiB.
  - Broad assistant-quality and knowledge equivalence was not established.
  - Narrow supported observation: the 14B repeat (3.81 → 3.83 tok/s) shows its low throughput was not materially changed when observed paging pressure fell sharply.
  - Updated with the same scope: the EXP-012 README (research question, setup, interpretation, limitations), the global README (F-008 and its table-of-contents entry, the falsified-hypotheses table), FH-004, the research-history entry of 2026-10-02 and the falsified-hypotheses list in `ARGINE2-MEMORY-CHARACTERIZATION.md`. The EXP-012 title and config.json are unchanged: the title is the historical research question.
- **EXP-013.**
  - "2.56 bpw is the knee" is narrowed to a descriptive empirical elbow at 2.56 expert bpw among the tested 30B grid.
  - No universal optimum or formal changepoint is established.
  - "About 4x steeper" is refined to adjacent descriptive slopes ≈0.67 / 2.92 / 3.35 PPL per expert bit, with the lower intervals ≈4–5x the upper interval. This is a finite-difference description, not a significance test.
  - The selected 15.1946 IQ2_XXS point is historically identified as re-calibrated; raw PPL metadata does not encode the calibration identity.
  - The extra 16.4396 IQ2_XXS record and the 14B row remain preserved but are not part of the audited four-point curve. They are in the pool because the historical builder selected every 60-chunk PPL record; the builder is unchanged.
  - IQ3_XXS is not a measured quality point in EXP-013.
  - Updated with the same scope: the EXP-013 README (research question, hypothesis reading, setup, raw-data note, result table, interpretation, limitations) and the research-history entry of 2026-10-01. The title and config.json are unchanged.
- **EXP-014.**
  - "TTFT stays flat" is narrowed to the observed 4.78–5.62 s across six turns in this benchmark.
  - 7.8x is the turn-6 within-benchmark contrast, not a universal effect size.
  - The two conditions have matched cumulative prompt footprints (188 / 444 / 697 / 950 / 1207 / 1467); reuse splits them into cached + processed tokens (turn 6: 1293 + 174).
  - "Disk KV adds nothing" is too broad and removed from current scientific conclusions. The disk KV cache itself was not benchmarked, and the ">100x" figure was an engineering estimate against no cache.
  - The historical cancellation remains an engineering prioritization decision for the ordinary same-session conversation case.
  - Restart, persistence and cross-conversation use cases remain untested.
  - The exact condition order is not established by the archived raw record or the historical source.
  - Updated with the same scope: the EXP-014 README, FH-016 (premise kept, scope note added) and the research-history entry of 2026-10-02. The title and config.json are unchanged.
- **EXP-015.**
  - "Page zeroing moved, not added" is narrowed: the historical counters are quantitatively consistent with substantial displacement plus explicit-touch overhead.
  - Approximately 73–81% of the explicit new-page cost corresponds numerically to the reported reduction in read-per-miss timing (0.301 / 0.370 ms at 4 GiB, 0.290 / 0.395 ms at 1.5 GiB).
  - The explicit PRETOUCH cost is approximately 23–36% larger than the saved read-path time (the historical text said 23–37%).
  - PRETOUCH was slower at both reported endpoints (≈−3.9% and ≈−8.0%); this is not a universal penalty.
  - "Does not change slope" is narrowed to a failure to reproduce the earlier large short-span slope effect (1.33x span: 0.70 ± 0.20 vs 0.47 ± 0.15, an inadequate span). No equivalence or exactly-zero slope effect is established.
  - "~4% intercept" is kept only as a historical fitted interpretation.
  - "Mechanism confirmed" is narrowed to provenance-limited consistency evidence. The PRETOUCH attribution and per-miss counter values cannot be independently reconstructed from raw logs: there is no experiment-local raw file, `results.csv` is sourced from the historical README, and the builder has no EXP-015 section.
  - The title is kept as historical shorthand, not a current absolute causal claim. The Origin field (README and config.json) was corrected from "data re-derived from raw logs by Argine2" to the historical-README provenance.
  - Updated with the same scope: the EXP-015 README, EXP-015 config.json (origin only), FH-014 and the research-history entry of 2026-10-02 → 03.
- **EXP-016.**
  - 4.69 vs 5.17 tok/s is the historical source's attribution. The raw benchmark groups have medians 5.17 / 4.57 / 4.43, and the raw records alone do not assign original/reordered conditions: all carry the base model path.
  - 4.69 is one repetition in the 22:45 group, whose median is 4.57.
  - `runs_prima` / `runs_dopo` are preserved under their exact field names, with no "mean run length" relabelling. `guadagno=1.716` is the tool's own recorded field.
  - The failure is scoped to the tested reorder. There is no universal statement that layout or co-activation is useless.
  - The page-fault evidence (EXP-003) is contextual, not causal.
  - Updated with the same scope: the EXP-016 README, FH-008, the global README falsified-hypotheses table and the research-history entry of 2026-09-27. The title and config.json are unchanged.
- **EXP-017.**
  - +2.06% (3.4258 vs 3.4963) remains a valid synthetic-signature observation. The negligible-general-damage verdict is what is superseded.
  - Old IQ2_XXS on real prose: +36.3% (12.0642 vs 16.4396). Recalibrated IQ2_XXS on real prose: +26.0% (15.1946). Recalibration lowers the same-real-test PPL from 16.4396 to 15.1946.
  - The synthetic and real signatures are chunk18 vs chunk60, so the text effect is not perfectly isolated, and there is no full factorial decomposition.
  - The old calibration and test texts are distinct files (`testo-imatrix.txt`, `testo-ppl.txt`) from the same historical templated distribution. The real calibration and reference texts are distinct files, historically described as sentence-disjoint.
  - The global raw log provides the imatrix/quantize lineage, including the recalibrated model's imatrix (22:18:32), quantize (22:52:31) and perplexity (23:06:55) events.
  - The three later PPL records in the EXP-017 raw/results (IQ2_XS, IQ2_S, Qwen3-14B) are builder over-inclusion, not core evidence.
  - Status semantics: EXP-017 SUPERSEDED (the verdict was replaced) and FH-019 FALSIFIED (the evaluation design could not support the broad verdict) are compatible.
  - Updated with the same scope: the EXP-017 README, FH-019 and the research-history entry of 2026-09-30. The title and config.json are unchanged.
- **EXP-018.**
  - The historical ≤5% reproducibility assumption is falsified operationally.
  - Within-run CV reaches ≈6.43% (0.06433166, one 8B invocation); this is the provenance-cleanest evidence.
  - 30B legacy nominal groups show ≈15.36% (`78ee4096`, 4 GiB, five records over about ten hours) and ≈16.04% (`757a17a4`, 6 GiB) range spreads, defined as `(max - min) / min`, a descriptive range.
  - Hidden read mode and PRETOUCH state prevent a pure-noise interpretation.
  - 80B 7.10 / 5.74 is a supplementary raw comparison, not a results.csv group (the builder key splits it on `prompt_chars`). Its read-mode provenance is asymmetric (`asincrono` recorded at 13:42:06; absent at 13:50:24).
  - Denominators are kept distinct: −19.2% relative to 7.10 vs 23.7% under `(max - min) / min`.
  - The builder's mixed-read-mode `3cac5531` rows (≈10.8 / 6.1 / 5.1 / 2.2%) are excluded from reproducibility inference.
  - The 8B ≈21% group is excluded because of known plan differences (vulkan vs cpu, repack); its within-run CVs stay valid.
  - The 14B ≈0.5% group illustrates heterogeneity.
  - The `sessions` column is the grouped-record count (`len(rr)`), not verified sessions.
  - 15% is not a universal significance or noise threshold; the ~15% rule is a conservative legacy guardrail. Interleaving is the modern protocol, not a design EXP-018 proves uniquely necessary. Thermal, load and cache state are possible contributors, not measured causes.
  - Updated with the same scope: the EXP-018 README, FH-020, U-002, the root methodology bullet and the hypothesis table, docs/METHODOLOGY.md and the research-history entry of 2026-10-03. The title and config.json are unchanged.

## Registry state

There were no status changes:
- EXP-011 remains SUPPORTED;
- EXP-012 remains FALSIFIED;
- EXP-013 remains SUPPORTED;
- EXP-014 remains SUPPORTED;
- EXP-015 remains SUPPORTED;
- EXP-016 remains FALSIFIED;
- EXP-017 remains SUPERSEDED;
- EXP-018 remains FALSIFIED.

No new run was required.

Historical audit block EXP-011 ... EXP-018 closed on 2026-10-11. No experiment was rerun and no immutable measurement evidence was changed.
