# EXP-017 — A too-easy test text produced the wrong quality verdict

| field | value |
|---|---|
| ID | `EXP-017-calibration-text` |
| Status | **SUPERSEDED** |
| Period | 2026-09-30 |
| Origin | Argine v1 (historical); data re-derived from raw logs by Argine2 |
| Related questions | — |
| Supersedes / superseded | its first measurement (+2.06%) is superseded by the real-prose measurement in the same experiment |

> **Title scope (2026-10-11 audit).** The title preserves the historical shorthand. The recorded synthetic PPL measurement itself is not invalid; the general quality verdict inferred from it is what was superseded. The "Supersedes / superseded" field above (kept as recorded, also in config.json) should be read the same way: what is superseded is the first measurement's general negligible-damage verdict, not its recorded numbers.

## Research question

Did the initial +2.06% IQ2_XXS perplexity penalty measured on the historical synthetic templated evaluation generalize to a disjoint real-prose evaluation, and how did real-prose recalibration change the recorded IQ2_XXS penalty?

*(Original wording, kept as history: "Is the perplexity cost of 2-bit experts negligible?")*

## Hypothesis

[HYPOTHESIS] (first measurement) IQ2_XXS costs +2% perplexity, within noise.

*Audit reading (2026-10-11):*
- +2.06% was a valid descriptive result for the synthetic signature `ctx=512;chunk=18;testo=bf8011647e36fad6`.
- The general negligible-damage verdict inferred from it is what was superseded.
- "Within noise" is historical wording, not an equivalence or significance test.

## Setup

First: synthetic templated sentences for both calibration and test. Then: disjoint real-prose texts (testo-calibrazione / testo-riferimento).

*Audited setup (2026-10-11):*

**Stage 1: historical synthetic setup.**
- Models: the Qwen3-30B-A3B Q4_K_M baseline and the old IQ2_XXS model (`Qwen_Qwen3-30B-A3B-Q4_K_M-experts-IQ2_XXS.gguf`).
- PPL test: `testo-ppl.txt`, ctx=512, chunk=18, hash `bf8011647e36fad6`.
- The old imatrix was generated from `testo-imatrix.txt` (global run log, 2026-09-30 16:54:06, chunk 128), and the old model was quantized with it (17:41:17).
- The old calibration and test texts are **different files**. The historical source describes them as the same synthetic templated distribution: sentences built from 15 subjects, 13 verbs, 11 objects and 8 tails on one schema.

**Stage 2: real-prose evaluation and recalibration.**
- Test: `testo-riferimento.txt`, ctx=512, chunk=60, hash `da315163291ee5c9`. The old IQ2_XXS was also evaluated on this test.
- A new imatrix was generated from `testo-calibrazione.txt` (22:18:32, chunk 128), and `qwen30b-IQ2-reale.gguf` was quantized as IQ2_XXS with it (22:52:31). Both quantizations were run with the same `llama-patched\llama-quantize.exe` path, which is not fingerprinted.
- The historical source says the calibration and reference corpora have no sentences in common.

## Commands

```bat
argine ppl MODEL --text testo-riferimento.txt --ctx 512 --chunks 60
```

## Raw data

`raw/records.jsonl` (all perplexity records).

*Provenance (2026-10-11 audit):*
- The experiment-local raw file contains PPL records only (`data/raw/argine-v1/perplexity.jsonl`; these records carry no timestamps). `results.csv` is builder-derived from **all** eight PPL records.
- The global run log `data/raw/argine-v1/runs.jsonl` provides the imatrix/quantize lineage:
  - **old IQ2_XXS:** `imatrix` 2026-09-30 16:54:06 (`testo-imatrix.txt` → `Qwen_Qwen3-30B-A3B-Q4_K_M.imatrix`), then `quantize` 17:41:17 (→ `…-experts-IQ2_XXS.gguf`, expert_type IQ2_XXS, using that imatrix);
  - **recalibrated IQ2_XXS:** `imatrix` 22:18:32 (`testo-calibrazione.txt` → `qwen30b-reale.imatrix`), then `quantize` 22:52:31 (→ `qwen30b-IQ2-reale.gguf`, IQ2_XXS, using `qwen30b-reale.imatrix`), then a `perplexity` event at 23:06:55 (15.1946 ± 0.3709, signature `ctx=512;chunk=60;testo=da315163291ee5c9`).
- The local raw PPL record alone does not encode which imatrix generated each model. The lineage of `qwen30b-IQ2-reale.gguf` is independently supported by the global imatrix and quantize events.
- The historical README supplies the semantic description of the synthetic template and the sentence-level disjointness of the real-prose corpora. The raw log identifies the different files but does not itself prove sentence-level disjointness.
- The historical extraction is over-inclusive; see Result.

## Result

[MEASURED] templated text: 3.4258 vs 3.4963 (+2.06%); real prose: 12.0642 vs 16.4396 (+36.3%), and +26.0% (15.1946) after re-calibrating the importance matrix on real prose. *(Original wording; audited table below.)*

**Core comparison (2026-10-11 audit).** The PPL values are [MEASURED, raw records]; the percentages are descriptive, within each recorded signature.

| condition | baseline Q4_K_M | IQ2_XXS | relative PPL delta | signature |
|---|---:|---:|---:|---|
| synthetic (historical) | 3.4258 ± 0.119 | 3.4963 ± 0.1234 (old) | ≈ +2.06% | ctx512 / chunk18 / `bf8011647e36fad6` |
| real prose, old IQ2_XXS | 12.0642 ± 0.2912 | 16.4396 ± 0.4058 | ≈ +36.3% | ctx512 / chunk60 / `da315163291ee5c9` |
| real prose, recalibrated IQ2_XXS | 12.0642 ± 0.2912 | 15.1946 ± 0.3709 | ≈ +26.0% (exact 25.95%) | same real-prose signature |

- **Same-real-test recalibration contrast:** old vs recalibrated IQ2_XXS goes 16.4396 → 15.1946, ≈7.6% lower PPL. The degradation relative to Q4 drops by ≈10.3 percentage points (+36.3% → +26.0%). This is a descriptive historical comparison.
- **Not part of the core comparison:** three PPL records preserved in `results.csv` / `raw/records.jsonl` because the builder writes every PPL record. They are IQ2_XS 14.3578 ± 0.3472, IQ2_S 13.6283 ± 0.3290 and Qwen3-14B Q4_K_M 13.2766 ± 0.3176, all on the real-prose signature.

## Interpretation

*[Historical interpretation, narrowed by the 2026-10-11 audit: "The measuring instrument, not the model, produced the first verdict. Calibrating on the test distribution hid the damage." This is superseded by the wording below.]*

[SUPERSEDED, HISTORICAL / MEASUREMENT-DESIGN-SCOPED] **Audit closure (2026-10-11).**
- **First result still valid.** The initial measurement was not numerically invalid. It answered a much narrower question about one synthetic templated evaluation distribution. The superseded element is the general quality verdict inferred from that measurement: that the perplexity cost of the IQ2_XXS experts was negligible in general.
- **Did not generalize.** The initial synthetic-text measurement produced an optimistic +2.06% quantisation-cost estimate that did not generalize to the recorded real-prose evaluation. On real prose, the same historical IQ2_XXS model showed ≈+36.3% PPL versus Q4.
- **Recalibration.** Re-calibrating an IQ2_XXS quantization on a separate real-prose corpus reduced the recorded real-prose degradation to ≈+26.0%.
- **Both corpora matter here.** These observations implicate both evaluation-corpus choice and calibration-corpus choice. But EXP-017 is not a factorial experiment and does not causally decompose their contributions:
  - the synthetic vs real comparison also changes the chunk count (18 vs 60) and the text hash, so the effect of text easiness is not isolated as a single factor;
  - the recalibrated model was never evaluated on the synthetic signature, so there is no complete 2×2 design.
- **Scope.** No universal claim is made about all templated corpora. +36.3% and +26.0% are not universal IQ2_XXS degradation coefficients.
- **Status semantics.** EXP-017 stays SUPERSEDED because the +2.06%-based general verdict was replaced by the real-prose measurements. FH-019 stays FALSIFIED because the operational hypothesis that this synthetic templated evaluation could support the broad quantisation-damage verdict failed. These are compatible.
- No new measurement. No rerun is required for historical closure ([historical audit](../../docs/research/HISTORICAL-AUDIT-EXP011-018.md)).

## Limitations

[LIMITATION] Texts are not wikitext: numbers are not comparable with published ones.

[LIMITATION, historical audit 2026-10-11]
1. One model (Qwen3-30B-A3B).
2. One expert quantization type (IQ2_XXS) for the core comparison.
3. Perplexity only, not assistant capability.
4. The synthetic and real-prose comparisons use different text hashes (`bf8011647e36fad6` vs `da315163291ee5c9`).
5. The synthetic evaluation uses chunk=18, the real-prose one chunk=60.
6. Therefore the evaluation-corpus effect is not isolated as a single factor.
7. There is no full calibration-corpus × evaluation-corpus factorial.
8. There is no evaluation of the recalibrated IQ2_XXS on the synthetic signature.
9. The old synthetic imatrix text and test text are different files (`testo-imatrix.txt`, `testo-ppl.txt`); "same distribution" comes from the historical source.
10. The sentence-level disjointness of the real corpora comes from the historical source, not from a hash-level raw proof.
11. The local EXP-017 raw file contains PPL records only; the imatrix/quantize lineage comes from the global raw log.
12. The historical binaries are not fingerprinted to the current Argine2 standard.
13. "Within noise" is not an equivalence test.
14. The corpora are not Wikitext; published PPL values are not directly comparable.
15. The builder extraction is over-inclusive: eight PPL records, of which five are core.
16. The percentages are descriptive within the recorded signatures, not universal quantisation coefficients.

## Status history

2026-09-30 SUPERSEDED (the +2% verdict) by the real-prose measurements, kept visible here. 2026-10-11 historical audit: SUPERSEDED retained with measurement-design scope. The +2.06% synthetic-signature measurement remains valid as recorded, but its broad negligible-damage interpretation was superseded by real-prose measurements (+36.3% for the old IQ2_XXS and +26.0% after real-prose recalibration). No new measurement.
