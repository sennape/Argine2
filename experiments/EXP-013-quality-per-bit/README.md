# EXP-013 — Perplexity vs expert bits per weight

| field | value |
|---|---|
| ID | `EXP-013-quality-per-bit` |
| Status | **SUPPORTED** |
| Period | 2026-09-30 → 2026-10-01 |
| Origin | Argine v1 (historical); data re-derived from raw logs by Argine2 |
| Related questions | — |
| Supersedes / superseded | — |

## Research question

Among the tested expert quantization levels for Qwen3-30B-A3B, where does the measured perplexity-vs-expert-bpw curve show a descriptive elbow?

*(Original wording, kept as history: "Where is the knee of the quality / size curve for expert quantisation?" Perplexity is used here as a quality proxy only.)*

## Hypothesis

[HYPOTHESIS] Quality degrades smoothly down to ~2 bits.

*Audit reading (2026-10-11):* the historical word 'smoothly' had no pre-specified quantitative definition. The supported closure claim is narrower: the four tested 30B quantization points show a descriptive elbow at IQ2_S, with substantially larger adjacent PPL-per-expert-bit slopes below it.

## Setup

Experts re-quantised with an importance matrix calibrated on real prose; perplexity signature ctx=512, 60 chunks, reference text hash da315163291ee5c9.

*Audited setup (2026-10-11):*
- **Model.** The four audited curve points are expert requantizations of the same Qwen3-30B-A3B model, requantized from Q4_K_M, not BF16. The importance matrix is historically described as calibrated on real prose.
- **Perplexity.** ctx=512, 60 chunks, reference text hash `da315163291ee5c9`. All four curve points carry the same recorded signature, `ctx=512;chunk=60;testo=da315163291ee5c9`.
- **bpw.** The bpw values (4.88 / 2.56 / 2.31 / 2.06) are **expert** bits/weight, not whole-model or whole-file bpw. They come from the historical Argine v1 table; the PPL records themselves carry no bpw field.
- **No knee method.** No formal knee-detection method, changepoint test, optimization criterion or confidence interval for a knee location was pre-specified.

## Commands

```bat
argine imatrix MODEL --chunks 128
argine quantize MODEL --experts IQ2_S
argine ppl OUT.gguf --text testo-riferimento.txt --ctx 512 --chunks 60
```

## Raw data

`raw/records.jsonl` (perplexity.jsonl records). Full log: `data/raw/argine-v1/perplexity.jsonl`.

*Audit note (2026-10-11):*
- `raw/records.jsonl` contains the archived PPL records.
- results.csv/raw contain an over-inclusive historical PPL pool, because the historical builder selected every PPL record with 60 chunks. The audited EXP-013 quality/bit curve uses the four Qwen3-30B points Q4_K_M, IQ2_S, IQ2_XS and the historically selected re-calibrated IQ2_XXS point.
- The pool also contains both historical IQ2_XXS calibration records, and a Qwen3-14B Q4_K_M row. The Qwen3-14B row belongs to the broader historical PPL pool and is not evidence for the EXP-013 within-model quantization curve.
- No evidence record is removed by the audit. `results.csv`, `raw/records.jsonl` and `tools/build_historical_data.py` are unchanged; any future cleanup of the extraction logic would be a separate data-hygiene change.

## Result

[MEASURED] Q4_K_M 12.0642; IQ2_S 13.6283; IQ2_XS 14.3578; IQ2_XXS 15.1946 (re-calibrated) / 16.4396 (first calibration).

**Audited table (2026-10-11).** Qwen3-30B-A3B; all rows have signature `ctx=512;chunk=60;testo=da315163291ee5c9`.

| point | model file (record) | expert bpw | PPL | error | role in audited curve |
|---|---|---:|---:|---:|---|
| Q4_K_M | `Qwen_Qwen3-30B-A3B-Q4_K_M.gguf` | 4.88 | 12.0642 | 0.2912 | yes |
| IQ2_S | `qwen30b-IQ2S.gguf` | 2.56 | 13.6283 | 0.3290 | yes |
| IQ2_XS | `qwen30b-IQ2XS.gguf` | 2.31 | 14.3578 | 0.3472 | yes |
| IQ2_XXS, historically re-calibrated | `qwen30b-IQ2-reale.gguf` | 2.06 | 15.1946 | 0.3709 | yes |
| IQ2_XXS, historical first calibration | `Qwen_Qwen3-30B-A3B-Q4_K_M-experts-IQ2_XXS.gguf` | 2.06 | 16.4396 | 0.4058 | no; preserved provenance |

- **Not part of this curve:** Qwen3-14B Q4_K_M, 13.2766 ± 0.3176, is present in results.csv/raw but is not part of this within-model curve.
- **Calibration identity:** the historical source identifies 15.1946 as the re-calibrated IQ2_XXS point and 16.4396 as the first-calibration point, but that calibration identity is not encoded explicitly in the archived PPL record itself.

## Interpretation

*[Historical interpretation, narrowed by the 2026-10-11 audit: "2.56 bpw (IQ2_S) is the knee: above it the curve is nearly flat, below it about 4x steeper per bit." This is superseded by the wording below.]*

[DERIVED, HISTORICAL AUDIT 2026-10-11] IQ2_S at 2.56 expert bits/weight is a descriptive empirical elbow among the four tested Qwen3-30B quantization points on this corpus/signature.
- The observed adjacent PPL-per-expert-bit slope is approximately 0.67 from Q4_K_M to IQ2_S, 2.92 from IQ2_S to IQ2_XS, and 3.35 from IQ2_XS to the historically selected re-calibrated IQ2_XXS point.
- Thus the two tested intervals below IQ2_S are approximately 4–5x steeper descriptively than the interval above it (4.3x and 5.0x).
- No formal knee location, changepoint or universal optimum is established.
- The comparison is a finite-difference description, not a significance test. The historical table's 0.68 / 2.92 / 3.32 were computed from rounded PPL values.

[SUPPORTED, HISTORICAL] **Audit closure (2026-10-11).**
- Status SUPPORTED is retained.
- The supported claim is the descriptive elbow at IQ2_S (2.56 expert bpw) on the tested four-point Qwen3-30B grid, under the recorded PPL signature.
- It is not a universal or optimal bpw, not the best quantization for every workload, model or MoE, and not a formally estimated knee or changepoint.
- Expert bpw must not be confused with whole-model bpw.
- The selected IQ2_XXS point (15.1946) depends partly on historical source text for its calibration provenance. The raw PPL record does not encode the calibration identity.
- IQ3_XXS (≈3.06 expert bpw, ≈11.34 GiB in the historical text) is not a measured EXP-013 point. Its size/fit statement is a historical planner-derived estimate.
- No new measurement. No rerun is required for historical closure ([historical audit](../../docs/research/HISTORICAL-AUDIT-EXP011-018.md)).

## Limitations

[LIMITATION] Requantised from Q4_K_M, not from BF16.

[LIMITATION, historical audit 2026-10-11]
1. Perplexity is the only quality proxy.
2. One reference corpus/signature: `ctx=512;chunk=60;testo=da315163291ee5c9`.
3. Only four quantization levels define the audited same-model curve.
4. No formal knee criterion, changepoint test, optimization objective or confidence interval for the knee location was pre-specified.
5. 2.56 is expert bpw, not whole-model bpw.
6. Quantizations were requantized from Q4_K_M, not BF16.
7. The raw PPL record does not encode the imatrix/calibration identity that distinguishes the two IQ2_XXS historical calibrations.
8. results.csv/raw are over-inclusive. They contain the first-calibration IQ2_XXS point, the historically selected re-calibrated IQ2_XXS point, and a Qwen3-14B row unrelated to the within-30B quality/bit curve.
9. IQ3_XXS is not a measured EXP-013 PPL point; any size/fit statement about it is a historical planner-derived estimate.
10. EXP-013 predates the Argine2 reproducibility protocol.

## Status history

2026-10-01 SUPPORTED. 2026-10-11 historical audit: SUPPORTED retained for the descriptive empirical elbow on the tested Qwen3-30B quantization grid; universal/optimal-knee wording narrowed; expert-bpw semantics, duplicate IQ2_XXS calibration provenance and the over-inclusive historical extraction documented; no new measurement.
