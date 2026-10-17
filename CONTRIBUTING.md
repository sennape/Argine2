# Contributing to Argine2

Argine2 is a research project. The most valuable contributions produce evidence — including evidence that something does not work.

## Good contributions

- **Reproduce an existing experiment** (`EXP-NNN`) and report whether you get the same result.
- **Add a workload** to `workloads/v1/` (or propose `v2`), with disjoint `dev` and `eval` prompts.
- **Test an open hypothesis** (`RQ-NNN` in [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md)).
- **Implement a baseline policy** in `argine/research/policies.py` (ARC, LeCaR, a published MoE policy) — with a test that it never exceeds its budget.
- **Improve instrumentation** (`patches/argine-research.*`) — with a test in `argine-research-test.cpp`.
- **Analyse raw traces** already in `data/` or `experiments/*/raw/`.
- **Reproduce a result on other hardware**, clearly labelled as a different machine (it extends, it does not replace, the reference results).
- **Disprove an interpretation.** If a README paragraph says more than its data, say so with the data.

## Rules that keep results comparable

1. **Cite IDs.** Every result names the `RQ-` it addresses and the `EXP-` it belongs to. New experiments use the next free `EXP-NNN-short-name` identifier. IDs are never reused or renamed.
2. **Raw data first.** Commit raw outputs (`raw.tsv`, `trace.jsonl.gz`, CSV/JSON). Screenshots and Markdown tables are not data.
3. **Fingerprint.** Runs made with `argine research trace` carry `manifest.json`; keep it. A result without a complete fingerprint is not publication-quality (`argine research fingerprint` tells you what is missing).
4. **Repeat and interleave.** At least 3 runs (5 preferred) per configuration, configurations interleaved inside one session (`argine research schedule --configs A,B,C --repeat 3`).
5. **Separate tuning from evaluation.** Tune on `dev`, claim on `eval`. Label in-sample results as such.
6. **Tag every statement** with `[MEASURED]`, `[DERIVED]`, `[HYPOTHESIS]`, `[SUPPORTED]`, `[FALSIFIED]`, `[OPEN]` or `[LIMITATION]`.
7. **Never rewrite old conclusions.** Mark them `SUPERSEDED by EXP-NNN` and add the new one.
8. **Keep the hard memory invariant.** No policy may exceed its budget, ever; replay fails loudly if one does.
9. Run `python -m unittest discover -s tests` and `python tools/index_experiments.py --check` before submitting.

## Contribution template

Use this in the pull request or issue body (also in `.github/ISSUE_TEMPLATE/research-contribution.md`):

```text
Related research question:
RQ-XXX

Related experiment:
EXP-XXX (or: new, EXP-0NN-short-name)

Hypothesis:
...

Method:
(model, quantisation, budget(s), workload + split, binary fingerprint, repetitions, order)

Result:
...

Does this:
SUPPORT / FALSIFY / EXTEND / REPLICATE

Raw data:
(paths in the repository)
```

A negative result is acceptable and welcome.

## Code style

Pure Python standard library in `argine/research` (numpy and matplotlib are optional accelerators / figure makers, never required for a result). Every module documents the definitions it implements next to the code. C++ follows the existing patches: self-contained, testable without llama.cpp, clean under ThreadSanitizer and AddressSanitizer.
