# Reproducing Argine2

Argine2 distinguishes between **artifact reproduction** and **experimental replication**.

Artifact reproduction verifies analyses, tables, checks, and figures from the
evidence committed to this repository. Full reprocessing of experiments whose
run artifacts live under `experiments/EXP-*/raw/` requires the external raw
corpus identified by `data/external_raw_manifest.csv`. It does not require downloading or
running the evaluated language models.

Experimental replication performs new instrumented model runs. It requires
compatible GGUF models, an Argine expert store, the patched llama.cpp runtime,
and suitable local hardware. Absolute performance numbers are hardware- and
runtime-dependent and are therefore not expected to reproduce byte-for-byte on
a different machine.

## 1. Requirements

The analysis and verification path requires:

- Python 3.9 or newer;
- the Python dependencies listed in `requirements.txt`;
- Git for repository-integrity checks.

From the repository root:

```bash
python -m pip install -r requirements.txt
```

The repository can be used directly without installing the package. On Windows,
`argine.bat` invokes the code from the current checkout.

## 2. Verify the research artifact without a model

Run:

```bash
python tools/index_experiments.py --check
python tools/paper_figures.py --check
python -m unittest discover -s tests
```

These checks do not require model weights or network access.

Historical experiment tables can be regenerated from the committed Argine v1
raw logs with:

```bash
python tools/build_historical_data.py
```

The generated historical outputs are deterministic across supported platforms.

## 3. Regenerate the paper figures

The six main figures used by the paper are generated from committed experiment
data:

```bash
python tools/paper_figures.py
```

Use:

```bash
python tools/paper_figures.py --check
```

to validate the source data and figure invariants without rewriting the figure
files.

## 4. Raw and processed evidence

The main evidence locations are:

- `data/raw/argine-v1/` — imported Argine v1 execution records;
- `data/raw/legacy/` — legacy routing evidence;
- `data/processed/` — derived legacy analyses;
- `experiments/EXP-*/raw/` — external raw records for individual experiments; these are intentionally excluded from Git and cryptographically inventoried by `data/external_raw_manifest.csv` and `data/external_raw_summary.json`;
- `experiments/EXP-*/processed/` — derived experiment outputs;
- `experiments/EXP-*/results.csv` — experiment-level result tables;
- `experiments/INDEX.md` — experiment registry and final status.

Experiment conclusions are versioned with committed processed evidence and protocol metadata. The larger external raw corpus is bound to this publication candidate by per-file SHA-256 digests in `data/external_raw_manifest.csv`. Historical
conclusions are not silently rewritten when later experiments supersede or
qualify them.

## 5. Research command interface

The research command family is available with:

```bat
argine.bat research --help
```

Important commands include:

```text
trace        instrumented end-to-end runs
convert      legacy/raw log to router trace
characterize routing statistics per layer
replay       replay a policy under a hard memory budget
compare      compare policies across budgets
elasticity   per-layer memory elasticity
oracle       future-informed and static reference analyses
adaptive     adaptive residency controller and ablations
analyze      combined trace analysis
reprocess    regenerate a run from its raw runtime log
fidelity     runtime-versus-replay validation
iotimeline   queue-depth and latency reconstruction
predict      performance-model fitting and validation
report       experiment report generation
fingerprint  runtime/model/workload provenance
schedule     interleaved experimental schedules
```

Every research command writes its results to files; terminal output is not the
sole copy of an experimental result.

## 6. Experimental replication with models

Model weights are not distributed with Argine2.

The evaluated runtime is based on a pinned llama.cpp revision plus the patches
in `patches/`. Build instructions are given in the main `README.md`.

On the reference Windows workflow, a model run begins with a configured research
runtime, for example:

```bat
set ARGINE_LLAMA_DIR=%CD%\llama-research
argine.bat doctor
```

An expert store can then be built for a locally supplied GGUF model:

```bat
argine.bat store build C:\path\to\model.gguf
```

Instrumented experiments use commands such as:

```bat
argine.bat research trace C:\path\to\model.gguf --expert-store --budgets 4G,2G,1.5G --repeat 3 --workload coding --split dev --out experiments\SMOKE-example
```

Machine-local paths such as `C:\modelli\...` preserved in frozen experiment
configuration files are provenance from the reference machine. Replicators must
substitute their own local model and runtime paths.

## 7. Reproduction versus replication

The repository supports two distinct claims:

**Artifact reproduction** means checking and regenerating the repository-contained
derived outputs from committed evidence and verifying the internal consistency of the research artifact. Full raw reprocessing additionally requires the external corpus matching the committed SHA-256 manifest.

**Experimental replication** means collecting new measurements under a
compatible protocol. New performance measurements on other hardware extend the
evidence; they do not replace the reference-machine measurements.

Tokens/s, I/O latency, queue depth, and other wall-clock quantities can depend on
hardware, operating system, runtime build, background load, and storage state.
For this reason, the repository records fingerprints and experimental protocol
metadata rather than treating identical performance across machines as a
reproducibility requirement.

## 8. Paper

The paper source is in `paper/`.

Its figures are produced by `tools/paper_figures.py` from the committed
experimental evidence. The paper should be interpreted together with the
experiment registry and the individual `EXP-*` records, which contain the
scope, protocol, result, and limitations behind each reported claim.

## Public snapshot and historical provenance

The public Argine2 release is distributed from a clean repository snapshot rather than from the complete pre-publication Git history. Historical Argine2 commit identifiers embedded in frozen experiment protocols, configuration files, processed reports, and experiment narratives are intentionally retained as immutable provenance identifiers. They refer to the original research-development history and are not guaranteed to resolve as commits in the public snapshot history. External upstream commit identifiers, such as the pinned llama.cpp revision, retain their normal upstream meaning. The frozen experiment records themselves are part of the released artifact and are not rewritten to manufacture a new public lineage.
