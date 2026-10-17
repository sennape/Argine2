"""Argine2 research layer.

This package turns Argine from a runtime into an open research instrument for
one question:

    Is expert residency under severe memory constraints better modelled as an
    online resource-allocation problem than as a conventional cache-replacement
    problem?

It is organised along the research method of the project
(observe -> measure -> characterize -> hypothesize -> simulate -> implement ->
falsify -> document):

    trace        router-trace format, converters (route.log, raw runtime log)
    rawlog       parser of the runtime's raw research log (argine-raw v1)
    metrics      routing characterization: entropy, concentration, EWS,
                 working sets, reuse distances, co-activation, drift
    iotimeline   queue depth Q(t) with exact/upper-bound status, latency, hidden/unhidden I/O
    policies     cache / allocation policies for counterfactual replay
    replay       event-atomic shadow policy simulator
    oracle       Belady-style offline references and optimal static allocation
    elasticity   per-layer memory elasticity curves and stall-cost-per-byte
    allocator    allocation algorithms (uniform, greedy marginal, DP optimum)
    controller   AdaptiveResidencyController (replay prototype)
    costmodel    interpretable I/O stall model from measured device curves
    predictor    analytical per-token performance model and its error
    stats        repeated-measurement statistics (mean, CI, CV, schedules)
    fingerprint  build / model / workload fingerprints for every measurement
    workloads    versioned workload corpus loader
    runner       `argine research trace`: instrumented end-to-end runs
    analysis     one trace in, every Phase-1/Phase-2 table out
    io           deterministic CSV / JSON writers
    report       deterministic research report from stored experiment data
    cli          the `argine research ...` command family

Everything here is pure Python (standard library only) so that results can be
reproduced on any machine that can read the raw data. Figures use matplotlib
when it is available and are skipped (with a note) when it is not.
"""

RESEARCH_VERSION = "0.1.0"
