import csv
import os
import subprocess
from pathlib import Path

ROOT = Path.cwd()

EXP = ROOT / "experiments" / "EXP-007-router-characterization"

MODELS = {
    "30b": r"C:\modelli\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
    "80b": r"C:\modelli\Qwen3-Next-80B-A3B-Instruct-Q3_K_M.gguf",
}

CELLS = [
    ("30b", "structured-json", "dev"),
    ("30b", "structured-json", "eval"),
    ("80b", "structured-json", "dev"),
    ("80b", "structured-json", "eval"),
    ("30b", "italian", "dev"),
    ("30b", "italian", "eval"),
    ("80b", "italian", "dev"),
    ("80b", "italian", "eval"),
]

BUDGETS = ["4G", "2G", "1.5G"]


def run(cmd):
    print("\n============================================================")
    print(cmd)
    print("============================================================\n")

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        shell=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def valid_run(runtime_row, fidelity_row):
    reasons = []

    if int(runtime_row["violations"]) != 0:
        reasons.append("runtime violations")

    if runtime_row["trace_complete"].lower() != "true":
        reasons.append("trace incomplete")

    if int(fidelity_row["reconstruct_inconsistencies"]) != 0:
        reasons.append("reconstruct inconsistencies")

    if int(fidelity_row["incomplete_nodes_in_workload"]) != 0:
        reasons.append("incomplete workload nodes")

    if fidelity_row["workload_fidelity"] != "PASS":
        reasons.append("workload fidelity")

    if fidelity_row["exp021_closure"] != "PASS":
        reasons.append("EXP-021 closure")

    if int(fidelity_row["tail_disagreements_unexplained"]) != 0:
        reasons.append("unexplained tail disagreement")

    if int(fidelity_row["rp_budget_violations"]) != 0:
        reasons.append("replay budget violations")

    return len(reasons) == 0, reasons


def inspect_cell(raw_dir, fidelity_dir):
    runtime_rows = read_csv(raw_dir / "runs.csv")
    fidelity_rows = read_csv(
        fidelity_dir / "fidelity_runs.csv"
    )

    fidelity_map = {
        row["run"]: row
        for row in fidelity_rows
    }

    valid = {b: [] for b in BUDGETS}
    invalid = []

    for row in runtime_rows:

        run_id = int(row["run"])
        budget = row["budget"]

        key = f"run-{run_id:02d}-{budget}"

        f = fidelity_map[key]

        ok, reasons = valid_run(row, f)

        if ok:
            valid[budget].append(
                raw_dir / key
            )
        else:
            invalid.append(
                (key, budget, reasons)
            )

    return valid, invalid


def replacement(
    model,
    workload,
    split,
    budget,
    slot,
):
    safe_budget = budget.replace(".", "p")

    slug = (
        f"{model}-{workload}-{split}-v2-"
        f"replacement-{safe_budget}-slot{slot}"
    )

    raw = EXP / "raw" / slug
    fidelity = EXP / "fidelity" / slug

    if not (raw / "runs.csv").exists():

        run(
            f'argine.bat research trace '
            f'"{MODELS[model]}" '
            f'--expert-store '
            f'--budgets {budget} '
            f'--repeat 1 '
            f'--workload {workload} '
            f'--split {split} '
            f'--out "{raw}"'
        )

    if not (fidelity / "fidelity_runs.csv").exists():

        run(
            f'argine.bat research fidelity '
            f'"{raw}" '
            f'--out "{fidelity}"'
        )

    valid, invalid = inspect_cell(
        raw,
        fidelity,
    )

    if valid[budget]:
        return valid[budget][0]

    print(
        f"Replacement FAILED for "
        f"{model} {workload} {split} {budget}"
    )

    print(invalid)

    return None


def process_cell(
    model,
    workload,
    split,
):

    slug = (
        f"{model}-{workload}-{split}-v2"
    )

    raw = EXP / "raw" / slug

    fidelity = (
        EXP / "fidelity" / slug
    )

    print("\n\n############################################################")
    print(f"CELL: {slug}")
    print("############################################################\n")

    if not (raw / "runs.csv").exists():

        run(
            f'argine.bat research trace '
            f'"{MODELS[model]}" '
            f'--expert-store '
            f'--budgets 4G,2G,1.5G '
            f'--repeat 3 '
            f'--workload {workload} '
            f'--split {split} '
            f'--out "{raw}"'
        )

    if not (
        fidelity / "fidelity_runs.csv"
    ).exists():

        run(
            f'argine.bat research fidelity '
            f'"{raw}" '
            f'--out "{fidelity}"'
        )

    valid, invalid = inspect_cell(
        raw,
        fidelity,
    )

    if invalid:

        print("\nINVALID ORIGINAL RUNS:")

        for item in invalid:
            print(item)

    for budget in BUDGETS:

        missing = 3 - len(valid[budget])

        for slot in range(missing):

            rep = replacement(
                model,
                workload,
                split,
                budget,
                slot + 1,
            )

            if rep is not None:
                valid[budget].append(rep)

    qualified = all(
        len(valid[b]) >= 3
        for b in BUDGETS
    )

    if not qualified:

        print(
            f"\nCELL NOT QUALIFIED: {slug}"
        )

        for b in BUDGETS:
            print(
                b,
                len(valid[b]),
                "/3 valid"
            )

        return False

    print(
        f"\nCELL QUALIFIED: {slug}"
    )

    for b in BUDGETS:
        print(
            b,
            len(valid[b]),
            "/3 valid"
        )

    # Use first valid 4G run
    trace_dir = valid["4G"][0]

    analysis = (
        EXP
        / "analysis-v3"
        / (
            f"{model}-{workload}-"
            f"{split}-"
            f"{trace_dir.name}"
        )
    )

    report = analysis / "REPORT.md"

    if not report.exists():

        run(
            f'argine.bat research analyze '
            f'"{trace_dir / "trace.jsonl.gz"}" '
            f'--budgets 4G,2G,1.5G '
            f'--out "{analysis}"'
        )

    print(
        f"\nREPORT: {report}"
    )

    return True


def main():

    if "ARGINE_LLAMA_DIR" not in os.environ:

        raise RuntimeError(
            "ARGINE_LLAMA_DIR non impostato."
        )

    completed = 0

    for cell in CELLS:

        try:

            ok = process_cell(*cell)

            if ok:
                completed += 1

        except Exception as exc:

            print(
                "\nERROR:",
                cell,
                exc,
            )

    print("\n\n============================================================")
    print("EXP-007 BATCH COMPLETE")
    print("============================================================")

    print(
        f"Qualified cells in this batch: "
        f"{completed}/8"
    )

    if completed == 8:

        print(
            "\nEXP-007 progress:"
        )

        print(
            "24 / 28 = 85.7%"
        )

        print(
            "\nRemaining:"
        )

        print(
            "multi-turn 30B DEV/EVAL"
        )

        print(
            "multi-turn 80B DEV/EVAL"
        )


if __name__ == "__main__":
    main()