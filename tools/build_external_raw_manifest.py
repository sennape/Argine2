from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build(root: Path):
    exp_root = root / "experiments"
    raw_roots = sorted(
        (p for p in exp_root.glob("EXP-*/raw") if p.is_dir()),
        key=lambda p: p.as_posix(),
    )
    if not raw_roots:
        raise SystemExit("ERROR: no experiments/EXP-*/raw directories found")

    rows = []
    summaries = []
    total_files = 0
    total_bytes = 0

    for raw_root in raw_roots:
        experiment = raw_root.parent.name
        files = sorted(
            (p for p in raw_root.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(root).as_posix(),
        )

        exp_bytes = 0
        for idx, path in enumerate(files, 1):
            if path.is_symlink():
                raise SystemExit(
                    f"ERROR: refusing to hash symlink in raw data: "
                    f"{path.relative_to(root).as_posix()}"
                )

            size = path.stat().st_size
            digest = sha256_file(path)
            rel = path.relative_to(root).as_posix()

            rows.append(
                {
                    "experiment": experiment,
                    "relative_path": rel,
                    "size_bytes": str(size),
                    "sha256": digest,
                }
            )
            exp_bytes += size

            if idx % 100 == 0:
                print(
                    f"{experiment}: hashed {idx}/{len(files)} files "
                    f"({exp_bytes / 1024**3:.3f} GiB)"
                )

        summaries.append(
            {
                "experiment": experiment,
                "files": len(files),
                "bytes": exp_bytes,
            }
        )
        total_files += len(files)
        total_bytes += exp_bytes
        print(
            f"{experiment}: DONE {len(files)} files, "
            f"{exp_bytes / 1024**3:.3f} GiB"
        )

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=["experiment", "relative_path", "size_bytes", "sha256"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    csv_text = csv_buffer.getvalue()
    manifest_sha256 = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()

    summary = {
        "schema_version": 1,
        "algorithm": "sha256",
        "scope": "experiments/EXP-*/raw/**",
        "manifest": "data/external_raw_manifest.csv",
        "manifest_sha256": manifest_sha256,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "experiments": summaries,
    }
    json_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    return csv_text, json_text, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic SHA-256 metadata for Argine2 raw experiment "
            "files that are intentionally excluded from Git."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rehash all external raw files and verify the committed manifests",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "external_raw_manifest.csv"
    json_path = root / "data" / "external_raw_summary.json"

    csv_text, json_text, summary = build(root)

    if args.check:
        if not csv_path.exists() or not json_path.exists():
            raise SystemExit("CHECK FAILED: manifest outputs do not exist")
        if csv_path.read_text(encoding="utf-8") != csv_text:
            raise SystemExit("CHECK FAILED: data/external_raw_manifest.csv differs")
        if json_path.read_text(encoding="utf-8") != json_text:
            raise SystemExit("CHECK FAILED: data/external_raw_summary.json differs")
        print(
            "CHECK OK: "
            f"{summary['total_files']} files, "
            f"{summary['total_bytes'] / 1024**3:.3f} GiB, "
            f"manifest sha256={summary['manifest_sha256']}"
        )
        return 0

    csv_path.write_text(csv_text, encoding="utf-8", newline="\n")
    json_path.write_text(json_text, encoding="utf-8", newline="\n")
    print(
        "WROTE: "
        f"{csv_path.relative_to(root).as_posix()} and "
        f"{json_path.relative_to(root).as_posix()}"
    )
    print(
        f"TOTAL: {summary['total_files']} files, "
        f"{summary['total_bytes'] / 1024**3:.3f} GiB"
    )
    print(f"MANIFEST SHA256: {summary['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
