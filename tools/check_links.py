"""Check every relative link in every Markdown file of the repository.

    python tools/check_links.py

A link to a file must point to an existing file or directory; a link with
`#anchor` into a Markdown file must match a heading of that file, using the
same anchor rule as GitHub (implemented in strumenti/controlla_readme.py).
External links (http/https/mailto) are not fetched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "strumenti"))
import controlla_readme as cr  # noqa: E402

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
SKIP_DIRS = {".git", "__pycache__", "llama-noomp", "llama-omp", "llama-patched", "node_modules"}


def anchors(md: Path) -> set:
    lines = md.read_text(encoding="utf-8").splitlines()
    return {cr.ancora(t[2]) for t in cr.titoli(lines)}


def outside_code(text: str) -> str:
    out, inside = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
            continue
        if not inside:
            out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def markdown_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(
        ROOT / Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    )


def check() -> list:
    problems = []
    cache = {}
    for md in markdown_files():
        if any(p in SKIP_DIRS for p in md.relative_to(ROOT).parts):
            continue
        text = outside_code(md.read_text(encoding="utf-8"))
        for rx in (LINK, IMG):
            for target in rx.findall(text):
                if re.match(r"^(https?|mailto):", target):
                    continue
                if target.startswith("#"):
                    own = cache.setdefault(md.resolve(), anchors(md))
                    if target[1:] not in own:
                        problems.append(f"{md.relative_to(ROOT)}: no heading for {target}")
                    continue
                path, _, anchor = target.partition("#")
                dest = (md.parent / path).resolve()
                if not dest.exists():
                    problems.append(f"{md.relative_to(ROOT)}: missing target {target}")
                    continue
                if anchor and dest.suffix == ".md":
                    if dest not in cache:
                        cache[dest] = anchors(dest)
                    if anchor not in cache[dest]:
                        problems.append(f"{md.relative_to(ROOT)}: no heading for #{anchor} in "
                                        f"{dest.relative_to(ROOT)}")
    return problems


if __name__ == "__main__":
    probs = check()
    print("\n".join(probs) if probs else "all relative links resolve")
    sys.exit(1 if probs else 0)
