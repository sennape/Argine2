"""Versioned workload corpus (sections 24-25).

Workloads live in `workloads/v<N>/<name>.json` inside the repository. Each file
holds two disjoint prompt sets:

* "dev"  — may be used to tune controller constants;
* "eval" — used only for final claims. Never tune on it.

A prompt is either single-turn ({"prompt": ...}) or multi-turn ({"turns":
[...]}: the turns are sent in sequence in one conversation, so the prefix
grows and the KV cache is reused). Each prompt carries a `segment` label that
ends up in the router trace, which is how workload transitions (section 22)
are located in the data.

The version of a workload is the SHA-256 of its file: changing one character
of a prompt makes it a different workload, and every trace records which.

Conversation-history semantics (multi-turn)
-------------------------------------------
A workload file may declare `conversation_history` (see `history.py`). Without
it the workload keeps the v1 behaviour: the assistant turn appended to the
history is `message.content` as returned, possibly "". `v2/multi-turn.json`
(name `multi-turn-v2`) declares the v2 semantics: only non-empty final content
enters the history, reasoning never does, and a turn without final content
invalidates the sample. Declared semantics other than the supported ones are
rejected, so a file can never be run under semantics it did not ask for.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2] / "workloads"
CURRENT = "v1"


@dataclass
class Prompt:
    id: str
    segment: str
    n_predict: int
    prompt: Optional[str] = None
    turns: List[Dict[str, object]] = field(default_factory=list)

    @property
    def multi_turn(self) -> bool:
        return bool(self.turns)


@dataclass
class Workload:
    name: str
    version: str
    path: Path
    sha256: str
    description: str
    prompts: Dict[str, List[Prompt]]
    # declared conversation-history semantics; None = v1 legacy behaviour
    history: Optional[Dict[str, object]] = None

    def split(self, which: str) -> List[Prompt]:
        if which not in self.prompts:
            raise KeyError(f"workload {self.name}: no '{which}' split "
                           f"(available: {', '.join(self.prompts)})")
        return self.prompts[which]


def available(version: str = CURRENT) -> List[str]:
    d = ROOT / version
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


SUPPORTED_HISTORY = {
    "assistant_source": "content",
    "reasoning_content_in_history": False,
    "empty_final_content_policy": "invalidate",
    "provenance_text": "reasoning_content_plus_content",
}


def _check_history(p: Path, h: object) -> Optional[Dict[str, object]]:
    if h is None:
        return None
    if not isinstance(h, dict):
        raise ValueError(f"{p}: conversation_history must be an object")
    for k, v in SUPPORTED_HISTORY.items():
        if h.get(k) != v:
            raise ValueError(f"{p}: unsupported conversation_history.{k}={h.get(k)!r} "
                             f"(supported: {v!r})")
    out = dict(h)
    out.setdefault("final_turn_requires_content", True)
    return out


def _resolve(name: str, version: str) -> Path:
    p = Path(name)
    if p.suffix:
        return p
    p = ROOT / version / f"{name}.json"
    if not p.exists():
        # versioned name, e.g. "multi-turn-v2" -> workloads/v2/multi-turn.json
        m = re.fullmatch(r"(.+)-(v\d+)", name)
        if m and (ROOT / m.group(2) / f"{m.group(1)}.json").exists():
            return ROOT / m.group(2) / f"{m.group(1)}.json"
    return p


def load(name: str, version: str = CURRENT) -> Workload:
    p = _resolve(name, version)
    if not p.exists():
        raise FileNotFoundError(f"workload not found: {p} (available: {', '.join(available(version))})")
    raw = p.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    prompts: Dict[str, List[Prompt]] = {}
    for split, items in d.get("splits", {}).items():
        prompts[split] = [Prompt(id=str(it["id"]), segment=str(it.get("segment", d["name"])),
                                 n_predict=int(it.get("n_predict", d.get("n_predict", 128))),
                                 prompt=it.get("prompt"), turns=list(it.get("turns", [])))
                          for it in items]
    ids = [x.id for v in prompts.values() for x in v]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{p}: duplicate prompt ids")
    texts = {}
    for split, v in prompts.items():
        for x in v:
            t = x.prompt or json.dumps(x.turns, sort_keys=True)
            if t in texts and texts[t] != split:
                raise ValueError(f"{p}: prompt {x.id} appears in both '{texts[t]}' and '{split}'")
            texts[t] = split
    return Workload(name=str(d["name"]), version=str(d.get("version", version)), path=p,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    description=str(d.get("description", "")), prompts=prompts,
                    history=_check_history(p, d.get("conversation_history")))
