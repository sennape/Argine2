"""Conversation-history semantics of multi-turn workloads (2026-10-10).

Two semantics exist; a workload file selects one and its SHA-256 pins it.

v1 (legacy, no `conversation_history` in the workload file)
    The assistant turn appended to the history is `message.content` as
    returned. With a thinking model whose answer stays inside an unclosed
    `<think>` block (Qwen3-30B-A3B at 128-300 tokens) that is "": the next user
    turn follows an empty assistant answer. Limitation identified after the
    2026-10-04 reasoning_content provenance fix (TEXT-PROVENANCE.md). v1 runs
    keep exactly this behaviour; their records are unchanged.

v2 (`conversation_history` declared, e.g. workloads/v2/multi-turn.json)
    provenance_text        = reasoning_content + content   (textprov, unchanged)
    assistant_history_text = content                        (never reasoning)
    A turn may be continued only if its final content is non-empty (not
    whitespace-only). Otherwise nothing is appended, nothing falls back to
    reasoning, the remaining turns of the sample are not sent, and the sample
    is not eligible for EXP-007.

Turn statuses (`turn_history_status`, v2 only)
    ok                            non-empty content appended to the history
    no_final_assistant_content    text was generated (reasoning) but content is
                                  empty: the model did not reach a final answer
    no_generated_text             neither reasoning nor content was returned
                                  (text_provenance no_text_for_generated_tokens
                                  or empty_output)
    extraction_failed             the response could not be parsed
    request_failed                the HTTP request failed
    not_run_previous_turn_invalid the turn was never sent (conversation_history.json only)
Only `ok` is valid. The text-level status stays in `text_provenance`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from . import textprov

VALID = "ok"
SKIPPED = "not_run_previous_turn_invalid"
STATUSES = ("ok", "no_final_assistant_content", "no_generated_text",
            "extraction_failed", "request_failed", SKIPPED)
FILE = "conversation_history.json"

V1_SEMANTICS = {
    "version": "v1",
    "assistant_source": "content_as_returned",
    "reasoning_content_in_history": False,
    "empty_final_content_policy": "append_empty",
    "provenance_text": "reasoning_content_plus_content",
    "note": ("legacy: an empty final content is appended as an empty assistant turn; "
             "limitation identified after the 2026-10-04 reasoning_content fix"),
}


def semantics(wl_history: Optional[Dict[str, Any]], multi_turn: bool) -> Optional[Dict[str, Any]]:
    """The `conversation_history_semantics` block of a run (None for
    single-turn workloads, which have no conversation history)."""
    if wl_history is None:
        return dict(V1_SEMANTICS) if multi_turn else None
    return {"version": "v2", **{k: wl_history[k] for k in (
        "assistant_source", "reasoning_content_in_history", "empty_final_content_policy",
        "provenance_text", "final_turn_requires_content")}}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def turn(obj: Any, prov: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    """v2 decision for one assistant turn.

    Returns (text to append to the history or None, audit fields). The text is
    `message.content` as returned; it is never reasoning_content."""
    tp = prov.get("text_provenance")
    content = None
    if tp not in ("request_failed", "extraction_failed"):
        content = textprov.extract(obj)["content"]
    rchars = prov.get("reasoning_length_chars") or 0
    if tp in ("request_failed", "extraction_failed"):
        status = tp
    elif isinstance(content, str) and content.strip():
        status = VALID
    elif rchars > 0:
        status = "no_final_assistant_content"
    else:
        status = "no_generated_text"
    hist = content if status == VALID else None
    fields = {
        "turn_history_status": status,
        "reasoning_content_present": rchars > 0,
        "content_present": status == VALID,
        "content_truncated": bool(status == VALID and prov.get("finish_reason") == "length"),
        "assistant_history_appended": hist is not None,
        "assistant_history_source": "content" if hist is not None else None,
        "assistant_history_chars": len(hist) if hist is not None else 0,
        "assistant_history_sha256": _sha(hist) if hist is not None else None,
        # canonical_text_hash is textprov's text_sha256 (reasoning + content);
        # content_length_chars / reasoning_length_chars are textprov's too.
    }
    return hist, fields


def _turn_ok(r: Dict[str, Any], require_content: bool) -> List[str]:
    pid = r.get("prompt_id")
    bad = []
    st = r.get("turn_history_status")
    if st != VALID:
        if require_content:
            bad.append(f"{pid}: turn_history_status={st}")
        return bad
    if r.get("assistant_history_source") != "content":
        bad.append(f"{pid}: assistant_history_source={r.get('assistant_history_source')}")
    if not (r.get("assistant_history_chars") or 0) > 0:
        bad.append(f"{pid}: assistant_history_chars=0")
    if not r.get("content_present"):
        bad.append(f"{pid}: content_present is false")
    if r.get("assistant_history_sha256") != r.get("content_sha256"):
        bad.append(f"{pid}: history hash differs from content hash")
    if (r.get("reasoning_length_chars") or 0) > 0 and \
            r.get("assistant_history_sha256") in (r.get("text_sha256"), r.get("reasoning_sha256")):
        bad.append(f"{pid}: reasoning_content found in the assistant history")
    return bad


def eligibility(responses: Optional[List[Dict[str, Any]]],
                plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """EXP-007 eligibility of a multi-turn v2 run.

    Eligible only if every sample planned for the run has all its turns, and
    every assistant turn followed by another user turn (and the final turn,
    when final_turn_requires_content) has content_present, history source
    `content`, assistant_history_chars > 0, history hash == content hash, and
    no reasoning copied into the history.

    `plan` is conversation_history.json (written by run_workload); without it
    the expected samples are inferred from the records."""
    rs = [r for r in (responses or []) if r.get("history_semantics") == "v2"]
    sem = (plan or {}).get("conversation_history_semantics")
    if not rs and not (sem and sem.get("version") == "v2"):
        return {"applicable": False,
                "semantics": sem or None,
                "verdict": "not applicable (no v2 conversation-history semantics)"}
    require_final = bool((sem or {}).get("final_turn_requires_content", True))
    expected: Dict[str, int] = {}
    for s in (plan or {}).get("samples", []):
        expected[s["sample_id"]] = int(s["turns_expected"])
    for r in rs:
        expected.setdefault(r["sample_id"], int(r.get("turns_expected") or 0))
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rs:
        by.setdefault(r["sample_id"], []).append(r)
    samples = []
    counts: Dict[str, int] = {}
    for sid, n in expected.items():
        turns = sorted(by.get(sid, []), key=lambda r: r.get("turn_index", 0))
        problems: List[str] = []
        for r in turns:
            counts[r.get("turn_history_status")] = counts.get(r.get("turn_history_status"), 0) + 1
            followed = r.get("turn_index", 0) < n
            problems += _turn_ok(r, require_content=followed or require_final)
        if len(turns) < n:
            counts[SKIPPED] = counts.get(SKIPPED, 0) + (n - len(turns))
            problems.append(f"{sid}: {n - len(turns)} of {n} turns not run")
        samples.append({"sample_id": sid, "turns_expected": n, "turns_run": len(turns),
                        "eligible": not problems, "problems": problems})
    ok = bool(samples) and all(s["eligible"] for s in samples)
    return {"applicable": True, "semantics": sem, "eligible": ok,
            "verdict": "eligible" if ok else "not eligible",
            "samples": samples, "turn_status_counts": counts,
            "rule": ("every assistant turn followed by a user turn"
                     + (" and the final turn" if require_final else "")
                     + " needs non-empty final content as its history; reasoning never in history")}
