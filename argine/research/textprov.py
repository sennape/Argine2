"""Provenance of the generated text of every research request.

Why this module exists (2026-10-04)
-----------------------------------
Until this date `responses.jsonl` hashed only `choices[0].message.content` and
hashed "" whenever that field was missing or empty. llama-server is started
with `--jinja`; its default `--reasoning-format` (deepseek) moves everything
between `<think>` and `</think>` into `message.reasoning_content`, and an
UNCLOSED thinking block (Qwen3 with 128-300 token budgets never leaves it) goes
entirely there. `content` was therefore "" although `predicted_n` tokens had
been generated, and every record carried SHA-256("") =
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.

Hashing semantics (documented, stable)
--------------------------------------
* The generated text of a request is what the server returned for the
  generated tokens: `reasoning_content` followed by `content`, as returned,
  concatenated without separator (reasoning first: it is generated first).
  For schemas without reasoning (`/completion` `content`, legacy
  `choices[].text`) it is that string.
* Encoding: UTF-8 of the Python str decoded from the JSON response. No
  normalization of any kind (no strip, no Unicode normalization, no newline
  conversion).
* `text_sha256` = SHA-256 of those bytes, lowercase hex. The two parts are
  also hashed separately (`content_sha256`, `reasoning_sha256`), which removes
  the ambiguity of the concatenation.
* The text itself is NOT stored (only hashes and lengths).
* The parser strips the `<think>`/`</think>` tags, so the hash covers the
  returned text, not the raw token stream; `text_schema` records which fields
  were used.

Status of every record (`text_provenance`)
------------------------------------------
* ok                          text extracted, non-empty
* empty_output                fields present and empty, and the server reports
                              0 generated tokens: a real empty answer
* no_text_for_generated_tokens fields present and empty but tokens were
                              generated: NOT hashed as "" (text_sha256 = null)
* extraction_failed           no known response schema: text_sha256 = null
* request_failed              the HTTP request failed: text_sha256 = null
A record never carries SHA-256("") unless its status is empty_output.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
HASH_SEMANTICS = ("sha256(utf8(reasoning_content + content)) as returned by the server, "
                  "no normalization; parts hashed separately too; text not stored")
OK_STATES = ("ok", "empty_output")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract(obj: Any) -> Dict[str, Any]:
    """Find the generated text in a llama-server / OpenAI-style response.

    Returns {"schema", "content", "reasoning_content", "finish_reason",
    "completion_tokens", "problem"}. `content`/`reasoning_content` are None
    when the field is absent (not "")."""
    out: Dict[str, Any] = {"schema": None, "content": None, "reasoning_content": None,
                           "finish_reason": None, "completion_tokens": None, "problem": None}
    if not isinstance(obj, dict):
        out["problem"] = f"response is {type(obj).__name__}, not an object"
        return out
    usage = obj.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        out["completion_tokens"] = usage["completion_tokens"]
    choices = obj.get("choices")
    if isinstance(choices, list):
        if not choices or not isinstance(choices[0], dict):
            out["problem"] = "choices is empty or not a list of objects"
            return out
        ch = choices[0]
        out["finish_reason"] = ch.get("finish_reason")
        msg = ch.get("message")
        if isinstance(msg, dict):
            c, r = msg.get("content"), msg.get("reasoning_content")
            if isinstance(c, list):   # content parts
                parts = [p.get("text") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str)]
                c = "".join(parts) if parts or not c else None
            if c is not None and not isinstance(c, str):
                out["problem"] = f"message.content has type {type(c).__name__}"
                return out
            if r is not None and not isinstance(r, str):
                out["problem"] = f"message.reasoning_content has type {type(r).__name__}"
                return out
            if c is None and r is None:
                out["problem"] = "message has neither content nor reasoning_content"
                return out
            out.update(schema="chat.message" + ("+reasoning_content" if r is not None else ""),
                       content=c, reasoning_content=r)
            return out
        if isinstance(ch.get("text"), str):
            out.update(schema="completion.choices.text", content=ch["text"])
            return out
        out["problem"] = "choices[0] has neither message nor text"
        return out
    if isinstance(obj.get("content"), str):        # llama.cpp native /completion
        out.update(schema="native.content", content=obj["content"])
        if isinstance(obj.get("tokens_predicted"), int):
            out["completion_tokens"] = obj["tokens_predicted"]
        return out
    out["problem"] = "no choices and no content field"
    return out


def record(obj: Any, error: Optional[str], predicted_n: Optional[int]) -> Dict[str, Any]:
    """The text-provenance fields of one responses.jsonl record."""
    base: Dict[str, Any] = {"text_sha256": None, "text_length_chars": None,
                            "text_length_bytes": None, "content_sha256": None,
                            "content_length_chars": None, "reasoning_sha256": None,
                            "reasoning_length_chars": None, "text_schema": None,
                            "finish_reason": None, "completion_tokens": None,
                            "text_provenance": None, "text_provenance_detail": None,
                            "text_hash_semantics": HASH_SEMANTICS}
    if error:
        base.update(text_provenance="request_failed", text_provenance_detail=str(error)[:300])
        return base
    ex = extract(obj)
    base.update(text_schema=ex["schema"], finish_reason=ex["finish_reason"],
                completion_tokens=ex["completion_tokens"])
    if ex["problem"]:
        base.update(text_provenance="extraction_failed", text_provenance_detail=ex["problem"])
        return base
    c = ex["content"] if ex["content"] is not None else ""
    r = ex["reasoning_content"] if ex["reasoning_content"] is not None else ""
    text = r + c
    ntok = predicted_n if isinstance(predicted_n, int) else ex["completion_tokens"]
    if ex["content"] is not None:
        base.update(content_sha256=_sha(c), content_length_chars=len(c))
    if ex["reasoning_content"] is not None:
        base.update(reasoning_sha256=_sha(r), reasoning_length_chars=len(r))
    if text:
        base.update(text_sha256=_sha(text), text_length_chars=len(text),
                    text_length_bytes=len(text.encode("utf-8")), text_provenance="ok")
    elif ntok == 0:
        base.update(text_sha256=EMPTY_SHA256, text_length_chars=0, text_length_bytes=0,
                    text_provenance="empty_output")
    else:
        base.update(text_provenance="no_text_for_generated_tokens",
                    text_provenance_detail=(f"{ntok} tokens generated but the returned text fields "
                                            f"({ex['schema']}) are empty"
                                            if ntok is not None else
                                            "returned text is empty and the generated token count is unknown"))
    return base


def status_of(rec: Dict[str, Any]) -> str:
    """Status of any responses.jsonl record, including records written before
    this module existed (no `text_provenance` field)."""
    if rec.get("text_provenance"):
        return str(rec["text_provenance"])
    if rec.get("error"):
        return "request_failed"
    h = rec.get("text_sha256")
    if not h:
        return "legacy_no_hash"
    if h == EMPTY_SHA256:
        n = rec.get("predicted_n")
        return "legacy_empty_output" if n == 0 else "legacy_empty_hash_unverifiable"
    return "legacy_hash"


def summarize(responses: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Run-level verdict: complete only if every request has verified text
    provenance (ok or a real empty output)."""
    rs = list(responses or [])
    counts: Dict[str, int] = {}
    for r in rs:
        s = status_of(r)
        counts[s] = counts.get(s, 0) + 1
    bad = {k: v for k, v in counts.items() if k not in OK_STATES}
    if not rs:
        verdict = "no responses"
    elif not bad:
        verdict = "complete"
    elif set(bad) <= {"legacy_hash", "legacy_empty_output"}:
        # pre-fix records with a NON-empty hash: sha256(utf8(message.content)).
        # Equal to today's text_sha256 whenever the server returned no
        # reasoning_content (non-thinking models), which a legacy record cannot prove.
        verdict = "legacy content-only hash (recorded before the 2026-10-04 fix)"
    elif all(k.startswith("legacy_") for k in bad):
        verdict = "provenance-limited (empty-string hashes recorded before the 2026-10-04 fix)"
    else:
        verdict = "incomplete"
    return {"verdict": verdict, "complete": verdict == "complete", "requests": len(rs),
            "counts": counts, "hash_semantics": HASH_SEMANTICS}


def history_content(obj: Any) -> str:
    """The assistant turn appended to a multi-turn conversation: `content` as
    returned, or "" if absent. Unchanged from the pre-fix behaviour on purpose
    (changing it would change the workload's prompts); see TEXT-PROVENANCE.md."""
    try:
        c = obj["choices"][0]["message"]["content"]
        return c if isinstance(c, str) else ""
    except Exception:  # noqa: BLE001
        return ""
