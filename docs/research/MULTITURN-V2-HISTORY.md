# Multi-turn v2: conversation-history semantics (2026-10-10)

> **Update 2026-10-11: the frozen version is `e978bde0341f`, with `n_predict` = 1536.**
> - The version described below as `ff89429e96c0` (`n_predict` 1024) was used only by the smoke run `experiments/EXP-007-router-characterization/raw/30b-multi-turn-v2-smoke-4G`. All four DEV turns ended with `finish_reason` = `length`: content was present but truncated.
> - The file was then changed in place to 1536 per turn (`e978bde0341f`). At 1536 all four turns ended with `stop` and non-empty content (`…-smoke-1536-4G`).
> - All 30B and 80B multi-turn-v2 cell runs record `e978bde0341f` and are `eligible`.
> - Editing a used file in place departs from the registry rule "add a new version". Only the smoke run used the 1024 file, and its manifest records that hash.
> - The 1024 file has been reconstructed byte-exactly and preserved, not as canonical, at [`workloads/superseded/v2-multi-turn-n1024-ff89429e96c0.json`](../../workloads/superseded/v2-multi-turn-n1024-ff89429e96c0.json). Its SHA-256 is `ff89429e96c0e8db9dca6b69bc85307b6f8f6206d25fde83b7f2cb7e6aacfe93`, equal to the smoke manifest's.
> - It differs from the frozen file only in the nine `n_predict` values (1024 vs 1536) and in line endings (LF vs CRLF).
> - The text below is kept as written on 2026-10-10. Where it says `ff89429e96c0` or 1024, the frozen values are `e978bde0341f` and 1536. The 4096 context was kept; `context_tokens` in each record shows the margin.

## Why a new workload version

[MEASURED, 2026-10-04 post-fix smoke] Qwen3-30B-A3B produced only `reasoning_content` at 128–300 tokens per turn: 554–1489 reasoning characters, `content` 0 characters, `finish_reason` = `length`.

[DERIVED] Under the v1 harness the assistant turn appended to a multi-turn conversation was `message.content` as returned. So on the 30B every assistant turn in the history was `""`, and each new user turn followed an empty answer ([TEXT-PROVENANCE](TEXT-PROVENANCE.md)).

That is a property of the workload, not of the model. Fixing it changes the prompts, so it needs a new workload identity, not an edit of v1.

## The two semantics

| | v1 (`workloads/v1/multi-turn.json`, `41909d23a8a7…`) | v2 (`workloads/v2/multi-turn.json`, name `multi-turn-v2`, `ff89429e96c0…`) |
|---|---|---|
| assistant history | `content` as returned | non-empty final `content` only |
| empty final content | appended as an empty assistant turn | **not appended**; the remaining turns of the sample are not sent; the sample is **not EXP-007-eligible** |
| `reasoning_content` in history | no (it was never read) | **never** (no fallback) |
| provenance text | `reasoning_content + content` (textprov, since 2026-10-04) | same |
| `n_predict` per turn | 128 | 1024 |
| prompt ids | `mt-dev-1`, `mt-eval-1` | `mt2-dev-1`, `mt2-eval-1` |
| user turns | identical | identical |
| status | preserved; superseded for the final EXP-007 comparison | **canonical for the final EXP-007 multi-turn campaign** |

The workload file declares v2 itself:

```json
"conversation_history": {
  "assistant_source": "content",
  "reasoning_content_in_history": false,
  "empty_final_content_policy": "invalidate",
  "final_turn_requires_content": true,
  "provenance_text": "reasoning_content_plus_content"
}
```

How the declaration is enforced:
- **Exact values only.** The loader accepts only these values and rejects any other declared semantics.
- **v1 cannot pick up v2 behaviour.** A file without the block keeps v1 behaviour exactly, and its `responses.jsonl` records are byte-for-byte the same format as before.
- **Pinned by hash.** The semantics are part of the file, so they are part of its SHA-256 and of every trace's `workload@sha`.

Two concepts that stay distinct:

```
provenance_text        = reasoning_content + content   -> text_sha256 (canonical_text_hash)
assistant_history_text = content                       -> assistant_history_sha256
```

## Turn statuses (`turn_history_status`, v2 only)

| status | condition | valid |
|---|---|---|
| `ok` | `content` is non-empty (not whitespace-only) and is appended as returned | yes |
| `no_final_assistant_content` | reasoning was returned but `content` is empty. The model did not reach a final answer within the budget. | no |
| `no_generated_text` | neither reasoning nor content was returned (`text_provenance` = `no_text_for_generated_tokens` or `empty_output`) | no |
| `extraction_failed` | the response could not be parsed | no |
| `request_failed` | the HTTP request failed | no |
| `not_run_previous_turn_invalid` | the turn was never sent because an earlier turn of the sample was invalid. Recorded in `conversation_history.json`, not in `responses.jsonl`, since no request was made. | no |

Missing text, parse failures and request failures keep their own statuses. They are never folded into `no_final_assistant_content`. The text-level status stays in `text_provenance`.

## Per-turn audit fields (`responses.jsonl`, v2 records)

New fields:
- `history_semantics` = `v2`, `sample_id`, `turn_index`, `turns_expected`, `followed_by_user_turn`;
- `turn_history_status`, `reasoning_content_present`, `content_present`;
- `content_truncated` (final content present but `finish_reason` = `length`);
- `assistant_history_appended`, `assistant_history_source` (`content` or null), `assistant_history_chars`, `assistant_history_sha256`;
- `context_tokens` = `cache_n + prompt_n + predicted_n`.

These extend the existing text-provenance fields, which are not duplicated:
- `text_sha256` is the canonical text hash;
- `content_sha256` and `content_length_chars`;
- `reasoning_sha256` and `reasoning_length_chars`;
- `finish_reason`, `text_provenance`.

## EXP-007 eligibility

A run is eligible only if both hold:
1. **Every planned sample ran all of its turns.** The plan comes from `conversation_history.json`.
2. **Every assistant turn passes the content checks.** This applies to each turn followed by a user turn, and also to the final turn (`final_turn_requires_content`). Each such turn needs:
   - `turn_history_status` = `ok`;
   - `content_present`;
   - `assistant_history_source` = `content`;
   - `assistant_history_chars` > 0;
   - `assistant_history_sha256` = `content_sha256`;
   - no reasoning in the history: the history hash equals neither `text_sha256` nor `reasoning_sha256` when reasoning was produced.

Where the verdict appears:
- `summary.json` → `conversation_history` (verdict, per-sample problems, turn status counts);
- `manifest.json` → `conversation_history_semantics` and `conversation_history`;
- the console line `conversation history v2: eligible | not eligible`;
- `publication_check`: an ineligible v2 run lists "conversation history (multi-turn v2 not EXP-007 eligible …)" under `missing`, so it is not publication-quality;
- `argine research reprocess` recomputes the verdict.

`argine research workloads` prints each multi-turn workload's semantics.

## Generation budget

[DERIVED] **Choice: 1024 tokens per turn, only in multi-turn v2.**
- Other workloads and `--ctx` (4096) are unchanged. A context change would alter the runtime configuration (KV memory) relative to the other EXP-007 cells.
- Reasoning is not in the history, so the prompt grows only by the final answers. A four-turn conversation therefore fits 4096 tokens unless the answers are very long.
- `context_tokens` records the margin. An overflow would surface as `request_failed`, never silently.

[OPEN] Whether 1024 is enough for Qwen3-30B-A3B to finish thinking on these prompts is measured by the smoke test, not assumed.
- If turns come back `no_final_assistant_content`, the sample is invalid by design.
- The remedy is a new workload version with a larger budget and an explicit larger `--ctx`. It is not a fallback to reasoning, and not disabling thinking.

## Claim discipline

- [LIMITATION] **v1 multi-turn results keep their own semantics.** They stay valid only for those original semantics (empty assistant turns on thinking models) and are never re-labelled as v2. No EXP-007 multi-turn run was recorded under v1.
- **The transition workload stays v1.** `30b-transition-dev-v2` keeps its v1 history semantics and is not affected. Giving transition v2 semantics would need its own new version; it is not part of this change.
- **Untouched:** raw traces, historical workload hashes, EXP-021 replay semantics, routing and cache instrumentation, and the single-turn workloads.

## Validation

- [`tests/test_multiturn_v2.py`](../../tests/test_multiturn_v2.py) has 16 tests. They cover:
  - reasoning + content → content-only history;
  - reasoning without content → invalid, nothing appended, provenance still complete;
  - whitespace-only content counts as no final answer;
  - content without reasoning → valid;
  - both empty with generated tokens → `no_generated_text`;
  - request and parse failures stay separate statuses;
  - the second request contains only assistant 1's content;
  - an invalid turn stops the sample and invalidates it, with skipped turns recorded;
  - a final turn without content → invalid;
  - forged reasoning in the history is detected;
  - `text_sha256` = sha256(reasoning + content);
  - an ineligible run is flagged by `publication_check`;
  - v1 is unchanged (hash, records, empty-history behaviour);
  - v2 has a distinct, reproducible hash and version;
  - unsupported semantics are rejected.
- **Smoke test (reference machine):** the command and expected output are in the 2026-10-10 report and in [EXP-007](../../experiments/EXP-007-router-characterization/README.md).
