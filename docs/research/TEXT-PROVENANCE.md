# Generated-text provenance (fix of 2026-10-04)

## Symptom

[MEASURED] Smoke run `experiments/SMOKE-text-provenance` (30B, natural-language eval, 4 GiB) had three successful requests. Each showed:
- `predicted_n = 128`;
- `error = null`;
- `text_sha256 = e3b0c442…b855`, which is SHA-256 of the empty string.

Tokens were generated, but the hash was of an empty string.

## Root cause

The text was lost in the harness's response parsing. It was neither a serialization problem nor an empty answer.

1. **The harness reads only one field.** `argine research trace` calls `/v1/chat/completions` (non-streaming, see `runner._post_chat`) and read only `choices[0].message.content`.
2. **The generated text was in another field.** llama-server is started with `--jinja`. At the pinned llama.cpp commit (254b177) its default `--reasoning-format` is `deepseek` (`common/common.h`: `COMMON_REASONING_FORMAT_DEEPSEEK`). With reasoning extraction on, the auto-generated response parser is `optional(<think> reasoning(until </think>) </think>?)` (`common/chat-auto-parser-generator.cpp`).
3. **An unclosed thinking block fills `reasoning_content` completely.** Qwen3-30B-A3B is a hybrid thinking model: it starts every answer with `<think>`. Within 128–300 tokens it never reaches `</think>`, so all the text went into `message.reasoning_content` and `content` was `""` (`common/chat.cpp`, `to_json_oaicompat`).
4. **The harness then hashed the empty string silently.** `text = obj["choices"][0]["message"]["content"] or ""` sat inside a bare `try/except: pass`. An empty field, a missing field and a malformed response all gave `text = ""`, and therefore SHA-256("").

Corroboration across models (audit below):
- every Qwen3-30B-A3B run (a thinking model) has the empty-string hash;
- every Qwen3-Next-80B-A3B-**Instruct** run (a non-thinking model) has real, distinct content hashes.

Streaming, `choices[].text` versus `message.content`, the timing extraction and the JSONL serialization were all inspected; none of them loses text.

## Fix (provenance only)

New module [`argine/research/textprov.py`](../../argine/research/textprov.py), used by `runner.run_workload`.
- **Formats recognised:** `/v1/chat/completions` (`message.content` and `message.reasoning_content`, including content parts), the legacy `choices[].text`, and llama.cpp's native `/completion` `content`.
- **Hashing semantics:**
  - the generated text is `reasoning_content` followed by `content`, as returned, joined without a separator;
  - it is hashed as UTF-8 bytes with **no normalization** (no strip, no Unicode normalization, no newline changes);
  - both parts are also hashed separately (`reasoning_sha256`, `content_sha256`), which removes the ambiguity of the concatenation;
  - the text itself is **not stored**.

  The server strips the `<think>` tags, so the hash covers the returned text, not the raw token stream; `text_schema` records which fields were used.
- **New fields per record:** `workload`, `workload_sha256`, `split`, `text_length_chars`, `text_length_bytes`, `content_length_chars`, `reasoning_length_chars`, `text_schema`, `finish_reason`, `completion_tokens` (from `usage`), `text_provenance`, `text_provenance_detail`, `text_hash_semantics`.

  These come in addition to `prompt_id`, `segment`, `n_predict` and `predicted_n`.
- **Statuses (`text_provenance`):**

  | status | meaning | `text_sha256` |
  |---|---|---|
  | `ok` | text extracted, non-empty | hash of the text |
  | `empty_output` | the server reports 0 generated tokens and the fields are present but empty | SHA-256("") — the only case allowed to carry it |
  | `no_text_for_generated_tokens` | tokens were generated but the returned text fields are empty | `null` |
  | `extraction_failed` | no known response format | `null` |
  | `request_failed` | the HTTP request failed | `null` |
- **Failures are surfaced, not hidden:**
  - each non-OK record prints `TEXT PROVENANCE …` to stderr;
  - `summary.json` and `manifest.json` gain a `text_provenance` block;
  - `publication_check` lists "generated-text provenance" among the missing items, so the run is not publication-quality;
  - `argine research reprocess` prints the verdict.
- **Unchanged on purpose:**
  - **Multi-turn conversation history.** The assistant turn appended to the history is still `message.content` as returned. Changing it would change the prompts of the multi-turn workloads. (2026-10-10: v1 workloads keep this. The new workload `multi-turn-v2` declares content-only, non-empty history with invalidation; see [MULTITURN-V2-HISTORY](MULTITURN-V2-HISTORY.md).)
  - **Everything else:** routing, Expert Store, replay, controller and results are untouched.
- **Read-only audit:** [`tools/audit_text_provenance.py`](../../tools/audit_text_provenance.py).

## Historical runs (audit of 2026-10-04; nothing rewritten)

| runs | model | `responses.jsonl` status | verdict |
|---|---|---|---|
| EXP-007 `30b-coding-dev` (9), `30b-transition-dev-v2` (9); EXP-021 `30b-coding-dev-v2`, `30b-coding-dev-v2-confirm`; SMOKE-30B, SMOKE-30B-r2, SMOKE-text-provenance | Qwen3-30B-A3B (thinking) | every record SHA-256("") with predicted_n > 0 | **provenance-limited**: the generated text cannot be identified |
| EXP-007 `80b-coding-dev-v2` (9) | Qwen3-Next-80B-A3B-Instruct | non-empty content hashes | **legacy content-only hash**: equals today's `text_sha256` if the server returned no `reasoning_content`, as expected for an Instruct model |
| EXP-021 `linux-mid-moe-v2` (2) | test fixture | non-empty content hashes | legacy content-only hash |

The historical hashes stay as recorded. The audit derives the verdicts from the records; it does not write them into the run directories.

## What this means for the research

- [SUPPORTED] **Routing and cache analyses stay valid.** The router trace, the access records and every replay result are recorded by the runtime itself and never used the text. Bit-identical routing across repetitions (EXP-007, EXP-021) independently pins down what was computed. What is missing is the ability to say **which text** a trace belongs to.
- [DERIVED] **On the 30B, the generated tokens are the model's thinking.** This is inferred from the parser and the cross-model contrast; the post-fix smoke confirms it if `reasoning_length_chars > 0` and `content_length_chars = 0`. On the 30B the "natural-language / coding / math / JSON" segments are therefore thinking **about** those tasks, not the answers. The segments are still task-conditioned, but a characterization of "coding" routing on the 30B describes reasoning-mode generation. This is a scope note for EXP-007, not an error in the data.
- [DERIVED] **The 30B multi-turn sessions had empty assistant turns.** In the transition workload (and `multi-turn`), every assistant turn appended to the history was `""` on the 30B. Each new user turn therefore followed an empty assistant answer, not the previous answer. The recorded traces are what the model computed under that history. For comparability, the eval campaign should keep the same harness behaviour; this fix does. Whether the workload *should* send back the previous answers is a workload-design decision, and it would need a new workload version. That version is `multi-turn-v2` (2026-10-10, [MULTITURN-V2-HISTORY](MULTITURN-V2-HISTORY.md)).

## Validation

- Automated tests in `tests/test_textprov.py` (12) cover:
  1. legacy `choices[].text`;
  2. chat `message.content`;
  3. the Qwen3 case, with all text in `reasoning_content`, giving a non-empty hash;
  4. a real empty output versus "tokens but no text";
  5. malformed or missing fields never hashed as "";
  6. failures surfaced in the summary and the publication check.

  They also check:
  - historical empty hashes are reported as provenance-limited, not repaired;
  - end to end: a fake llama-server drives `run_workload`, and the test checks the hashes, that the text is not stored, and that the conversation history is unchanged.
- Smoke after the fix: the command is in the report of 2026-10-04 (`experiments\SMOKE-text-provenance-fixed`). Expected:
  - `text_provenance = ok`;
  - `reasoning_length_chars > 0`;
  - no `e3b0c442…` in `responses.jsonl`;
  - `text_provenance.verdict = complete` in `summary.json`.
