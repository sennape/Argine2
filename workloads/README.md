# Workload corpus

Versioned prompts for Argine2 measurements (requirement sections 24-25). `v1/` holds seven workloads, each with disjoint `dev` and `eval` prompt sets; the loader (`argine/research/workloads.py`) rejects a prompt that appears in both.

| workload | content | dev / eval |
|---|---|---|
| natural-language | expository explanations (English) | 3 / 3 |
| coding | writing, explaining and fixing code | 3 / 3 |
| reasoning-math | step-by-step mathematics | 3 / 3 |
| structured-json | JSON output | 3 / 3 |
| italian | Italian-language prompts | 3 / 3 |
| multi-turn | four-turn conversations, prefix reuse on | 1 / 1 |
| transition | one session switching natural language → code → math → JSON every ~300 tokens | 1 / 1 |

### v2

| workload | file | sha256 | change from v1 |
|---|---|---|---|
| multi-turn-v2 | `v2/multi-turn.json` | `e978bde0341f` (frozen 2026-10-10; a first 1024-token version `ff89429e96c0` was used only by one smoke run and is preserved, not canonical, at [`superseded/v2-multi-turn-n1024-ff89429e96c0.json`](superseded/v2-multi-turn-n1024-ff89429e96c0.json)) | Same user turns as `v1/multi-turn` (`41909d23a8a7`, unchanged). Declares `conversation_history`: the assistant history is the non-empty final `content` only, `reasoning_content` never enters the history, and an empty final content invalidates the sample. `n_predict` is 1536 per turn (v1: 128); 1024 truncated every turn of the 30B smoke run. Ids are `mt2-*`. |

Load it with `--workload multi-turn-v2` (or `--workload multi-turn --workload-version v2`).

**v1 `multi-turn`** is preserved and superseded for the final EXP-007 comparison. Its assistant history is `content` as returned, so on thinking models an empty final content becomes an empty assistant turn. This limitation was identified after the 2026-10-04 reasoning_content provenance fix.

**v2** is canonical for the final EXP-007 multi-turn campaign. **`transition`** keeps v1 semantics.

Details: [docs/research/MULTITURN-V2-HISTORY.md](../docs/research/MULTITURN-V2-HISTORY.md).

Rules: tune on `dev`, claim on `eval`. A workload's version is the SHA-256 of its file, recorded in every trace (`argine research workloads` prints them). Never edit a file in place once it has been used for a published result: add `v2/`.
