"""Multi-turn v2 conversation-history semantics (2026-10-10).

v2: assistant history = non-empty final `content` only; reasoning_content is
kept for provenance (text_sha256 = sha256(reasoning + content)) and never
enters the history; a turn without final content invalidates the sample.
v1: unchanged (content as returned, possibly "").
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argine.research import history, runner, textprov, workloads  # noqa: E402

V1_SHA = "41909d23a8a739e3"
V2_SHA = "e978bde0341f"   # frozen multi-turn-v2 used by every EXP-007 cell


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def chat(content, reasoning=None, n=5, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"completion_tokens": n},
            "timings": {"predicted_n": n, "prompt_n": 10, "cache_n": 0}}


class _Fake(BaseHTTPRequestHandler):
    bodies: list = []
    replies: list = []

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        _Fake.bodies.append(json.loads(self.rfile.read(n)))
        rep = _Fake.replies.pop(0) if _Fake.replies else chat("default answer", n=2)
        raw = json.dumps(rep).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


V2_DECL = {"assistant_source": "content", "reasoning_content_in_history": False,
           "empty_final_content_policy": "invalidate", "final_turn_requires_content": True,
           "provenance_text": "reasoning_content_plus_content"}


def _wl(d: Path, v2: bool, n_turns: int = 3):
    doc = {"name": "mt-test", "version": "v2" if v2 else "v1", "n_predict": 8, "splits": {
        "dev": [{"id": "s1", "segment": "multi-turn", "turns": [
            {"content": f"user{i + 1}", "segment": "multi-turn", "n_predict": 8}
            for i in range(n_turns)]}]}}
    if v2:
        doc["conversation_history"] = V2_DECL
    f = d / ("v2.json" if v2 else "v1.json")
    f.write_text(json.dumps(doc))
    return workloads.load(str(f))


def run(replies, v2=True, n_turns=3):
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        d = Path(tempfile.mkdtemp())
        wl = _wl(d, v2, n_turns)
        _Fake.bodies.clear()
        _Fake.replies[:] = list(replies)
        recs = runner.run_workload(srv.server_address[1], wl, "dev", d, warmup=False)
    finally:
        srv.shutdown()
        srv.server_close()
    plan = json.loads((d / history.FILE).read_text())
    return recs, [b["messages"] for b in _Fake.bodies], plan, d


class TestTurnDecision(unittest.TestCase):
    def _turn(self, obj, err=None):
        prov = textprov.record(obj, err, (obj.get("timings") or {}).get("predicted_n") if obj else None)
        return history.turn(obj, prov), prov

    def test_1_reasoning_and_content_history_is_content_only(self):
        (h, f), prov = self._turn(chat("Final answer.", reasoning="Let me think."))
        self.assertEqual(h, "Final answer.")
        self.assertEqual(f["turn_history_status"], "ok")
        self.assertEqual(f["assistant_history_source"], "content")
        self.assertEqual(f["assistant_history_sha256"], prov["content_sha256"])
        self.assertNotEqual(f["assistant_history_sha256"], prov["text_sha256"])
        self.assertTrue(f["reasoning_content_present"] and f["content_present"])

    def test_2_reasoning_without_content_is_invalid_and_never_history(self):
        (h, f), prov = self._turn(chat("", reasoning="thinking only", n=1024, finish="length"))
        self.assertIsNone(h)
        self.assertEqual(f["turn_history_status"], "no_final_assistant_content")
        self.assertFalse(f["assistant_history_appended"])
        self.assertIsNone(f["assistant_history_source"])
        self.assertEqual(f["assistant_history_chars"], 0)
        self.assertEqual(prov["text_provenance"], "ok")      # provenance still complete
        self.assertEqual(prov["text_sha256"], sha("thinking only"))

    def test_2b_whitespace_only_content_is_not_a_final_answer(self):
        (h, f), _ = self._turn(chat("\n\n", reasoning="r", n=50))
        self.assertIsNone(h)
        self.assertEqual(f["turn_history_status"], "no_final_assistant_content")

    def test_3_content_without_reasoning_is_valid(self):
        (h, f), prov = self._turn(chat("Plain answer."))
        self.assertEqual(h, "Plain answer.")
        self.assertEqual(f["turn_history_status"], "ok")
        self.assertFalse(f["reasoning_content_present"])
        self.assertEqual(f["assistant_history_sha256"], prov["text_sha256"])

    def test_4_both_empty_with_generated_tokens_is_invalid_and_distinct(self):
        (h, f), prov = self._turn(chat("", reasoning="", n=64))
        self.assertIsNone(h)
        self.assertEqual(prov["text_provenance"], "no_text_for_generated_tokens")
        self.assertEqual(f["turn_history_status"], "no_generated_text")

    def test_failure_statuses_stay_separate(self):
        (h, f), _ = self._turn({}, err="HTTP 500")
        self.assertEqual((h, f["turn_history_status"]), (None, "request_failed"))
        (h, f), _ = self._turn({"choices": [{"message": {"role": "assistant"}}]})
        self.assertEqual((h, f["turn_history_status"]), (None, "extraction_failed"))


class TestConversation(unittest.TestCase):
    def test_5_only_assistant1_content_reaches_request_2(self):
        recs, msgs, plan, _ = run([chat("A1 visible", reasoning="A1 hidden reasoning"),
                                   chat("A2 visible", reasoning="A2 hidden"),
                                   chat("A3 visible", reasoning="A3 hidden")])
        self.assertEqual(msgs[1], [{"role": "user", "content": "user1"},
                                   {"role": "assistant", "content": "A1 visible"},
                                   {"role": "user", "content": "user2"}])
        self.assertNotIn("hidden", json.dumps(msgs))
        self.assertEqual([r["turn_history_status"] for r in recs], ["ok"] * 3)
        self.assertEqual([r["turn_index"] for r in recs], [1, 2, 3])
        self.assertEqual([r["followed_by_user_turn"] for r in recs], [True, True, False])
        self.assertEqual(recs[0]["context_tokens"], 15)
        el = history.eligibility(recs, plan)
        self.assertTrue(el["eligible"], el)
        self.assertEqual(plan["conversation_history_semantics"]["version"], "v2")

    def test_empty_content_stops_the_sample_and_invalidates_it(self):
        recs, msgs, plan, d = run([chat("A1", reasoning="r1"),
                                   chat("", reasoning="still thinking", n=8, finish="length"),
                                   chat("never requested")])
        self.assertEqual(len(msgs), 2)                       # turn 3 not sent
        self.assertNotIn("still thinking", json.dumps(msgs))
        self.assertEqual(recs[1]["turn_history_status"], "no_final_assistant_content")
        self.assertEqual([t["turn_history_status"] for t in plan["samples"][0]["turns"]],
                         ["ok", "no_final_assistant_content", history.SKIPPED])
        el = history.eligibility(recs, plan)
        self.assertFalse(el["eligible"])
        self.assertEqual(el["turn_status_counts"][history.SKIPPED], 1)
        # also detected from the records alone (no plan file)
        self.assertFalse(history.eligibility(recs)["eligible"])
        on_disk = [json.loads(l) for l in (d / "responses.jsonl").read_text().splitlines()]
        self.assertEqual(len(on_disk), 2)

    def test_final_turn_without_content_invalidates(self):
        recs, _, plan, _ = run([chat("A1"), chat("", reasoning="r", n=8)], n_turns=2)
        self.assertFalse(history.eligibility(recs, plan)["eligible"])

    def test_reasoning_copied_into_history_is_detected(self):
        recs, _, plan, _ = run([chat("A1", reasoning="R1"), chat("A2")], n_turns=2)
        forged = [dict(r) for r in recs]
        forged[0]["assistant_history_sha256"] = forged[0]["text_sha256"]
        el = history.eligibility(forged, plan)
        self.assertFalse(el["eligible"])
        self.assertTrue(any("reasoning" in p or "differs" in p for p in el["samples"][0]["problems"]))

    def test_6_provenance_hash_is_canonical_reasoning_plus_content(self):
        recs, _, _, _ = run([chat("C1", reasoning="R1"), chat("C2"), chat("C3", reasoning="R3")])
        self.assertEqual(recs[0]["text_sha256"], sha("R1C1"))
        self.assertEqual(recs[0]["assistant_history_sha256"], sha("C1"))
        self.assertEqual(recs[2]["text_sha256"], sha("R3C3"))
        self.assertEqual(recs[0]["text_hash_semantics"], textprov.HASH_SEMANTICS)

    def test_publication_check_flags_ineligible_v2(self):
        el = history.eligibility([], {"conversation_history_semantics": {"version": "v2"},
                                      "samples": [{"sample_id": "s1", "turns_expected": 2}]})
        self.assertFalse(el["eligible"])
        run_ = {"validation": {"instrumentation_confirmed": True}, "read_mode": {"effective": "x"},
                "expert_store": {"effective": "s", "fingerprint_matches_effective": True},
                "final_store_stats": {"last_snapshot": {"graph": 1}},
                "trace_completeness": {"complete": True}, "budget_violations": 0,
                "machine_state": {"warnings": []}, "conversation_history": el}
        chk = runner.publication_check({"missing": []}, run_)
        self.assertTrue(any("conversation history" in m for m in chk["missing"]))


class TestVersioning(unittest.TestCase):
    def test_7_v1_unchanged_and_keeps_legacy_history(self):
        v1 = workloads.load("multi-turn")
        self.assertTrue(v1.sha256.startswith(V1_SHA))
        self.assertIsNone(v1.history)
        self.assertEqual(v1.prompts["dev"][0].n_predict, 128)
        recs, msgs, plan, _ = run([chat("", reasoning="r1", n=8), chat("A2"), chat("A3")], v2=False)
        self.assertEqual(len(msgs), 3)                       # v1 never stops a sample
        self.assertEqual(msgs[1][1], {"role": "assistant", "content": ""})
        self.assertNotIn("turn_history_status", recs[0])      # v1 records unchanged
        self.assertNotIn("history_semantics", recs[0])
        self.assertEqual(plan["conversation_history_semantics"]["version"], "v1")
        self.assertFalse(history.eligibility(recs, plan)["applicable"])

    def test_8_v2_is_a_distinct_reproducible_workload(self):
        v1, v2 = workloads.load("multi-turn"), workloads.load("multi-turn-v2")
        self.assertNotEqual(v1.sha256, v2.sha256)
        self.assertEqual(v2.sha256, workloads.load("multi-turn", "v2").sha256)
        self.assertEqual(v2.sha256, hashlib.sha256(v2.path.read_bytes()).hexdigest())
        self.assertEqual((v2.name, v2.version), ("multi-turn-v2", "v2"))
        self.assertEqual(v2.history["empty_final_content_policy"], "invalidate")
        self.assertFalse(v2.history["reasoning_content_in_history"])
        # same conversations, larger budget, distinct ids
        t1 = [t["content"] for s in v1.prompts.values() for p in s for t in p.turns]
        t2 = [t["content"] for s in v2.prompts.values() for p in s for t in p.turns]
        self.assertEqual(t1, t2)
        # frozen 2026-10-10 at 1536 tokens per turn (1024 truncated every 30B turn)
        self.assertTrue(v2.sha256.startswith(V2_SHA))
        self.assertTrue(all(p.n_predict == 1536 for s in v2.prompts.values() for p in s))
        self.assertTrue(all(int(t["n_predict"]) == 1536
                            for s in v2.prompts.values() for p in s for t in p.turns))
        self.assertFalse({p.id for s in v1.prompts.values() for p in s}
                         & {p.id for s in v2.prompts.values() for p in s})

    def test_unsupported_semantics_are_rejected(self):
        d = Path(tempfile.mkdtemp())
        f = d / "bad.json"
        f.write_text(json.dumps({"name": "bad", "conversation_history": {
            **V2_DECL, "reasoning_content_in_history": True}, "splits": {"dev": []}}))
        with self.assertRaises(ValueError):
            workloads.load(str(f))

    def test_other_v1_workloads_have_no_history_semantics(self):
        for name in workloads.available("v1"):
            self.assertIsNone(workloads.load(name).history, name)


if __name__ == "__main__":
    unittest.main()
