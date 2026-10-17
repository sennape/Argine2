"""Generated-text provenance of research requests (2026-10-04 fix).

The bug: responses.jsonl hashed `choices[0].message.content` only, and hashed
"" when it was missing. With llama-server --jinja and Qwen3 the generated
tokens are returned in `message.reasoning_content`, so every record carried
SHA-256("") although predicted_n tokens had been generated.
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

from argine.research import runner, textprov, workloads  # noqa: E402

EMPTY = hashlib.sha256(b"").hexdigest()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def chat(content, reasoning=None, n=5, finish="length"):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"completion_tokens": n}, "timings": {"predicted_n": n}}


class TestExtraction(unittest.TestCase):
    def test_1_legacy_completion_choices_text(self):
        obj = {"choices": [{"text": "Hello world", "finish_reason": "stop"}]}
        r = textprov.record(obj, None, 3)
        self.assertEqual(r["text_schema"], "completion.choices.text")
        self.assertEqual(r["text_sha256"], sha("Hello world"))
        self.assertEqual(r["text_provenance"], "ok")

    def test_2_chat_message_content(self):
        r = textprov.record(chat("Answer."), None, 2)
        self.assertEqual(r["text_schema"], "chat.message")
        self.assertEqual(r["text_sha256"], sha("Answer."))
        self.assertEqual(r["content_sha256"], sha("Answer."))
        self.assertIsNone(r["reasoning_sha256"])

    def test_2b_reasoning_content_is_the_generated_text(self):
        """The case that produced SHA-256("") in every historical run."""
        r = textprov.record(chat("", reasoning="Okay, the user asks about vaccines..."), None, 128)
        self.assertEqual(r["text_provenance"], "ok")
        self.assertEqual(r["text_sha256"], sha("Okay, the user asks about vaccines..."))
        self.assertNotEqual(r["text_sha256"], EMPTY)
        self.assertEqual(r["content_sha256"], EMPTY)
        self.assertEqual(r["reasoning_length_chars"], len("Okay, the user asks about vaccines..."))
        self.assertEqual(r["text_schema"], "chat.message+reasoning_content")

    def test_2c_reasoning_then_content_concatenated_in_order(self):
        r = textprov.record(chat("JSON here", reasoning="think "), None, 9)
        self.assertEqual(r["text_sha256"], sha("think JSON here"))
        self.assertEqual(r["reasoning_sha256"], sha("think "))
        self.assertEqual(r["content_sha256"], sha("JSON here"))

    def test_3_non_empty_text_never_hashes_to_empty(self):
        for obj in (chat("x"), chat("", reasoning="y"), {"content": "z"},
                    {"choices": [{"text": "w"}]}):
            r = textprov.record(obj, None, 1)
            self.assertNotEqual(r["text_sha256"], EMPTY, obj)
            self.assertEqual(r["text_provenance"], "ok")

    def test_hash_is_utf8_without_normalization(self):
        s = "  Città\r\n ü \t"
        r = textprov.record(chat(s), None, 4)
        self.assertEqual(r["text_sha256"], hashlib.sha256(s.encode("utf-8")).hexdigest())
        self.assertEqual(r["text_length_chars"], len(s))
        self.assertEqual(r["text_length_bytes"], len(s.encode("utf-8")))

    def test_4_real_empty_output_is_distinguishable(self):
        real = textprov.record(chat("", n=0, finish="stop"), None, 0)
        self.assertEqual(real["text_provenance"], "empty_output")
        self.assertEqual(real["text_sha256"], EMPTY)
        lost = textprov.record(chat("", n=128), None, 128)
        self.assertEqual(lost["text_provenance"], "no_text_for_generated_tokens")
        self.assertIsNone(lost["text_sha256"])

    def test_5_malformed_or_missing_content_is_not_hashed_as_empty(self):
        for obj in ({}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]},
                    {"choices": [{"message": {"content": 7}}]}, "not json object",
                    {"choices": [{"message": {"role": "assistant"}}]}):
            r = textprov.record(obj, None, 10)
            self.assertEqual(r["text_provenance"], "extraction_failed", obj)
            self.assertIsNone(r["text_sha256"], obj)
            self.assertTrue(r["text_provenance_detail"])

    def test_request_failure_is_not_hashed(self):
        r = textprov.record({}, "HTTP 500", None)
        self.assertEqual(r["text_provenance"], "request_failed")
        self.assertIsNone(r["text_sha256"])


class TestRunLevel(unittest.TestCase):
    def test_6_failure_is_surfaced_in_summary_and_publication_check(self):
        recs = [{"prompt_id": "a", **textprov.record(chat("", n=128), None, 128)},
                {"prompt_id": "b", **textprov.record(chat("ok"), None, 2)}]
        sm = textprov.summarize(recs)
        self.assertFalse(sm["complete"])
        self.assertEqual(sm["verdict"], "incomplete")
        fp = {"missing": []}
        run = {"validation": {"instrumentation_confirmed": True}, "read_mode": {"effective": "x"},
               "expert_store": {"effective": "s", "fingerprint_matches_effective": True},
               "final_store_stats": {"last_snapshot": {"graph": 1}}, "trace_completeness": {"complete": True},
               "budget_violations": 0, "machine_state": {"warnings": []}, "text_provenance": sm}
        chk = runner.publication_check(fp, run)
        self.assertFalse(chk["publication_quality"])
        self.assertTrue(any("generated-text provenance" in m for m in chk["missing"]))
        run["text_provenance"] = textprov.summarize([recs[1]])
        self.assertTrue(runner.publication_check(fp, run)["publication_quality"])

    def test_historical_empty_hashes_are_provenance_limited_not_repaired(self):
        old = [{"prompt_id": "tr-dev-1#t1", "text_sha256": EMPTY, "predicted_n": 300, "error": None}]
        self.assertEqual(textprov.status_of(old[0]), "legacy_empty_hash_unverifiable")
        sm = textprov.summarize(old)
        self.assertFalse(sm["complete"])
        self.assertTrue(sm["verdict"].startswith("provenance-limited"))
        self.assertEqual(old[0]["text_sha256"], EMPTY)   # untouched
        content_only = [{"prompt_id": "x", "text_sha256": sha("abc"), "predicted_n": 3, "error": None}]
        self.assertEqual(textprov.status_of(content_only[0]), "legacy_hash")
        self.assertTrue(textprov.summarize(content_only)["verdict"].startswith("legacy content-only"))
        self.assertFalse(textprov.summarize(content_only)["complete"])


class _Fake(BaseHTTPRequestHandler):
    bodies = []
    replies = []

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        _Fake.bodies.append(json.loads(self.rfile.read(n)))
        rep = _Fake.replies.pop(0) if _Fake.replies else chat("", reasoning="r", n=2)
        raw = json.dumps(rep).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


class TestRunWorkloadEndToEnd(unittest.TestCase):
    def test_responses_jsonl_records_reasoning_text_and_history_is_unchanged(self):
        srv = HTTPServer(("127.0.0.1", 0), _Fake)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            d = Path(tempfile.mkdtemp())
            wf = d / "wl.json"
            wf.write_text(json.dumps({"name": "t", "version": "v1", "n_predict": 4, "splits": {
                "dev": [{"id": "s1", "segment": "a", "turns": [
                    {"content": "q1", "segment": "a", "n_predict": 4},
                    {"content": "q2", "segment": "b", "n_predict": 4}]}]}}))
            wl = workloads.load(str(wf))
            _Fake.bodies.clear()
            _Fake.replies[:] = [chat("", n=8),                                   # warm-up
                                chat("", reasoning="thinking about q1", n=4),
                                chat("visible", reasoning="thinking about q2", n=4)]
            recs = runner.run_workload(srv.server_address[1], wl, "dev", d, warmup=True)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual([r["text_provenance"] for r in recs], ["ok", "ok"])
        self.assertEqual(recs[0]["text_sha256"], sha("thinking about q1"))
        self.assertEqual(recs[1]["text_sha256"], sha("thinking about q2visible"))
        self.assertEqual(recs[0]["workload"], "t")
        self.assertEqual(recs[0]["split"], "dev")
        on_disk = [json.loads(l) for l in (d / "responses.jsonl").read_text().splitlines()]
        self.assertNotIn(EMPTY, [r["text_sha256"] for r in on_disk])
        self.assertNotIn("thinking about q1", (d / "responses.jsonl").read_text())   # no text stored
        # conversation history: assistant turn = `content` as returned (unchanged behaviour)
        second = _Fake.bodies[2]["messages"]
        self.assertEqual(second[1], {"role": "assistant", "content": ""})


if __name__ == "__main__":
    unittest.main()
