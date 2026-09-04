"""Offline tests for instruction extraction, weighting, and release coverage."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_benchmark_instruction_lengths import (
    fetch, instruction, load_metadata, record, summarize,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_benchmark_instruction_lengths.py"
SNAPSHOT = ROOT / "analysis/benchmark_instruction_lengths.json"


class InstructionLengthTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"webarena": {"webarena.1": "test", "webarena.2": "test"}}
        self.row = record("webarena", "webarena.1", "test", 0, "Find the item.", "fixture", "a" * 64)

    def test_goal_not_observation_or_model_messages(self):
        obs = {"goal": "Find the item.", "axtree_txt": "noise " * 1000,
               "chat_messages": [{"role": "assistant", "message": "noise"}]}
        self.assertEqual(instruction(obs), "Find the item.")

    def test_multiline_and_unicode_whitespace(self):
        row = record("webarena", "webarena.1", "test", 0,
                     "Find\u00a0the\nblue-green\titem.", "fixture", "a" * 64)
        self.assertEqual(row["word_count"], 4)

    def test_numbered_substeps_retained(self):
        goal = "1. Navigate there.\n2. Sort by date."
        self.assertEqual(record("webarena", "webarena.1", "test", 0, goal, "", "")["word_count"], 7)

    def test_text_goal_object_fallback(self):
        self.assertEqual(instruction({"goal_object": [
            {"type": "text", "text": "First task."}, {"type": "text", "text": "Second task."}
        ]}), "First task.\nSecond task.")

    def test_nontext_goal_object_rejected(self):
        with self.assertRaises(ValueError):
            instruction({"goal_object": [{"type": "image", "image": "not text"}]})
        with self.assertRaises(ValueError):
            instruction({"goal": 42})

    def test_empty_goal_is_missing(self):
        self.assertIsNone(instruction({"goal": " \n "}))
        self.assertIsNone(instruction(None))

    def test_missing_not_zero(self):
        missing = record("webarena", "webarena.2", "test", 0, None, "", "", "empty")
        result = summarize([self.row, missing], self.metadata)["webarena/test"]
        self.assertEqual(result["mean_words"], 3)
        self.assertEqual(result["observed_instructions"], 1)
        self.assertEqual(result["expected_tasks"], 2)
        self.assertFalse(result["complete"])

    def test_absent_tasks_are_reported(self):
        result = summarize([self.row], self.metadata)["webarena/test"]
        self.assertEqual(result["missing_tasks"], ["webarena.2"])

    def test_duplicate_task_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate task/seed"):
            summarize([self.row, copy.deepcopy(self.row)], self.metadata)

    def test_invalid_count_or_split_rejected(self):
        for field, value in [("word_count", True), ("word_count", 0),
                             ("split", "train"), ("task_seed", 1)]:
            row = dict(self.row, **{field: value})
            with self.assertRaises(ValueError):
                summarize([row], self.metadata)

    def test_metadata_filters_l2_only(self):
        data = b"task_name,level,browsergym_split\none,l1,test\ntwo,l2,train\nthree,l3,test\n"
        self.assertEqual(load_metadata(data, "workarena_l2"), {"two": "train"})

    def test_bad_cached_source_digest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            url = "https://example.invalid/source"
            (cache / hashlib.sha256(url.encode()).hexdigest()).write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fetch(url, cache, "a" * 64)

    def test_snapshot_regression(self):
        report = json.loads(SNAPSHOT.read_text())
        result = summarize(report["records"], report["metadata"])
        self.assertEqual(result, report["summary"])
        self.assertEqual(result["webarena/test"]["total_words"], 5695)
        self.assertEqual(result["workarena_l2/test"]["total_words"], 23125)
        self.assertEqual(result["workarena_l2/test"]["mean_words"], 125)
        self.assertEqual(result["workarena_l2/all"]["observed_instructions"], 340)
        self.assertEqual(result["workarena_l2/all"]["expected_tasks"], 341)
        # Public release has no test tasks. It is not a substitute for the local result.
        audit = report["public_audit"]
        self.assertEqual(audit["summary"]["workarena_l2/test"]["observed_instructions"], 0)
        self.assertEqual(audit["summary"]["workarena_l2/train"]["observed_instructions"], 152)
        self.assertEqual(len(audit["exact_instruction_hash_matches"]), 152)
        self.assertEqual(audit["instruction_hash_mismatches"], [])
        self.assertTrue(all("goal" not in r and "obs" not in r for r in report["records"]))

    def test_offline_cli_determinism(self):
        command = [sys.executable, str(SCRIPT), "--snapshot", str(SNAPSHOT)]
        a = subprocess.run(command, check=True, capture_output=True, text=True)
        b = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(a.stdout, b.stdout)
        self.assertIn("125.000000 words; 185/185 tasks", a.stdout)

    def test_pickle_trust_flag_required(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--results-root", "unused"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--trust-local-pickles", result.stderr)


if __name__ == "__main__":
    unittest.main()
