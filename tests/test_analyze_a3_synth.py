"""Focused tests for the standalone A3-Synth statistics script."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_a3_synth import (
    DatasetFormatError,
    analyze_dataset,
    normalize_host,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "a3_synth_sample.jsonl"
SCRIPT = ROOT / "scripts" / "analyze_a3_synth.py"


class AnalyzeA3SynthTests(unittest.TestCase):
    def test_host_normalization_is_specific_to_instance_suffix(self) -> None:
        self.assertEqual(
            normalize_host("WA-Shopping-XL-12.Example.org"),
            "wa-shopping.example.org",
        )
        self.assertEqual(normalize_host("example-xl-name.org"), "example-xl-name.org")

    def test_fixture_statistics(self) -> None:
        stats = analyze_dataset(FIXTURE, max_n=4, top=2)

        self.assertEqual(stats.steps, 6)
        self.assertEqual(stats.steps_with_web_url, 5)
        self.assertEqual(stats.steps_without_web_url, 1)
        self.assertEqual(stats.distinct_instructions, 2)
        self.assertEqual(stats.average_steps_per_instruction, 3.0)
        self.assertEqual(stats.unique_paths, 3)
        self.assertEqual(stats.unique_urls, 5)
        self.assertEqual(
            [
                (row.site, row.unique_urls, row.unique_paths)
                for row in stats.site_coverage
            ],
            [("Shopping", 3, 2), ("Map", 2, 1)],
        )
        self.assertEqual(
            [(row.site, row.steps) for row in stats.top_functional_pages],
            [("Map", 2), ("Shopping", 2)],
        )
        self.assertEqual(stats.instruction_lengths.mean, 3.5)
        self.assertEqual(stats.instruction_lengths.median, 3.5)
        self.assertEqual(stats.instruction_lengths.minimum, 3)
        self.assertEqual(stats.instruction_lengths.maximum, 4)
        self.assertEqual(
            [
                (row.n, row.unique, row.total, row.distinct_n)
                for row in stats.ngram_diversity
            ],
            [(1, 7, 7, 1.0), (2, 5, 5, 1.0), (3, 3, 3, 1.0), (4, 1, 1, 1.0)],
        )

    def test_json_cli_is_deterministic(self) -> None:
        command = [
            sys.executable,
            str(SCRIPT),
            str(FIXTURE),
            "--top",
            "2",
            "--format",
            "json",
        ]
        first = subprocess.run(command, check=True, capture_output=True, text=True)
        second = subprocess.run(command, check=True, capture_output=True, text=True)

        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(json.loads(first.stdout)["unique_urls"], 5)

    def test_malformed_input_reports_the_source_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory) / "malformed.jsonl"
            malformed.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetFormatError, "line 1"):
                analyze_dataset(malformed)


if __name__ == "__main__":
    unittest.main()
