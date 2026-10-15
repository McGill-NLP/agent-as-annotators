"""Tests for Exploration v2 difficulty stratification.

These import only `agent_as_annotators.utils.difficulty` and
`agent_as_annotators.prompts`, both of which are dependency-free, so they run
without browsergym / webarena / playwright installed.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent_as_annotators import prompts
from agent_as_annotators.utils.difficulty import (
    DIFFICULTY_LEVELS,
    EXPLORATION_MAX_STEPS,
    STEP_BAND_WIDTH,
    difficulty_level_tally,
    format_difficulty_level_tally,
    level_for_step,
    normalize_difficulty_level,
    normalize_step_band,
    step_band,
    task_info_for_level,
    validate_task_record,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG_DIR = ROOT / "agent_as_annotators" / "configs" / "a3_synth"


class StepBandTests(unittest.TestCase):
    def test_bands_tile_the_budget_without_gap_or_overlap(self) -> None:
        covered = []
        for level in DIFFICULTY_LEVELS:
            first, last = step_band(level)
            covered.extend(range(first, last + 1))

        self.assertEqual(covered, list(range(EXPLORATION_MAX_STEPS)))

    def test_documented_band_edges(self) -> None:
        self.assertEqual(step_band(1), (0, 9))
        self.assertEqual(step_band(2), (10, 19))
        self.assertEqual(step_band(3), (20, 29))
        self.assertEqual(step_band(4), (30, 39))
        self.assertEqual(step_band(5), (40, 49))

    def test_every_band_is_the_same_width(self) -> None:
        widths = {last - first + 1 for first, last in map(step_band, DIFFICULTY_LEVELS)}
        self.assertEqual(widths, {STEP_BAND_WIDTH})

    def test_level_for_step_inverts_step_band(self) -> None:
        for step in range(EXPLORATION_MAX_STEPS):
            level = level_for_step(step)
            first, last = step_band(level)
            self.assertTrue(first <= step <= last, f"step {step} not in L{level}")

    def test_step_outside_the_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            level_for_step(-1)
        with self.assertRaises(ValueError):
            level_for_step(EXPLORATION_MAX_STEPS)

    def test_unknown_level_is_rejected(self) -> None:
        for level in (0, 6, -1):
            with self.assertRaises(ValueError):
                step_band(level)


class NormalizeDifficultyLevelTests(unittest.TestCase):
    def test_none_passes_through(self) -> None:
        self.assertIsNone(normalize_difficulty_level(None))

    def test_valid_levels_pass_through(self) -> None:
        for level in DIFFICULTY_LEVELS:
            self.assertEqual(normalize_difficulty_level(level), level)

    def test_out_of_range_is_rejected(self) -> None:
        for level in (0, 6, 50, -1):
            with self.assertRaises(ValueError):
                normalize_difficulty_level(level)

    def test_booleans_are_rejected(self) -> None:
        # bool subclasses int, so True would otherwise validate as level 1.
        for value in (True, False):
            with self.assertRaises(TypeError):
                normalize_difficulty_level(value)

    def test_strings_and_floats_are_rejected_not_coerced(self) -> None:
        for value in ("3", "L3", 3.0, [3]):
            with self.assertRaises(TypeError):
                normalize_difficulty_level(value)

    def test_error_message_names_the_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_id=1234"):
            normalize_difficulty_level(9, source="task_id=1234")


class NormalizeStepBandTests(unittest.TestCase):
    def test_none_passes_through(self) -> None:
        self.assertIsNone(normalize_step_band(None))

    def test_matching_band_passes(self) -> None:
        self.assertEqual(normalize_step_band([10, 19], level=2), (10, 19))

    def test_band_contradicting_its_level_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"step_band \[30, 39\]"):
            normalize_step_band([30, 39], level=2)

    def test_malformed_bands_are_rejected(self) -> None:
        for value in ([10], [10, 19, 20], [10, "19"], [True, 19]):
            with self.assertRaises(ValueError):
                normalize_step_band(value)
        for value in ("10-19", 10):
            with self.assertRaises(TypeError):
                normalize_step_band(value)


class ValidateTaskRecordTests(unittest.TestCase):
    def test_record_without_a_level_returns_none(self) -> None:
        self.assertIsNone(validate_task_record({"task_id": 1, "intent": "do a thing"}))

    def test_explicit_none_returns_none(self) -> None:
        self.assertIsNone(validate_task_record({"difficulty_level": None}))

    def test_level_is_returned(self) -> None:
        self.assertEqual(validate_task_record({"difficulty_level": 4}), 4)

    def test_consistent_step_and_band_pass(self) -> None:
        record = {
            "task_id": 7,
            "difficulty_level": 4,
            "step_band": [30, 39],
            "exploration_step_num": 32,
        }
        self.assertEqual(validate_task_record(record), 4)

    def test_step_outside_its_own_band_is_rejected(self) -> None:
        record = {
            "task_id": 7,
            "difficulty_level": 1,
            "exploration_step_num": 32,
        }
        with self.assertRaisesRegex(ValueError, "falls outside the band"):
            validate_task_record(record)

    def test_step_is_ignored_when_no_level_is_declared(self) -> None:
        # A pre-v2 record carries exploration_step_num but no level. No level is
        # derived from it and nothing is checked against it.
        self.assertIsNone(validate_task_record({"exploration_step_num": 32}))

    def test_task_id_appears_in_the_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_id=99"):
            validate_task_record({"task_id": 99, "difficulty_level": 8})


class TallyTests(unittest.TestCase):
    def test_unfilled_bands_are_reported_as_explicit_zeros(self) -> None:
        records = [
            {"difficulty_level": 1, "exploration_step_num": 3},
            {"difficulty_level": 1, "exploration_step_num": 8},
            {"difficulty_level": 3, "exploration_step_num": 25},
        ]
        tally = difficulty_level_tally(records)

        self.assertEqual(tally, {1: 2, 2: 0, 3: 1, 4: 0, 5: 0, None: 0})
        self.assertEqual(
            format_difficulty_level_tally(tally),
            "L1=2, L2=0, L3=1, L4=0, L5=0, unlevelled=0",
        )

    def test_unlevelled_records_are_counted_separately(self) -> None:
        tally = difficulty_level_tally([{"task_id": 1}, {"difficulty_level": 5}])

        self.assertEqual(tally[None], 1)
        self.assertEqual(tally[5], 1)

    def test_tally_raises_on_the_first_bad_record(self) -> None:
        with self.assertRaises(ValueError):
            difficulty_level_tally([{"difficulty_level": 1}, {"difficulty_level": 7}])


class BackwardCompatibilityTests(unittest.TestCase):
    """`difficulty_level` absent or None must behave exactly as it did before.

    The guarantee has two halves: validation must accept such records, and
    nothing downstream may gain a key because of them.
    """

    def test_absent_level_yields_an_empty_task_info(self) -> None:
        # This is the payload GenericWebSynthTask.setup() returns as
        # browsergym's `info["task_info"]`. For a pre-v2 record it must be the
        # empty dict, NOT {"difficulty_level": None}.
        self.assertEqual(task_info_for_level(None), {})

    def test_present_level_is_carried_into_task_info(self) -> None:
        self.assertEqual(task_info_for_level(3), {"difficulty_level": 3})

    def test_shipped_a3_synth_configs_validate_as_unlevelled(self) -> None:
        config_files = sorted(SHIPPED_CONFIG_DIR.glob("*.tasks.json"))
        self.assertTrue(config_files, f"no shipped configs found in {SHIPPED_CONFIG_DIR}")

        total = 0
        for config_file in config_files:
            records = json.loads(config_file.read_text(encoding="utf-8"))
            tally = difficulty_level_tally(records, source=str(config_file))

            # Every published record is pre-v2: it must validate, and it must
            # tally as unlevelled rather than being assigned a level from its
            # exploration_step_num.
            self.assertEqual(
                tally[None],
                len(records),
                f"{config_file.name} gained difficulty levels it did not declare",
            )
            for level in DIFFICULTY_LEVELS:
                self.assertEqual(tally[level], 0, f"{config_file.name} L{level}")
            total += len(records)

        self.assertGreater(total, 0)


class ExplorationPromptTests(unittest.TestCase):
    def test_stop_string_is_present_in_every_exploration_template(self) -> None:
        stop = prompts.EXPLORATION_STOP_MESSAGE
        self.assertEqual(stop, "I am done exploring the websites.")

        for name in (
            "TASK_EXPLORATION_PROMPT_TEMPLATE",
            "TASK_EXPLORATION_PROMPT_TEMPLATE_WITH_MIN_STEPS",
            "TASK_EXPLORATION_PROMPT_TEMPLATE_COVERAGE",
        ):
            self.assertIn(stop, getattr(prompts, name), name)

        self.assertIn(stop, prompts.EXPLORATION_STRATEGY_INSTRUCTIONS)

    def test_strategy_instructions_are_framed_as_coverage_not_completion(self) -> None:
        text = prompts.EXPLORATION_STRATEGY_INSTRUCTIONS.lower()
        for phrase in ("breadth", "feature surfaces", "affordance", "coverage"):
            self.assertIn(phrase, text)
        self.assertIn("not solving a task", text)

    def test_build_exploration_prompt_fills_every_placeholder(self) -> None:
        built = prompts.build_exploration_prompt(persona="Name: Ada", min_steps=40)

        self.assertIn("Name: Ada", built)
        self.assertIn("at least 40 steps", built)
        self.assertIn(prompts.EXPLORATION_STOP_MESSAGE, built)
        self.assertIn("Breadth of pages", built)
        self.assertNotIn("{", built.replace("{{", "").replace("}}", ""))

    def test_legacy_templates_are_unchanged(self) -> None:
        # The v1 templates are still used by already-collected pipelines, so the
        # coverage reframing is additive rather than a rewrite.
        self.assertNotIn("Breadth of pages", prompts.TASK_EXPLORATION_PROMPT_TEMPLATE)
        self.assertNotIn(
            "Breadth of pages", prompts.TASK_EXPLORATION_PROMPT_TEMPLATE_WITH_MIN_STEPS
        )


if __name__ == "__main__":
    unittest.main()
