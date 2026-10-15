"""Registry discovery must be shared by CLIs and shorthand resolution."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_as_annotators.utils import load_all_model_configs, shorthand_to_model

ROOT = Path(__file__).resolve().parents[1]


class ModelRegistryTests(unittest.TestCase):
    def test_default_is_independent_of_cwd(self):
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as directory:
            original = Path.cwd()
            try:
                os.chdir(directory)
                configs = load_all_model_configs()
                self.assertTrue(configs)
                name = next(iter(configs))
                self.assertEqual(shorthand_to_model(name), configs[name]["kwargs"]["model_name"])
            finally:
                os.chdir(original)

    def test_override_and_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps({"models": {"test": {
                "shorthand": "test", "kwargs": {"model_name": "example"}
            }}}))
            with patch.dict(os.environ, {"A3_MODEL_CONFIGS": str(path)}):
                self.assertEqual(shorthand_to_model("test"), "example")
            with patch.dict(os.environ, {"A3_MODEL_CONFIGS": "/missing/registry.json"}):
                self.assertIn("test", load_all_model_configs(path))
                with self.assertRaisesRegex(FileNotFoundError, "A3_MODEL_CONFIGS"):
                    load_all_model_configs()

    def test_clis_share_default_loader(self):
        for name in ("eval", "explore", "synth"):
            tree = ast.parse((ROOT / "agent_as_annotators" / "cli" / f"{name}.py").read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "load_all_model_configs"]
            self.assertEqual(len(calls), 1)
            self.assertFalse(calls[0].args)

    def test_cli_default_models_are_shipped(self):
        with patch.dict(os.environ, {}, clear=True):
            configs = load_all_model_configs()
        for name in ("eval", "explore", "synth"):
            tree = ast.parse((ROOT / "agent_as_annotators" / "cli" / f"{name}.py").read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and any(isinstance(a, ast.Constant) and a.value == "--models" for a in n.args)]
            self.assertEqual(len(calls), 1)
            default = next(k.value for k in calls[0].keywords if k.arg == "default")
            for model in ast.literal_eval(default):
                self.assertIn(model, configs)


if __name__ == "__main__":
    unittest.main()
