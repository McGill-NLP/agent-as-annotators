"""Exercise actual conversion properties without importing browser/GPU stacks."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "agent_as_annotators/utils/trajectories.py"


def properties(class_name, names):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    ns = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), ns)
    return type("ConversionFixture", (), {name: ns[name] for name in names})


class TrajectoryCompatibilityTests(unittest.TestCase):
    def test_both_model_argument_layouts(self):
        cls = properties("Trajectory", {"model_name", "model_args", "flags"})
        for args in (NS(model_name="example"), NS(chat_model_args=NS(model_name="example"), flags=NS(asdict=lambda: {"vision": True}))):
            obj = cls()
            obj.exp_args = NS(agent_args=args)
            self.assertEqual(obj.model_name, "example")
            self.assertIs(obj.model_args, getattr(args, "chat_model_args", args))
            self.assertEqual(obj.flags, {"vision": True} if hasattr(args, "flags") else {})

    def chat(self, messages, raw=None):
        cls = properties("Step", {"chat_messages"})
        obj = cls()
        obj.has_agent_info = lambda: True
        obj.info = NS(agent_info=NS(chat_messages=messages, extra_info={"raw_content": raw}))
        return obj.chat_messages

    def test_prompt_gets_exact_assistant_target_without_mutation(self):
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        raw = "<thought>reason</thought><action>click('1')</action>"
        self.assertEqual(self.chat(prompt, raw)[-1], {"role": "assistant", "content": raw})
        self.assertEqual(len(prompt), 2)

    def test_existing_assistant_and_discussion_are_preserved(self):
        messages = [{"role": "assistant", "content": "original"}]
        self.assertEqual(self.chat(messages, "replacement"), messages)
        self.assertEqual(self.chat(NS(to_openai=lambda: messages)), messages)

    def test_no_fabricated_completion(self):
        prompt = [{"role": "user", "content": "task"}]
        self.assertEqual(self.chat(prompt), prompt)
        self.assertEqual(self.chat(None), [])


if __name__ == "__main__":
    unittest.main()
