"""
Generate the `exploration.tasks.json` config consumed by `a3-explore`.

Reads personas produced by `scripts/create_personas.py`
(`outputs/personas/<exploration_model>/personas.json`) and writes one
exploration task per (persona, site) into
`agent_as_annotators/configs/<exploration_model>/exploration.tasks.json`,
which is where `a3-explore` discovers it via `importlib.resources`.
"""

import argparse
import json
from pathlib import Path

from agent_as_annotators.prompts import (
    TASK_EXPLORATION_PROMPT_TEMPLATE,
    TASK_EXPLORATION_PROMPT_TEMPLATE_WITH_MIN_STEPS,
)
from agent_as_annotators.utils import load_all_model_configs


SITE_NAME_TO_START_URL = {
    "shopping_admin": "__SHOPPING_ADMIN__",
    "shopping": "__SHOPPING__",
    "map": "__MAP__",
    "gitlab": "__GITLAB__",
    "reddit": "__REDDIT__",
    "wikipedia": "__WIKIPEDIA__",
}


def format_task_record_from_persona(persona_dict, site_name, intent_id, prompt=None, min_steps=10):
    """Format a single exploration task record from a persona/site pair."""
    matched_string = "I am done exploring the websites."

    persona = f"{persona_dict['name']}:\n{persona_dict['description']}"
    if min_steps is not None:
        intent = TASK_EXPLORATION_PROMPT_TEMPLATE_WITH_MIN_STEPS.format(persona=persona, min_steps=min_steps)
    else:
        intent = TASK_EXPLORATION_PROMPT_TEMPLATE.format(persona=persona)

    return {
        "sites": [site_name],
        "task_id": intent_id,
        "intent_template_id": intent_id,
        "intent": intent,
        "require_login": True,
        "storage_state": f"./.auth/{site_name}_state.json",
        "start_url": SITE_NAME_TO_START_URL[site_name],
        "geolocation": None,
        "intent_template": intent,
        "instantiation_dict": {},
        "require_reset": False,
        "eval": {
            "eval_types": ["string_match"],
            "reference_answers": {"exact_match": matched_string},
            "reference_url": "",
            "program_html": [],
            "string_note": "",
            "reference_answer_raw_annotation": matched_string,
        },
        "llm_prompt": prompt,
        "persona": persona_dict,
    }


def load_personas(model_name, base_dir="outputs/personas", filename="personas.json"):
    personas_path = Path(base_dir, model_name.replace("/", "_"), filename)
    with open(personas_path, "r") as f:
        return json.load(f)


def main(model_name, output_dir, min_steps=10):
    model_dir_name = model_name.replace("/", "_")
    personas = load_personas(model_name=model_name)
    site_names = list(SITE_NAME_TO_START_URL.keys())

    tasks = []
    for persona in personas:
        for site_name in site_names:
            tasks.append(
                format_task_record_from_persona(
                    persona, site_name, len(tasks), min_steps=min_steps
                )
            )

    output_path = Path(output_dir, model_dir_name, "exploration.tasks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(tasks)} tasks to {output_path}")
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=4)

    sample_output_path = Path(output_dir, model_dir_name, "exploration.sample.json")
    with open(sample_output_path, "w") as f:
        json.dump(tasks[:4], f, indent=4)
    print(f"Saved a 4-task sample to {sample_output_path}")


if __name__ == "__main__":
    this_file = Path(__file__).resolve()
    model_configs_path = this_file.parent.parent / "configs" / "model_configs.json"
    shorthands_to_configs = load_all_model_configs(model_configs_path)

    # Default output dir is the package's configs/ directory so that a3-explore
    # picks up the generated file automatically via importlib.resources.
    default_output_dir = this_file.parent.parent / "agent_as_annotators" / "configs"

    parser = argparse.ArgumentParser(
        description="Generate the exploration.tasks.json config for a3-explore",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen3-vl-32b-thinking",
        choices=list(shorthands_to_configs.keys()),
        help="Exploration model shorthand (used to namespace the output directory)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help="Base directory under which <exploration_model>/exploration.tasks.json is written",
    )
    parser.add_argument(
        "-n",
        "--min-steps",
        type=int,
        default=10,
        help="Minimum number of exploration steps requested in the persona prompt",
    )
    args = parser.parse_args()

    model_config = shorthands_to_configs[args.model]
    model = model_config["kwargs"]["model_name"]
    main(model_name=model, output_dir=args.output_dir, min_steps=args.min_steps)
