"""
This script generates 1000 personas with names, descriptions, skills, and interests.

It uses the OpenAI API to generate the personas.
"""

import os
import json
from pathlib import Path
import logging

import openai
from tqdm import tqdm

from agent_as_annotators.utils import (
    get_completion_kwargs,
    select_client_config,
    parse_completion_content,
    load_all_model_configs,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_personas_base_completions(
    model: str,
    client: openai.OpenAI,
    enable_thinking: bool = False,
    batch_size: int = 20,
    num_batches: int = 50,
    personas_dir: str = "outputs/personas",
    save: bool = True,
):
    """
    This function creates personas with names, skills, and interests.
    """
    message = f"Generate {batch_size} personas with a name, skills, and interests. Return only the personas in JSON format, one persona per line, with the keys being name, skills, and interests."

    out_path = Path(
        personas_dir, model.replace("/", "_"), "base_completions.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # if the file exists, load the completions
    if out_path.exists():
        with open(out_path, "r") as f:
            outputs = json.load(f)
    else:
        outputs = []

    # get the number of completions already generated
    num_completions = len(outputs)
    print("Found", num_completions, "completions")

    # then, generate the completions
    for i in tqdm(range(num_batches), desc="Generating personas"):
        if i <= (num_completions - 1):
            continue
        tqdm.write(f"Generating batch {i + 1} of {num_batches}")
        kwargs = get_completion_kwargs(model, enable_thinking)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            **kwargs,
        )
        outputs.append(json.loads(response.model_dump_json()))
        # dump model responses
        if save:
            with open(out_path, "w") as f:
                f.write(json.dumps(outputs, indent=4))

def romanize(n: int):
    """
    This function converts a number to a roman numeral from scratch.
    """
    if n <= 0:
        return ""
    
    # Mapping of values to roman numerals (from largest to smallest)
    val_to_roman = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'),
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'),
        (9, 'IX'),
        (5, 'V'),
        (4, 'IV'),
        (1, 'I')
    ]
    
    result = ""
    for value, numeral in val_to_roman:
        count = n // value
        if count:
            result += numeral * count
            n -= value * count
    
    return result
    
def deduplicate_personas(personas: list[dict]):
    """
    This function goes through the personas and deduplicates the names by
    appending a number to the end of the name using the roman numeral suffix.
    """
    names = set()

    for persona in personas:
        original_name = name = persona["name"]
        if name in names:
            i = 0
            while name in names:
                i += 1
                name = f"{original_name} {romanize(i)}"
            persona["name"] = name
        names.add(name)
    
    return personas

def generate_description_completions(
    personas: list[dict],
    model,
    client: openai.OpenAI,
    enable_thinking: bool = False,
    save=True,
):
    """
    This function adds a description to the personas.
    """

    out_path = Path(
        "outputs",
        "personas",
        model.replace("/", "_"),
        "descriptions_completions.json",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        with open(out_path, "r") as f:
            outputs = json.load(f)
        remaining_personas = [p for p in personas if p["name"] not in outputs.keys()]
        print("Found", len(outputs), "descriptions already generated.")
        print("Found", len(remaining_personas), "personas remaining to generate descriptions for.")
    else:
        outputs = {}
        remaining_personas = personas
        print("No descriptions found, starting from scratch with", len(remaining_personas), "personas")

    for persona in tqdm(remaining_personas, desc="Adding descriptions"):
        if not isinstance(persona, dict):
            print(f"Skipping {persona} because it is not a valid persona")
            continue
        if "name" not in persona:
            print(f"Skipping {persona} because it does not have a name")
            continue
        if "skills" not in persona:
            print(f"Skipping {persona} because it does not have skills")
            continue
        if "interests" not in persona:
            print(f"Skipping {persona} because it does not have interests")
            continue
        
        messages = [
            {
                "role": "user",
                "content": f"Generate a description for the persona {persona['name']}. Here are the details: {persona['skills']} and {persona['interests']}",
            }
        ]
        kwargs = get_completion_kwargs(model, enable_thinking)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        outputs[persona["name"]] = json.loads(completion.model_dump_json())

        if save:
            with open(out_path, "w") as f:
                json.dump(outputs, f, indent=4)

    return outputs


def parse_description_completions(descriptions: dict):
    """
    This function parses the description completions and returns a list of descriptions.
    """
    descriptions_out = {}
    for name, completion in descriptions.items():
        content = completion["choices"][0]["message"]["content"]
        parsed = parse_completion_content(content)
        descriptions_out[name] = parsed["message"]
    return descriptions_out


def parse_json_results(message: str):
    """
    The results are in the message, and are in JSON format
    """
    # if it's already a list, return it
    try:
        loaded = json.loads(message.strip())
        if isinstance(loaded, list):
            return loaded
    except Exception as e:
        pass
    
    results_stripped = [
        r.strip() for r in message.strip().split("\n") if r.strip() != ""
    ]
    out = []
    for r in results_stripped:
        try:
            out.append(json.loads(r))
        except Exception as e:
            # logger.error(f"Error parsing result {r}: {e}")
            continue
    return out

def is_valid_persona(persona: dict):
    """
    This function checks if the persona is incorrect.
    """
    if not isinstance(persona, dict):
        return False
    
    if not "name" in persona:
        return False

    return True

def process_personas(personas_dir: str, model: str, max_personas: int = 1000, save=False, remove_incorrect=True):
    """
    This function processes the personas and returns a list of personas with the keys being name, description, skills, and interests.
    """
    base_completions_path = Path(personas_dir, model.replace("/", "_"), "base_completions.json")
    personas_path = Path(personas_dir, model.replace("/", "_"), "personas.json")
    with open(base_completions_path, "r") as f:
        completions = json.load(f)

    results = []
    for completion in completions:
        content = completion["choices"][0]["message"]["content"]
        parsed = parse_completion_content(content)
        partial_results = parse_json_results(parsed["message"])
        print("Found", len(partial_results), "personas in the completion")
        results.extend(partial_results)
    
    if remove_incorrect:
        print("Total of", len(results), "personas found.")
        results = [r for r in results if is_valid_persona(r)]
        print("Total of", len(results), "valid personas remaining.")



    results = deduplicate_personas(results)
    print("Deduplicated", len(results), "personas")

    print(f"Truncating from {len(results)} to {max_personas} personas")
    results = results[:max_personas]

    # save the results to a file
    if save:
        save_path = personas_path
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

        print("Saved", len(results), "personas to", save_path)

    return results


def add_descriptions_to_personas(personas: list[dict], descriptions: dict):
    """
    This function adds the descriptions to the personas.
    """
    results = []
    for persona in personas:
        if "name" in persona and persona["name"] in descriptions:
            persona = persona.copy()
            persona["description"] = descriptions[persona["name"]]
        else:
            persona = persona.copy()
            persona["description"] = "No description available"
        results.append(persona)
    return results

def parse_args():
    import argparse

    this_file = Path(__file__)
    model_configs_path = this_file.parent.parent / "configs" / "model_configs.json"
    shorthands_to_configs = load_all_model_configs(model_configs_path)

    parser = argparse.ArgumentParser(
        description="Create personas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen3-vl-32b-thinking",
        choices=list(shorthands_to_configs.keys()),
        help="Model to use",
    )
    parser.add_argument(
        "-n",
        "--max-personas",
        type=int,
        default=250,
        help="Max personas to create",
    )
    parser.add_argument(
        "-d",
        "--personas-dir",
        type=str,
        default="outputs/personas",
        help="Directory to load the base completions from, in order to process the personas. The base completions file is expected to be named base_completions.json.",
    )

    args = parser.parse_args()
    args.model_config = shorthands_to_configs[args.model]
    return args

def save_personas(personas: list[dict], output_dir: str, model: str, personas_filename: str = "personas.json"):
    """
    This function saves the personas to a file.
    """
    save_path = Path(output_dir, model.replace("/", "_"), personas_filename)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(personas, f, indent=4)
    print("Saved", len(personas), "personas to", save_path)

if __name__ == "__main__":
    args = parse_args()
    max_personas = args.max_personas
    model_config = args.model_config
    model = model_config["kwargs"]["model_name"]

    config = select_client_config(model, model_config=model_config)
    client = openai.OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    generate_personas_base_completions(model=model, client=client, enable_thinking=True, save=True, personas_dir=args.personas_dir)
    personas = process_personas(personas_dir=args.personas_dir, model=model, max_personas=max_personas, save=False)
    raw_descriptions = generate_description_completions(
        personas, model, client=client, save=True
    )
    descriptions = parse_description_completions(raw_descriptions)
    personas = add_descriptions_to_personas(personas, descriptions)

    save_personas(personas=personas, output_dir=args.personas_dir, model=model)
