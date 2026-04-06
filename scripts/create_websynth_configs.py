"""
Generate WebSynth task config files in the same format as exploration.tasks.json.

This script creates websynth task configs from WebArena task files, converting them
into the format expected by the websynth benchmark. The output files will be saved
in llm_annotators/benchmarks/websynth/configs/ directory.
"""

import json
import ast
import re
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def instantiate_template(template: str, variables: dict, protected_keys: tuple = ('__SHOPPING__', '__SHOPPING_ADMIN__', '__REDDIT__', '__GITLAB__', '__WIKIPEDIA__', '__MAP__')):
    """
    Instantiate the task intents.
    """
    out = template
    for key, value in variables.items():
        if not isinstance(value, str):
            raise ValueError(f"Instantiation value must be a string: {value}. Please sanitize the instantiation.")

        if f"__{key}__" in protected_keys or key in protected_keys:
            continue
        
        out = out.replace(f"__{key}__", value)
        out = out.replace(f"{{{{{key}}}}}", value)
        out = out.replace(f"{{{key}}}", value)
        out = out.replace(key, value)
    return out

def dictify_exploration_config(exploration_config: list) -> dict:
    """
    Dictify the exploration config.
    """
    return {conf['task_id']: conf for conf in exploration_config}

def parse_completion(completion_text: str) -> list:
    """
    Parse the completion from the model.
    The format of the completion is:
    "<think>...</think>...<intent>...</intent><eval>...</eval><end><intent>...</intent><eval>...</eval><instantiation>...</instantiation><end>..."
    """    

    # first get the thinking
    if completion_text is None:
        logger.warning("Completion text is None. Skipping this completion.")
        return None

    think_start = completion_text.find("<think>")
    think_end = completion_text.find("</think>")
    if think_start == -1 or think_end == -1:
        # Try <thought> as an alternative (e.g. Gemini models)
        think_start = completion_text.find("<thought>")
        think_end = completion_text.find("</thought>")
        if think_start == -1 or think_end == -1:
            logger.warning("Think section not found in completion")
            think_content = ""
            think_offset = 0  # Start from beginning if no think tags
        else:
            think_content = completion_text[think_start + len("<thought>"):think_end].strip()
            think_offset = think_end + len("</thought>")
    else:
        think_content = completion_text[think_start + len("<think>"):think_end].strip()
        think_offset = think_end + len("</think>")

    # now, find everything inside <intent> and </intent>
    intent_templates = []
    evals = []
    instantiation_dicts = []

    remain_completion = completion_text[think_offset:]
    while remain_completion:
        intent_start_raw = remain_completion.find("<intent>")
        if intent_start_raw == -1:
            break
        intent_start = intent_start_raw + len("<intent>")
        intent_end = remain_completion.find("</intent>")
        if intent_end == -1:
            break

        # get the task intent
        intent_template = remain_completion[intent_start:intent_end].strip()
        intent_templates.append(intent_template)
        remain_completion = remain_completion[intent_end + len("</intent>"):]
        if intent_template == "":
            logger.warning("Incomplete intent found in completion")
        
        # get the eval
        eval_start = remain_completion.find("<eval>") + len("<eval>")
        eval_end = remain_completion.find("</eval>")
        eval_content = remain_completion[eval_start:eval_end].strip()
        evals.append(eval_content)
        remain_completion = remain_completion[eval_end + len("</eval>"):]
        if eval_content == "":
            logger.warning("Incomplete eval found in completion")

        # get the instantiation
        instantiation_start = remain_completion.find("<instantiation>") + len("<instantiation>")
        instantiation_end = remain_completion.find("</instantiation>")
        instantiation_dict = remain_completion[instantiation_start:instantiation_end].strip()
        if instantiation_dict == "":
            logger.warning("Incomplete instantiation found in completion")
            instantiation_dict = {}
        else:
            try:
                instantiation_dict = json.loads(instantiation_dict)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(instantiation_dict)
                    if not isinstance(parsed, dict):
                        raise ValueError(f"Expected dict, got {type(parsed).__name__}")
                    instantiation_dict = parsed
                except Exception:
                    # Try adding quotes to unquoted keys: {key: "val"} -> {"key": "val"}
                    try:
                        fixed = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', instantiation_dict)
                        instantiation_dict = json.loads(fixed)
                    except Exception as e:
                        logger.warning("Invalid instantiation found in completion: %s", e)
                        instantiation_dict = {}
        
        if isinstance(instantiation_dict, str):
            instantiation_dict = {}
        
        instantiation_dict = sanitize_instantiation(instantiation_dict)
        instantiation_dicts.append(instantiation_dict)
        remain_completion = remain_completion[instantiation_end + len("</instantiation>"):]

    out = []
    
    for intent_template, eval_content, instantiation_dict in zip(intent_templates, evals, instantiation_dicts):
        out.append({
            "thoughts": think_content,
            "intent_template": intent_template,
            "eval_template": eval_content,
            "instantiation": instantiation_dict,
            "intent": instantiate_template(intent_template, instantiation_dict),
            "eval": instantiate_template(eval_content, instantiation_dict),
        })
    
    return out

def create_task_configs_from_intents(intents: list, exploration_dict: dict) -> list:
    """
    Create task configs from the intents.
    """
    out =  []

    for intent in intents:
        task_dict = intent['task_dict']
        step_num = task_dict['step_num']
        task_id = task_dict['task_num']
        exploration_config = exploration_dict[task_id]
        assert exploration_config['task_id'] == task_id, f"Exploration config task_id {exploration_config['task_id']} does not match task_id {task_id}"

        intent_str = intent['intent']
        eval_str = intent['eval']

        out.append({
            "sites": exploration_config['sites'],
            "task_id": None,
            "intent_template_id": None,
            "intent": intent_str,
            "require_login": True,
            "storage_state": exploration_config['storage_state'],
            "start_url": exploration_config['start_url'],
            "geolocation": None,
            "intent_template": intent['intent_template'],
            "instantiation_dict": intent['instantiation'],
            "require_reset": False,
            "eval": {
                "eval_types": ["llm_judge_with_hint"],
                "reference_answers": {
                    "hint": eval_str,
                    "hint_template": intent['eval_template'],
                },
                "reference_url": "",
                "program_html": [],
                "string_note": "",
                "reference_answer_raw_annotation": eval_str,
            },
            "persona": exploration_config['persona'],
            "exploration_task_id": task_id,
            "exploration_step_num": step_num,
        })

    
    return out

def assign_task_ids(configs):
    """
    Assign task ids to the intents.
    """
    n = 0
    for config in configs:
        config['task_id'] = n
        config['intent_template_id'] = n
        n += 1
    return configs


def is_incomplete_completion(intent):
    return intent['instantiation'] == {} or intent['instantiation'] == "" or intent['eval'] == "" or intent['intent'] == ""

def count_incomplete_completions(intents):
    """
    Count the number of incomplete completions.
    """
    incomplete_completions = 0
    for intent in intents:
        if is_incomplete_completion(intent):
            incomplete_completions += 1
    return incomplete_completions

def remove_incomplete_completions(intents):
    """
    Remove incomplete completions.
    """
    return [intent for intent in intents if not is_incomplete_completion(intent)]

def is_incorrect_instantiation(instantiation, protected_keys: tuple = ('__SHOPPING__', '__SHOPPING_ADMIN__', '__REDDIT__', '__GITLAB__', '__WIKIPEDIA__', '__MAP__')):
    """
    Checks if the instantiation has __<SITE_NAME>__ in it, which is incorrect.
    """
    for key, value in instantiation.items():
        if not isinstance(value, str):
            raise ValueError(f"Instantiation value must be a string: {value}. Please sanitize the instantiation.")
        if f"__{value}__" in protected_keys or value in protected_keys:
            continue
        if value.startswith("__") and value.endswith("__"):
            return True
    return False

def count_incorrect_instantiations(intents):
    """
    Checks how many value of instantiation has __<SITE_NAME>__ in it, which is incorrect.
    """
    incorrect_instantiations = 0
    for intent in intents:
        if is_incorrect_instantiation(intent['instantiation']):
            incorrect_instantiations += 1
    return incorrect_instantiations

def remove_incorrect_instantiations(intents):
    """
    Removes the instantiations that have __<SITE_NAME>__ in it, which is incorrect.
    """
    return [intent for intent in intents if not is_incorrect_instantiation(intent['instantiation'])]

def sanitize_instantiation(instantiation: dict, protected_keys: tuple = ('__SHOPPING__', '__SHOPPING_ADMIN__', '__REDDIT__', '__GITLAB__', '__WIKIPEDIA__', '__MAP__')):
    """
    When the instantiation key is __<SITE_NAME>__, the value should be the same
    as the key.
    """
    new_instantiation = {}
    if isinstance(instantiation, list):
        # this is broken, we need to handle this case
        instantiation = instantiation[0]
    
    for key, value in instantiation.items():
        if isinstance(value, list):
            value = value[0]
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, str):
            continue # skip non-string values
        
        if key in protected_keys:
            new_instantiation[key] = key
        else:
            new_instantiation[key] = value
    
    return new_instantiation

def add_task_dict_to_intents(intents, task_dict):
    """
    Add the task dict to the intents.
    """
    for intent in intents:
        intent['task_dict'] = task_dict
    return intents

def create_chunks(configs, chunk_size, output_filename, max_intents_per_site=-1):
    """
    Create chunks of configs, optionally limiting total intents per site.
    """
    # first, we need to group the configs by sites
    grouped_configs = {}
    for config in configs:
        site = config['sites'][0]
        if site not in grouped_configs:
            grouped_configs[site] = []
        grouped_configs[site].append(config)

    out = []
    # then, we need to create chunks of configs for each site
    for site, site_configs in grouped_configs.items():
        # Limit intents per site if specified
        if max_intents_per_site > 0:
            site_configs = site_configs[:max_intents_per_site]

        n = 0
        for i in range(0, len(site_configs), chunk_size):
            out.append({
                "name": f"{site}-{n}.{output_filename}",
                "configs": site_configs[i:i+chunk_size]
            })
            n += 1
    return out

def main(input_dir, output_dir, output_filename, exploration_config_path, chunk_size=-1, max_intents_per_site=-1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # list all json files in the directory
    json_files = list(Path(input_dir).glob("*.json"))
    json_files.sort()

    with open(exploration_config_path, "r") as f:
        exploration_config = json.load(f)
    exploration_dict = dictify_exploration_config(exploration_config)

    intents = []
    n_unsuccessful_completions = 0
    for json_file in tqdm(json_files, desc="Parsing completions"):
        with open(json_file, "r") as f:
            task_dict = json.load(f)

        completion_text = task_dict['completion']['choices'][0]['message']['content']
        intents_partial = parse_completion(completion_text)

        if intents_partial is None:
            n_unsuccessful_completions += 1
            continue
        intents_partial = add_task_dict_to_intents(intents_partial, task_dict)
        intents.extend(intents_partial)

    print(f"Number of unsuccessful completions: {n_unsuccessful_completions} / {len(json_files)}")
    print(f"Number of incomplete completions: {count_incomplete_completions(intents)} / {len(intents)}")
    print("Removing incomplete completions...")
    intents = remove_incomplete_completions(intents)

    print(f"Number of incorrect instantiations: {count_incorrect_instantiations(intents)} / {len(intents)}")
    print("Removing incorrect instantiations...")
    intents = remove_incorrect_instantiations(intents)

    print(f"Number of intents remaining: {len(intents)}")

    configs = create_task_configs_from_intents(intents, exploration_dict=exploration_dict)
    configs = assign_task_ids(configs)

    if chunk_size > 0:
        print(f"Creating chunks of {chunk_size} configs per site (max {max_intents_per_site} per site)...")
        chunks = create_chunks(configs, chunk_size, output_filename=output_filename, max_intents_per_site=max_intents_per_site)
        for chunk in chunks:
            with open(output_dir / chunk['name'], "w") as f:
                json.dump(chunk['configs'], f, indent=4)
            print(f"Saved {len(chunk['configs'])} tasks to {output_dir}/{chunk['name']}")
    else:
        print(f"Saving all configs to {output_dir}/{output_filename}")
        with open(output_dir / output_filename, "w") as f:
            json.dump(configs, f, indent=4)
        print(f"Saved {len(configs)} tasks to {output_dir}/{output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate WebSynth task config files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/task_intents/completions/Qwen_Qwen3-32B",
        help="Path to input WebArena task file directory, where each json file is a LLM completion for a task intent"
    )
    parser.add_argument(
        "--exploration-config-path",
        type=str,
        default="llm_annotators/configs/exploration.tasks.json",
        help="Path to exploration config file, which will be used to get the persona and start url"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llm_annotators/configs/websynth/",
        help="Directory to save output configs"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="websynth.tasks.json",
        help="Name of the output file"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=-1,
        help="Chunk size to save the configs. If -1, save all configs in one file. Otherwise, save the configs in chunks of this size."
    )
    parser.add_argument(
        "--max-intents-per-site",
        type=int,
        default=-1,
        help="Maximum number of intents per site. If -1, no limit."
    )

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        exploration_config_path=args.exploration_config_path,
        chunk_size=args.chunk_size,
        max_intents_per_site=args.max_intents_per_site
    )


