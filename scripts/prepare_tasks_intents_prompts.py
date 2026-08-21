"""
Script to prepare prompts for task generation from agentlab chat messages.
Randomly samples n (default 4) step_*.json files from each trajectory and adds a user message
requesting the model to come up with a task following the original instructions.
"""

import datetime
import random
import re
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

import orjson

from agent_as_annotators.utils import load_all_model_configs


def find_latest_exp_dir(base_dir, agent_name, benchmark="exploration"):
    """
    Find the latest experiment directory matching the agent name and benchmark.

    Directory naming pattern: YYYY-MM-DD_HH-MM-SS_genericagent-{agent}-on-{benchmark}-{suffix}
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory {base_dir} does not exist")
        return None

    # Pattern to match directories with timestamp and keywords
    matching_dirs = []

    for d in base_path.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        # Check if it contains the benchmark pattern (e.g., "on-exploration")
        if f"on-{benchmark}" not in name:
            continue
        # Check if it contains the agent name
        if agent_name not in name:
            continue
        # Check if it starts with a date pattern
        if not re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", name):
            continue
        matching_dirs.append(d)

    if not matching_dirs:
        print(f"No directories found matching agent '{agent_name}' and benchmark '{benchmark}'")
        return None

    # Sort by name (timestamp is at the start, so lexicographic sort works)
    matching_dirs.sort(key=lambda x: x.name, reverse=True)
    latest = matching_dirs[0]

    print(f"Found {len(matching_dirs)} matching directories")
    print(f"Selected latest: {latest.name}")

    return latest

from agent_as_annotators.prompts import (
    TASK_INTENT_PROMPT_TEMPLATE,
    WEBARENA_ANNOTATOR_INSTRUCTIONS
)

def is_trajectory_dir(d):
    if d.name.startswith('.'):
        return False
    if not d.is_dir():
        return False
    if d.name.startswith('_'):
        return False
    if not "_on_" in d.name:
        return False
    return True

def parse_trajectory_dir_name(d):
    # first split by . to get <suffix>.{num}_{seed}
    before, _, num_seed = d.name.rpartition('.')
    num, _, seed = num_seed.partition('_')
    # get YYYY-MM-DD_HH-MM-SS_<agent>-<model>-on-<task>
    date_str, _, rest = before.partition('_')
    time_str, _, rest = rest.partition('_')
    agent, _, task = rest.partition('-on-')
    return {
        'date': date_str,
        'time': time_str,
        'agent': agent,
        'task_name': task,
        'task_num': int(num),
        'seed': int(seed)
    }

def get_trajectory_dirs(exp_dir):
    return [d for d in exp_dir.iterdir() if is_trajectory_dir(d)]

def get_step_number(step_file):
    return int(step_file.stem.split('_')[1].split('.')[0])

def main(exp_dir, output_dir, num_samples=2, seed=42, num_intents=2, skip_empty_steps=False):
    exp_dir = Path(exp_dir)
    output_dir = Path(output_dir)
    trajectory_dirs = get_trajectory_dirs(exp_dir)

    print(f"Found {len(trajectory_dirs)} trajectory directories in {exp_dir}")

    # if the output file exists, remove it to start fresh, but prompt the user
    if output_dir.exists():
        response = input(f"Output directory '{output_dir}' already exists. (d)elete / (c)ontinue? ")
        if response.lower() == 'd':
            # delete the output directory
            shutil.rmtree(output_dir)
        elif response.lower() == 'c':
            print("Continuing...")
        else:
            print("Invalid response. Exiting...")
            exit(0)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)  # For reproducibility

    fails = 0
    written = 0
    empty_conversation_prompts = 0
    dropped_empty_steps = 0
    for traj_dir in tqdm(trajectory_dirs, desc="Processing trajectories"):
        traj_info = parse_trajectory_dir_name(traj_dir)
        step_files = list(traj_dir.glob("step_*.json"))
        step_files.sort(key=get_step_number)
        step_files = step_files[1:]  # remove step 0

        if skip_empty_steps:
            # A trajectory's terminal step records no agent call, so it carries no
            # chat_messages and extracts to `messages: []`. Sampling it produces a prompt
            # whose only turn is the appended instruction -- the Task Designer is asked to
            # write tasks "based on the conversation above" with no conversation above, so
            # the resulting intents are ungrounded in the site.
            #
            # OFF BY DEFAULT so that the released A3-Synth generation reproduces exactly;
            # in that run 381 of 4497 prompts (8.5%) were of this kind. Turn it on for new
            # collections.
            keep = []
            for f in step_files:
                with open(f, 'rb') as fh:
                    if orjson.loads(fh.read()).get('messages'):
                        keep.append(f)
            dropped_empty_steps += len(step_files) - len(keep)
            step_files = keep

        if len(step_files) == 0:
            fails += 1
            print(f"No step files found in {traj_dir}, skipping.")
            continue
        
        # Sample up to 4 random step files
        sampled_files = random.sample(step_files, min(num_samples, len(step_files)))
        
        for step_file in sampled_files:
            task_num = traj_info['task_num']
            step_num = get_step_number(step_file)

            save_fname = f"task_{task_num}.step_{step_num}.json"
            save_path = output_dir / save_fname

            if save_path.exists():
                tqdm.write(f"Skipping {save_path} because it already exists")
                continue
            with open(step_file, 'rb') as f:
                step_data = orjson.loads(f.read())
            
            step_data['task_num'] = task_num
            step_data['step_num'] = step_num
            step_data['traj_info'] = traj_info  # Add trajectory info to step data
            
            # Add user message requesting task generation
            user_message = {
                "role": "user",
                "content": TASK_INTENT_PROMPT_TEMPLATE.format(
                    annotator_instructions=WEBARENA_ANNOTATOR_INSTRUCTIONS,
                    num_intents=num_intents
                )
            }

            if not step_data['messages']:
                empty_conversation_prompts += 1

            step_data['messages'].append(user_message)

            # add to messages

            with open(save_path, 'w') as f:
                f.write(orjson.dumps(step_data).decode('utf-8'))
            written += 1

    if fails > 0:
        print(f"Failed to process {fails} trajectories")

    print(f"Prompt files written: {written}")
    if dropped_empty_steps:
        print(f"Skipped {dropped_empty_steps} message-less step(s) (--skip-empty-steps)")
    if empty_conversation_prompts:
        pct = 100.0 * empty_conversation_prompts / written if written else 0.0
        print(
            f"WARNING: {empty_conversation_prompts} of {written} prompts ({pct:.1f}%) "
            "contain NO conversation -- only the appended task-generation instruction. "
            "These come from a trajectory's terminal step, which records no agent call. "
            "The intents generated from them are not grounded in any observed page. "
            "Re-run with --skip-empty-steps to exclude them."
        )

    print(f"Prepared prompts saved in {output_dir}")
    return {
        "written": written,
        "empty_conversation_prompts": empty_conversation_prompts,
        "dropped_empty_steps": dropped_empty_steps,
        "failed_trajectories": fails,
    }

if __name__ == "__main__":
    this_file = Path(__file__)
    model_configs_path = this_file.parent.parent / "configs" / "model_configs.json"
    shorthands_to_configs = load_all_model_configs(model_configs_path)

    parser = argparse.ArgumentParser(description="Prepare prompts for task generation from chat messages")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="",
        help="Directory containing trajectory subdirectories with step_*.json files",
    )
    parser.add_argument(
        "--find-latest",
        "-l",
        metavar="AGENT",
        help="Find the latest exp directory for the specified agent name (e.g., 'gemini-3-flash-preview')",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        default="exploration",
        help="Benchmark name to match when using --find-latest (default: exploration)",
    )
    parser.add_argument(
        "--base-dir",
        default="outputs/chat_messages",
        help="Base directory to search in when using --find-latest (default: outputs/chat_messages)",
    )
    parser.add_argument(
        "--output-dir-base",
        type=str,
        default="outputs/task_intents/prompts",
        help="Base directory to save the prepared prompts. The final output directory will be <output_dir_base>/<exploration_model_name>/",
    )
    parser.add_argument(
        "-e",
        "--exploration-model",
        type=str,
        default=None,
        choices=list(shorthands_to_configs.keys()),
        help="Model used for exploration (used for output directory naming). If --find-latest is provided, this is inferred automatically.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of step files to sample from each trajectory",
    )
    parser.add_argument(
        "--num_intents",
        type=int,
        default=2,
        help="Number of intents to request from the model per prompt",
    )
    parser.add_argument(
        "--skip-empty-steps",
        action="store_true",
        help="Exclude steps that carry no chat messages (a trajectory's terminal step) "
             "before sampling. OFF by default so the released A3-Synth generation "
             "reproduces exactly; in that run 381/4497 prompts (8.5%%) had no "
             "conversation at all. Recommended ON for new collections.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    # Determine exp directory
    if args.find_latest:
        exp_dir = find_latest_exp_dir(args.base_dir, args.find_latest, args.benchmark)
        if exp_dir is None:
            exit(1)
    elif args.exp_dir:
        exp_dir = args.exp_dir
    else:
        parser.error("Either --exp_dir or --find-latest must be specified")

    # Determine exploration model: --find-latest supersedes -e
    if args.find_latest and args.exploration_model is None:
        # Use --find-latest value as the exploration model
        exploration_model = args.find_latest
        if exploration_model not in shorthands_to_configs:
            parser.error(f"Model '{exploration_model}' from --find-latest not found in model configs. "
                         f"Available: {list(shorthands_to_configs.keys())}")
    elif args.exploration_model is not None:
        exploration_model = args.exploration_model
    else:
        parser.error("Either --find-latest or -e/--exploration-model must be specified")

    # Build output directory with model name subdirectory (matching generate_task_intents.py convention)
    exploration_model_name = shorthands_to_configs[exploration_model]["kwargs"]["model_name"]
    exploration_model_dirname = exploration_model_name.replace('/', '_')
    output_dir = Path(args.output_dir_base) / exploration_model_dirname
    print(f"Output directory: '{output_dir}'")

    main(exp_dir, output_dir, num_samples=args.num_samples, seed=args.seed,
         num_intents=args.num_intents, skip_empty_steps=args.skip_empty_steps)