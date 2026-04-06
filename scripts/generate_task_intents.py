#!/usr/bin/env python3
"""
Script to generate task intents from prepared prompts using OpenAI API.
Based on the selected instructions from README.md for running LLM to generate task intents.
"""

import argparse
import asyncio
import datetime
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List
import openai
from tqdm import tqdm
import orjson

from agent_as_annotators.utils import select_client_config, get_completion_kwargs, load_all_model_configs


def with_timeout_retry(func, max_retries: int = 4, retry_delay: float = 60.0) -> Any:
    """Wrap a function to retry on API timeout after a delay."""
    def wrapper(*args, **kwargs):
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except openai.APITimeoutError:
                tqdm.write(f"Request timed out. Retrying in {int(retry_delay / 60)} minutes... ({i + 1}/{max_retries})")
                time.sleep(retry_delay)
        return None
    return wrapper


async def async_with_timeout_retry(func, max_retries: int = 4, retry_delay: float = 60.0):
    """Async version: retry on API timeout."""
    for i in range(max_retries):
        try:
            return await func()
        except openai.APITimeoutError:
            tqdm.write(f"Request timed out. Retrying in {int(retry_delay)} seconds... ({i + 1}/{max_retries})")
            await asyncio.sleep(retry_delay)
    return None

# Pricing per 1M tokens (input, output) in USD
MODEL_PRICING = {
    # OpenAI models
    "gpt-5-mini": (.250, 2.000),
    # Gemini models (shorter keys to match both preview and non-preview)
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3-pro": (2.00, 12.00),
    # Default for self-hosted/unknown models (free)
    "default": (0.0, 0.0),
}

def get_model_pricing(model: str) -> tuple[float, float]:
    """Get pricing per 1M tokens (input, output) for a model."""
    model_lower = model.lower()
    for key in MODEL_PRICING:
        if key in model_lower:
            return MODEL_PRICING[key]
    return MODEL_PRICING["default"]

def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate cost in USD for given token counts."""
    input_price, output_price = get_model_pricing(model)
    cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    return cost

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: Input file {file_path} does not exist")
        return

    print(f"Loading data from {file_path}")

    data = []
    with open(file_path, 'rb') as f:
        for line in f:
            if line.strip():
                data.append(orjson.loads(line))
    return data

def create_completions(args, output_dir, input_dir, model_config: dict | None = None):
    # if it's vllm server, set base url (port auto-inferred from model_config if VLLM_BASE_URL not set)
    config = select_client_config(args.model, model_config=model_config)
    client = openai.OpenAI(api_key=config["api_key"], base_url=config["base_url"], timeout=args.timeout)
    # Get the actual model name for API calls (not the shorthand)
    model_name = model_config["kwargs"]["model_name"] if model_config else args.model
    enable_thinking = args.thinking_mode == "enable"
    # kwargs = get_qwen_completion_kwargs(enable_thinking) if "qwen" in args.model.lower() else {}
    kwargs = get_completion_kwargs(args.model, enable_thinking, model_config=model_config)
    # Prepare output directory
    # if it's already exists, remove it
    if output_dir.exists():
        # prompt user
        if args.delete_existing_output_dir:
            shutil.rmtree(output_dir)
            print(f"Deleted existing output directory {output_dir}")
        else:
            print(f"Output directory {output_dir} already exists. Use --delete-existing-output-dir to delete it.")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob("*.json"))
    input_files.sort()
    print(f"Found {len(input_files)} input files in {input_dir}")
    
    # Cost tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    completed_count = 0
    skipped_count = 0
    failed_to_complete_count = 0
    
    pbar = tqdm(input_files, desc="Generating intents")
    for json_path in pbar:
        with open(json_path, 'rb') as f:
            item = orjson.loads(f.read())
        
        task_num = item['task_num']
        out_fname = f"task_{task_num}.step_{item['step_num']}.json"
        out_path = output_dir / out_fname
        out_path_lock = out_path.with_suffix('.lock')
        failed_path = out_path.with_suffix('.failed')
        if out_path.exists():
            # Load existing file to count its tokens
            try:
                with open(out_path, 'rb') as f:
                    existing = orjson.loads(f.read())
                if 'completion' in existing and 'usage' in existing['completion']:
                    usage = existing['completion']['usage']
                    prompt_tokens = usage.get('prompt_tokens', 0) or 0
                    completion_tokens = usage.get('completion_tokens', 0) or 0
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_cost += calculate_cost(prompt_tokens, completion_tokens, args.model)
            except Exception:
                pass  # Skip if file can't be read
            skipped_count += 1
            pbar.set_postfix({
                'cost': f'${total_cost:.4f}',
                'in': f'{total_prompt_tokens:,}',
                'out': f'{total_completion_tokens:,}',
            })
            continue
        if out_path_lock.exists():
            tqdm.write(f"Skipping {out_path_lock} because it already exists")
            skipped_count += 1
            continue
        if failed_path.exists():
            tqdm.write(f"Skipping {json_path} because it previously failed to complete")
            skipped_count += 1
            continue
        # create lock file
        cur_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(out_path_lock, 'w') as f:
            f.write(cur_date)

        create_with_retry = with_timeout_retry(client.chat.completions.create, max_retries=args.max_retries)
        response = create_with_retry(model=model_name, messages=item['messages'], **kwargs)

        if response is None:
            print(f"Error: Failed to get a response from the API after {args.max_retries} retries")
            failed_to_complete_count += 1
            # delete lock file
            out_path_lock.unlink(missing_ok=True)
            # create a new .failed file
            with open(out_path.with_suffix('.failed'), 'w') as f:
                f.write(f"Error: Failed to get a response from the API after {args.max_retries} retries")
            
            pbar.set_postfix({'fail': str(failed_to_complete_count)})
            continue
        
        item['completion'] = orjson.loads(response.model_dump_json())

        # Track tokens and cost
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            cost = calculate_cost(prompt_tokens, completion_tokens, args.model)
            total_cost += cost
        
        completed_count += 1
        
        # Update progress bar with cost info
        pbar.set_postfix({
            'cost': f'${total_cost:.2f}',
            'in': f'{total_prompt_tokens:,}',
            'out': f'{total_completion_tokens:,}',
            'fail': f'{failed_to_complete_count}',
        })

        # save to jsonl
        with open(out_path, 'wb') as f:
            f.write(orjson.dumps(item))
        
        # delete lock file
        out_path_lock.unlink(missing_ok=True)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"Model:              {args.model}")
    print(f"New completions:    {completed_count}")
    print(f"Existing (cached):  {skipped_count}")
    print(f"Total files:        {completed_count + skipped_count}")
    print(f"Prompt tokens:      {total_prompt_tokens:,}")
    print(f"Completion tokens:  {total_completion_tokens:,}")
    print(f"Total tokens:       {total_prompt_tokens + total_completion_tokens:,}")
    input_price, output_price = get_model_pricing(args.model)
    print(f"Pricing (per 1M):   ${input_price:.2f} input / ${output_price:.2f} output")
    print(f"Total cost:         ${total_cost:.4f}")
    print("=" * 60)

async def create_completions_async(args, output_dir, input_dir, model_config: dict | None = None):
    """Async parallel version of create_completions using asyncio + semaphore."""
    config = select_client_config(args.model, model_config=model_config)
    client = openai.AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"], timeout=args.timeout)
    model_name = model_config["kwargs"]["model_name"] if model_config else args.model
    enable_thinking = args.thinking_mode == "enable"
    kwargs = get_completion_kwargs(args.model, enable_thinking, model_config=model_config)

    if output_dir.exists():
        if args.delete_existing_output_dir:
            shutil.rmtree(output_dir)
            print(f"Deleted existing output directory {output_dir}")
        else:
            print(f"Output directory {output_dir} already exists. Use --delete-existing-output-dir to delete it.")

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob("*.json"))
    input_files.sort()
    print(f"Found {len(input_files)} input files in {input_dir}")

    # Shared counters (safe because we're single-threaded with asyncio)
    stats = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cost": 0.0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
    }

    # Filter to only files that need processing
    to_process = []
    for json_path in input_files:
        with open(json_path, 'rb') as f:
            item = orjson.loads(f.read())
        task_num = item['task_num']
        out_fname = f"task_{task_num}.step_{item['step_num']}.json"
        out_path = output_dir / out_fname
        failed_path = out_path.with_suffix('.failed')

        if out_path.exists():
            try:
                with open(out_path, 'rb') as f:
                    existing = orjson.loads(f.read())
                if 'completion' in existing and 'usage' in existing['completion']:
                    usage = existing['completion']['usage']
                    stats["total_prompt_tokens"] += usage.get('prompt_tokens', 0) or 0
                    stats["total_completion_tokens"] += usage.get('completion_tokens', 0) or 0
                    stats["total_cost"] += calculate_cost(
                        usage.get('prompt_tokens', 0) or 0,
                        usage.get('completion_tokens', 0) or 0,
                        args.model
                    )
            except Exception:
                pass
            stats["skipped"] += 1
            continue
        if failed_path.exists():
            stats["skipped"] += 1
            continue
        to_process.append((json_path, item, out_path))

    print(f"Skipped {stats['skipped']} already completed/failed. Processing {len(to_process)} remaining.")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    pbar = tqdm(total=len(to_process), desc="Generating intents")

    async def process_one(json_path, item, out_path):
        async with semaphore:
            try:
                response = await async_with_timeout_retry(
                    lambda: client.chat.completions.create(
                        model=model_name, messages=item['messages'], **kwargs
                    ),
                    max_retries=args.max_retries,
                )
            except Exception as e:
                tqdm.write(f"Error processing {json_path.name}: {e}")
                response = None

            if response is None:
                stats["failed"] += 1
                with open(out_path.with_suffix('.failed'), 'w') as f:
                    f.write(f"Error: Failed after {args.max_retries} retries")
                pbar.update(1)
                pbar.set_postfix(cost=f'${stats["total_cost"]:.2f}', ok=stats["completed"], fail=stats["failed"])
                return

            item['completion'] = orjson.loads(response.model_dump_json())

            if response.usage:
                pt = response.usage.prompt_tokens or 0
                ct = response.usage.completion_tokens or 0
                stats["total_prompt_tokens"] += pt
                stats["total_completion_tokens"] += ct
                stats["total_cost"] += calculate_cost(pt, ct, args.model)

            stats["completed"] += 1

            with open(out_path, 'wb') as f:
                f.write(orjson.dumps(item))

            pbar.update(1)
            pbar.set_postfix(
                cost=f'${stats["total_cost"]:.2f}',
                in_tok=f'{stats["total_prompt_tokens"]:,}',
                out_tok=f'{stats["total_completion_tokens"]:,}',
                fail=stats["failed"],
            )

    tasks = [process_one(jp, item, op) for jp, item, op in to_process]
    await asyncio.gather(*tasks)

    pbar.close()

    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"Model:              {args.model}")
    print(f"New completions:    {stats['completed']}")
    print(f"Existing (cached):  {stats['skipped']}")
    print(f"Failed:             {stats['failed']}")
    print(f"Total files:        {stats['completed'] + stats['skipped']}")
    print(f"Prompt tokens:      {stats['total_prompt_tokens']:,}")
    print(f"Completion tokens:  {stats['total_completion_tokens']:,}")
    print(f"Total tokens:       {stats['total_prompt_tokens'] + stats['total_completion_tokens']:,}")
    input_price, output_price = get_model_pricing(args.model)
    print(f"Pricing (per 1M):   ${input_price:.2f} input / ${output_price:.2f} output")
    print(f"Total cost:         ${stats['total_cost']:.4f}")
    print("=" * 60)


def archive_mismatches(output_dir: Path, input_dir: Path):
    # go through all the json files in the output directory, if the corresponding
    # input file does not exist, move the json file to a separate directory
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist")
        exit(1)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        exit(1)
    
    # we should get all the json files in the output directory
    # and check if the corresponding input file exists
    
    archive_dir = output_dir.parent / "archived_mismatches" / output_dir.name
    archive_dir.mkdir(parents=True, exist_ok=True)
    input_file_names = set(f.name for f in input_dir.glob("**/*.json"))
    output_files = list(output_dir.glob("**/*.json"))
    files_to_not_move = []
    files_to_move = []
    for out_file in output_files:
        if out_file.name not in input_file_names:
            files_to_move.append({
                "from": out_file,
                "to": archive_dir / out_file.name
            })
        else:
            files_to_not_move.append(out_file)

    # prompt user to confirm
    ans = input(f"Found {len(files_to_move)} files to move to {archive_dir}. Moreover, {len(files_to_not_move)} files will not be moved. Continue? (y/n) ")
    ans = ans.strip().lower()
    if ans != 'y':
        print("Aborting")
        return
    
    # Move all files to the archive directory
    for file in files_to_move:
        shutil.move(str(file["from"]), str(file["to"]))
        print(f"Moved {file['from']} to {file['to']}")
    print(f"Moved {len(files_to_move)} files to {archive_dir}")
    return

if __name__ == "__main__":
    this_file = Path(__file__)
    model_configs_path = this_file.parent.parent / "configs" / "model_configs.json"
    shorthands_to_configs = load_all_model_configs(model_configs_path)

    parser = argparse.ArgumentParser(
        description="Generate task intents from prepared prompts using OpenAI API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir-base",
        type=str,
        help="Path to base input directory for prepared prompts. The final input directory will be <input-dir-base>/<exploration-model>",
        default="outputs/task_intents/prompts"
    )
    parser.add_argument(
        "-o",
        "--output-dir-base",
        type=str,
        help="Path to base output directory for generated task intents. The final output directory will be <output-dir-base>/<exploration-model>/<model>/",
        default="outputs/task_intents/completions"
    )
    parser.add_argument(
        '-e',
        '--exploration-model',
        type=str,
        default="qwen3-vl-32b-thinking",
        choices=list(shorthands_to_configs.keys()),
        help="Model used to generate exploration tasks, which we will use to group the task intents",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen3-vl-32b-thinking",
        choices=list(shorthands_to_configs.keys()),
        help="Model to use for generating task intents",
    )
    parser.add_argument(
        "-t",
        "--thinking_mode",
        type=str,
        choices=["enable", "disable"],
        default="enable",
        help="Enable or disable thinking mode",
    )
    parser.add_argument(
        "--delete-existing-output-dir",
        action="store_true",
        help="Delete existing output directory if it exists"
    )
    parser.add_argument(
        "--only-remove-locks",
        action="store_true",
        help="Only remove lock files and skip the rest"
    )
    parser.add_argument(
        "--only-remove-failures",
        action="store_true",
        help="Only remove failure files and skip the rest"
    )
    parser.add_argument(
        "--only-archive-mismatches",
        action="store_true",
        help="Only archive mismatched files into a separate directory and skip the rest"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout for API requests"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum number of retries for API requests"
    )
    parser.add_argument(
        "-n",
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum number of concurrent async API requests (0 = sequential/sync mode)"
    )
    args = parser.parse_args()
    model_name = shorthands_to_configs[args.model]["kwargs"]["model_name"]
    exploration_model_name = shorthands_to_configs[args.exploration_model]["kwargs"]["model_name"]

    exploration_model_dirname = exploration_model_name.replace('/', '_')
    task_generation_model_dirname = model_name.replace('/', '_')

    print('-' * 60)
    print(f"Model used for task generation: {args.model}")
    print(f"Model used for exploration: {args.exploration_model}")
    print(f"Output directory base: {args.output_dir_base}")
    print(f"Exploration model directory: {exploration_model_dirname}")
    print(f"Task generation model directory: {task_generation_model_dirname}")
    print('-' * 60)
    # output_dir is specific to the task generation model, whereas input_dir is shared by all task generation models 
    # for the same exploration model, so we have to specify the task generation model directory in the output directory
    # but not in the input directory
    output_dir = Path(args.output_dir_base, exploration_model_dirname, task_generation_model_dirname)
    input_dir = Path(args.input_dir_base, exploration_model_dirname)

    if args.only_remove_failures:
        failure_files = list(output_dir.glob("**/*.failed"))
        for failure_file in failure_files:
            failure_file.unlink()
        print(f"Cleaned {len(failure_files)} failure files")
        exit(0)
    
    if args.only_remove_locks:
        lock_files = list(output_dir.glob("**/*.lock"))
        for lock_file in lock_files:
            lock_file.unlink()
        print(f"Cleaned {len(lock_files)} lock files")
        exit(0)
    
    if args.only_archive_mismatches:
        archive_mismatches(output_dir=output_dir, input_dir=input_dir)
        exit(0)

    model_config = shorthands_to_configs.get(args.model)
    if args.max_concurrent > 0:
        asyncio.run(create_completions_async(args=args, output_dir=output_dir, input_dir=input_dir, model_config=model_config))
    else:
        create_completions(args=args, output_dir=output_dir, input_dir=input_dir, model_config=model_config)