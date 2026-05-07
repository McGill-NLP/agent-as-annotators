"""
Given a directory of agentlab results (e.g. agentlab_results/2025-09-17_01-28-02_genericagent-qwen-qwen3-32b-on-exploration-exploration),
extract all the chat messages into json files matching each of the pickles. Save them into a new directory called chat_messages.
"""

import pickle
import gzip
import orjson
import argparse
import re
from pathlib import Path
from tqdm import tqdm


def find_latest_results_dir(base_dir, agent_name, benchmark="exploration"):
    """
    Find the latest results directory matching the agent name and benchmark.

    Directory naming pattern: YYYY-MM-DD_HH-MM-SS_genericagent-{agent}-on-{benchmark}-{suffix}
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory {base_dir} does not exist")
        return None

    # Pattern to match directories with timestamp and keywords
    # Looking for: on-{benchmark} and containing agent_name
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


def extract_chat_messages(results_dir, output_dir, force_reprocess=False):
    """Extract chat messages from all pickle files in the results directory"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist")
        return

    # Create chat_messages output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create subdirectory matching the results directory name
    output_subdir = output_dir / results_path.name
    output_subdir.mkdir(exist_ok=True)

    print(f"Extracting chat messages from {results_dir}")
    print(f"Output directory: {output_subdir}")

    # First pass: count all pickle files for progress bar
    all_pickle_files = []
    subdirs = [d for d in results_path.iterdir() if d.is_dir()]

    print("Scanning for pickle files...")
    for subdir in tqdm(subdirs, desc="Scanning directories"):
        step_files = list(subdir.glob("step_*.pkl.gz"))
        step_files.sort(key=lambda x: int(x.stem.split("_")[1].split(".")[0]))
        all_pickle_files.extend([(subdir, pickle_file) for pickle_file in step_files])

    print(f"Found {len(all_pickle_files)} pickle files to process")

    # Process each pickle file with progress bar
    processed_count = 0
    skipped_count = 0

    for subdir, pickle_file in tqdm(all_pickle_files, desc="Extracting messages"):
        # Create corresponding output directory
        sub_output_dir = output_subdir / subdir.name
        sub_output_dir.mkdir(exist_ok=True)

        # Check if output file already exists (caching)
        json_filename = pickle_file.stem.replace(".pkl", ".json")
        json_path = sub_output_dir / json_filename

        if json_path.exists() and not force_reprocess:
            # Check if output is newer than input (basic cache validation)
            if json_path.stat().st_mtime > pickle_file.stat().st_mtime:
                skipped_count += 1
                continue

        # Extract chat messages
        pickle_file_path = Path(pickle_file).resolve()
        try:
            with gzip.open(pickle_file_path, "rb") as f:
                data = pickle.load(f)

            # Extract chat messages from the obs
            if "chat_messages" not in data.agent_info:
                messages = []
                chat_model_args = {}
                goal = ""
            else:
                messages = data.agent_info["chat_messages"].to_openai()
                chat_model_args = data.agent_info.extra_info.get("chat_model_args", {})
                goal = data.obs.get("goal", "")

        except Exception as e:
            tqdm.write(f"Error processing {pickle_file_path}: {e}")
            continue

        pickle_file_path = Path(pickle_file).resolve()
        image_file_path = (
            str(pickle_file_path)
            .replace(".pkl.gz", ".png")
            .replace("step", "screenshot_step")
        )

        extracted = {
            "pickle_path": str(pickle_file_path),
            "image_path": str(image_file_path),
            "goal": goal,
            "chat_model_args": chat_model_args,
            "step_num": data.step,
            "messages": messages,
        }
        # Save to JSON file with matching name
        with open(json_path, "w") as f:
            f.write(orjson.dumps(extracted).decode())

        processed_count += 1

    print(f"Processed {processed_count} pickle files")
    print(f"Skipped {skipped_count} files (already up-to-date)")
    print(f"Chat messages saved to {output_subdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract chat messages from agentlab results"
    )
    parser.add_argument(
        "--path",
        "-p",
        default="",
        help="Path to the agentlab results directory",
    )
    parser.add_argument(
        "--find-latest",
        "-l",
        metavar="AGENT",
        help="Find the latest results directory for the specified agent name (e.g., 'gemini-3-flash-preview')",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        default="exploration",
        help="Benchmark name to match when using --find-latest (default: exploration)",
    )
    parser.add_argument(
        "--base-dir",
        default="agentlab_results",
        help="Base directory to search in when using --find-latest (default: agentlab_results)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reprocessing of all files, ignoring cache",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="outputs/chat_messages",
        help="Directory to save extracted chat messages",
    )

    args = parser.parse_args()

    # Determine results directory
    if args.find_latest:
        results_dir = find_latest_results_dir(args.base_dir, args.find_latest, args.benchmark)
        if results_dir is None:
            return
    elif args.path:
        results_dir = args.path
    else:
        parser.error("Either --path or --find-latest must be specified")

    extract_chat_messages(results_dir, args.output_dir, force_reprocess=args.force)


if __name__ == "__main__":
    main()
