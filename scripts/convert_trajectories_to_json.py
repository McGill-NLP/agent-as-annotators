"""
The goal of this script is to convert the trajectories in the agentlab_results from pickles for each step into a single json file for each trajectory.
"""

import argparse
import orjson
from pathlib import Path
import os
import shutil
from tqdm.auto import tqdm


# from web_agents_sft.modeling import VllmModelArgs  # needed for Step.from_mini_dict
from llm_annotators.utils.trajectories import (
    TrajectoriesManager,
    list_experiments,
    Step,
)


def copy_screenshots(data, json_path, base_save_dir: Path):
    json_file_stem = str(Path(json_path).stem)
    # Get the benchmark, agent, and judge from the json file
    benchmark = data["benchmark"]
    agent = data["agent"]
    # for each step, copy the screenshot to the corresponding folder
    for step in data["steps"]:
        # Get the screenshot path
        screenshot_path = step["screenshot_path"]
        screenshot_fname = os.path.basename(screenshot_path)
        # Get the new screenshot path
        new_screenshot_path = os.path.join(
            base_save_dir,
            "screenshots",
            benchmark,
            agent,
            json_file_stem,
            screenshot_fname,
        )
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(new_screenshot_path), exist_ok=True)
        # Copy the screenshot to the new location, only if it doesn't exist
        if not os.path.exists(new_screenshot_path):
            shutil.copy2(screenshot_path, new_screenshot_path)

        # Update the json file to point to the new location
        step["screenshot_path"] = new_screenshot_path


def remove_keys(data, keys_to_remove):
    for key in keys_to_remove:
        if key in data:
            del data[key]


def main(
    agentlab_results_dir: Path,
    base_trajectories_dir: Path,
    base_json_save_dir: Path,
    add_suffix: bool,
):
    trajectories_manager = TrajectoriesManager()
    experiments = list_experiments(base_dir=agentlab_results_dir)
    trajectories_manager.add_trajectories_from_dirs(experiments)
    trajectories_manager.build_index()

    for benchmark in tqdm(trajectories_manager.get_benchmarks(), desc="Benchmarks"):
        for model in tqdm(
            trajectories_manager.get_model_names(benchmark), desc="Models", leave=False
        ):
            exps = trajectories_manager.get_exp_names(benchmark, model)
            # if there's more than one experiment, or less than one, we have a problem
            assert len(exps) == 1, (
                f"Expected 1 experiment for {benchmark} {model}, got {len(exps)}"
            )

            exp = exps[0]
            trajectories = trajectories_manager.get_trajectories(benchmark, model, exp)

            for traj in tqdm(trajectories, desc="Trajectories", leave=False):
                traj_save_dir: Path = base_json_save_dir / benchmark / model / exp
                traj_save_dir.mkdir(parents=True, exist_ok=True)
                task_id = traj.task_id
                if add_suffix:
                    # add the suffix from the traj.exp_dir
                    suffix = traj.exp_dir.name.split(benchmark)[1]
                    task_id = f"{task_id}{suffix}"

                save_path = traj_save_dir / f"{task_id}.json"

                # if the save path already exists, skip
                if save_path.exists():
                    continue

                traj_dict = traj.to_dict()
                traj_dict["steps"] = [
                    Step.from_mini_dict(s).to_dict(prune_axtree=True) for s in traj
                ]
                copy_screenshots(
                    traj_dict, save_path, base_save_dir=base_trajectories_dir
                )
                remove_keys(traj_dict, ["logs"])

                with open(save_path, "wb") as fb:
                    # use orjson for faster serialization
                    fb.write(orjson.dumps(traj_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert trajectories to json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agentlab_results_dir",
        type=str,
        default="agentlab_results/websynth",
        help="The base directory where the agentlab results are stored",
    )
    parser.add_argument(
        "--base_trajectories_dir",
        type=str,
        default="trajectories/cleaned",
        help="The base directory where the trajectories will be saved",
    )
    parser.add_argument(
        "--base_json_save_dir",
        type=str,
        default="trajectories/cleaned",
        help="The base directory where the json files will be saved",
    )
    parser.add_argument(
        "--add_suffix",
        action="store_true",
        help="Add the suffix to the task id",
    )

    args = parser.parse_args()
    agentlab_results_dir = Path(args.agentlab_results_dir)
    base_trajectories_dir = Path(args.base_trajectories_dir)
    base_json_save_dir = Path(args.base_json_save_dir)

    main(
        agentlab_results_dir=agentlab_results_dir,
        base_trajectories_dir=base_trajectories_dir,
        base_json_save_dir=base_json_save_dir,
        add_suffix=args.add_suffix,
    )
