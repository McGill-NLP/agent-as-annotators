import argparse
import datetime
import itertools
import os
from pathlib import Path

try:
    import orjson
except ImportError:
    import json as orjson

import openai
from tqdm.auto import tqdm

from agent_reward_bench.judge import create_chat_messages_from_trajectory
from agent_reward_bench.judge.existing import aer, nnetnav
from agent_reward_bench.judge.args import default_judge_args, judge_args

from agent_reward_bench.utils import (
    get_api_key_from_env_var,
    get_base_url_from_env_var,
    CostEstimator,
)

def run_judge_on_single_trajectory(
    traj_path,
    traj_dir,
    save_dir,
    estimator,
    pbar: tqdm,
    skip_errors: bool = False,
):
    """
    Function to run the judge on a single trajectory.
    """
    save_dir = Path(save_dir)
    traj_path = Path(traj_path)
    
    save_path = save_dir.joinpath(f"{traj_path.stem}.json")
    error_path = save_dir.joinpath(f"{traj_path.stem}_error.json")
    
    # if the file already exists, skip
    if save_path.exists():
        tqdm.write(f"Skipping {traj_path.name} because the file already exists")
        return "exists"

    if skip_errors and error_path.exists():
        tqdm.write(f"Skipping {traj_path.name} because the error file already exists")
        return "exists"
    
    # with open(path, "r") as f:
    #     trajectory = json.load(f)
    with open(traj_path, "rb") as f:
        trajectory = orjson.loads(f.read())

    # if it's not valid, skip
    if not trajectory["valid"]:
        tqdm.write(f"Skipping {traj_path} because it's not valid")
        return None

    # show the trajectory id in the progress bar
    pbar.set_postfix(
        task_id=traj_path.stem.split(".")[-1], estimator=estimator.total_cost
    )

    if judge == "functional":
        results = {
            "benchmark": trajectory["benchmark"],
            "goal": trajectory["goal"],
            "agent": trajectory["agent"],
            "judge": judge,
            "judge_model_name": "judge",
            "provider": provider,
            "cost": 0,
            "trajectory_info": {
                "valid": trajectory["valid"],
                "model": trajectory["model"],
                "trajectory_dir": trajectory.get("trajectory_dir"),
                "seed": trajectory["seed"],
                "model_args": trajectory["model_args"],
                "flags": trajectory["flags"],
                "summary_info": trajectory["summary_info"],
            },
        }
        return results

    results = {
        "benchmark": trajectory["benchmark"],
        "goal": trajectory["goal"],
        "agent": trajectory["agent"],
        "trajectory_info": {
            "valid": trajectory["valid"],
            "model": trajectory["model"],
            "trajectory_dir": trajectory.get("trajectory_dir"),
            "seed": trajectory["seed"],
            "model_args": trajectory["model_args"],
            "flags": trajectory["flags"],
            "summary_info": trajectory["summary_info"],
        },
        "reward": trajectory["reward"],
    }

    breakpoint()

    return results

benchmarks = ["a3_synth"]
agents = [
    "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct",
]


parser = argparse.ArgumentParser(
    description="Run the judge on a set of trajectories",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--base_dir",
    type=str,
    default="trajectories/cleaned",
    help="The base directory where the trajectories are stored",
)
parser.add_argument(
    "--base_save_dir",
    type=str,
    default="trajectories/judgments",
    help="The base directory where the judgments will be saved",
)
parser.add_argument(
    "--agents",
    nargs="+",
    default=agents,
    help="Select the agent to use",
)
args = parser.parse_args()

def main():
    agents = args.agents
    base_dir = Path(args.base_dir)
    base_save_dir = Path(args.base_save_dir)
    
    cost_across_runs = 0
    for agent, benchmark in itertools.product(agents, benchmarks):
        # get all json files that in the path tree of the benchmark and model
        traj_dir = Path(base_dir, benchmark, agent)
        save_dir = Path(base_save_dir, benchmark, agent)

        trajectories_paths = list(traj_dir.glob("**/*.json"))

        print("\nAgent:", agent)
        print("Benchmark:", benchmark)

        pbar = tqdm(trajectories_paths, desc="Running judge")

        for traj_path in pbar:
            # if the lock file exists, skip
            save_path_lock = save_dir.joinpath(f"{traj_path.stem}.lock")
            if save_path_lock.exists():
                tqdm.write(f"Skipping {traj_path.name} because the lock file exists")
                continue
            
            # create a lock file to prevent concurrent writes
            with open(save_path_lock, "w") as f:
                dt_now = datetime.datetime.now().isoformat()
                f.write(dt_now)
            try:
                out = run_judge_on_single_trajectory(
                    traj_path=traj_path,
                    traj_dir=traj_dir,
                    save_dir=save_dir,
                    skip_errors=False,  # set to False to handle errors in the function
                )
            # except if the user interrupts the process, then we delete the lockfile
            except KeyboardInterrupt:
                tqdm.write(f"Interrupted while processing {traj_path.name}, deleting lock file")
                save_path_lock.unlink(missing_ok=True)
                raise

            if out == "exists":
                save_path_lock.unlink(missing_ok=True)
                continue
            
            if out is None:
                # unlock the lock file
                save_path_lock.unlink(missing_ok=True)
                tqdm.write(f"Skipping {traj_path.name} because it was not valid or an error occurred")
                continue
            
            results = out
            try:
                save_path = save_dir.joinpath(f"{traj_path.stem}.json")
                save_dir.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(orjson.dumps(results))

            except KeyboardInterrupt:
                tqdm.write(f"Interrupted while processing {traj_path.name}, deleting lock file")
                save_path_lock.unlink(missing_ok=True)
                raise
            
            # unlock the lock file
            save_path_lock.unlink(missing_ok=True)

    print(f"Total cost across runs: ${round(cost_across_runs, 6)}")

if __name__ == "__main__":
    main()