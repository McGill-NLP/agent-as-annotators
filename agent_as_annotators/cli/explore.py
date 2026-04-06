import argparse
import os
import logging
from pathlib import Path

from browsergym.experiments.benchmark.configs import (
    DEFAULT_HIGHLEVEL_ACTION_SET_ARGS,
)

from agent_as_annotators.utils import shorthand_to_model

# Ensure AGENTLAB_EXP_ROOT is absolute (required for ray workers which run in a different cwd)
if "AGENTLAB_EXP_ROOT" in os.environ:
    os.environ["AGENTLAB_EXP_ROOT"] = str(Path(os.environ["AGENTLAB_EXP_ROOT"]).resolve())

def get_exploration_benchmark(
    reset_instance=True, config_name="exploration.tasks.json", model_name="Qwen/Qwen3-32B"
):
    os.environ["EXPLORATION_CONFIG_NAME"] = config_name
    os.environ["EXPLORATION_MODEL_NAME"] = model_name.replace("/", "_")
    from agent_as_annotators.exploration import (
        get_task_ids,
        ExplorationBenchmark,
        CONFIG_PATH,
        make_env_args_list_from_fixed_seeds_parallel,
    )

    task_ids = get_task_ids(CONFIG_PATH)
    logging.info(f'Using {len(task_ids)} tasks from config file "{CONFIG_PATH}"')

    benchmark_name = f"exploration_{model_name.replace('/', '_')}"

    b = ExplorationBenchmark(
        name=benchmark_name,
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=[],
        env_args_list=make_env_args_list_from_fixed_seeds_parallel(
            task_list=[f"exploration.{i}" for i in task_ids],
            max_steps=20,
            fixed_seeds=[0],
        ),
        task_metadata=None,
        reset_instance=reset_instance,
    )

    return b


logging.getLogger().setLevel(logging.INFO)

def main():
    from agentlab.experiments.study import Study
    import agent_as_annotators.utils as lau
    import agent_as_annotators.modeling as lam
    
    this_file = Path(__file__)
    model_configs_path = this_file.parent / "configs" / "model_configs.json"
    shorthands_to_configs = lau.load_all_model_configs(model_configs_path)
    
    parser = argparse.ArgumentParser(
        description="Run a generic agent on a benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "-r",
        "--relaunch",
        type=str,
        help="Relaunch an existing study with a string that matches the study name",
    )
    
    parser.add_argument(
        "-b",
        "--benchmark",
        choices=[
            "exploration",
        ],
        default="exploration",
        help="Select the benchmark to run on",
    )
    
    parser.add_argument(
        "-m",
        "--models",
        choices=[
            *[model for model in shorthands_to_configs.keys()],
        ],
        default=["qwen3-vl-32b-thinking"],
        help="Select the model to use",
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        default=4,
        help="Number of parallel jobs to run",
    )
    parser.add_argument(
        "--parallel",
        choices=["ray", "joblib", "sequential"],
        default="ray",
        help="Select the parallel backend to use",
    )
    
    parser.add_argument(
        "--config-name",
        type=str,
        default="exploration.tasks.json",
        help="Select the config name to use for the A3-Synth benchmark",
    )
    parser.add_argument(
        "--disable-reset",
        action="store_true",
        help="Disable the instance reset for the exploration benchmark",
    )
    
    args = parser.parse_args()
    model_name = args.models[0]
    
    # 1. select benchmark:
    if args.benchmark == "exploration":
        benchmark = get_exploration_benchmark(config_name=args.config_name, model_name=shorthand_to_model(model_name), reset_instance=not args.disable_reset)
    else:
        raise ValueError(f"Invalid benchmark: {args.benchmark}")
    
    # 2. Select model
    # since args.models is a list now, we'll make a list of agent_args
    agent_args = []
    
    for model in args.models:
        if model not in shorthands_to_configs:
            raise ValueError(f"Unknown model: {model}")
    
        model_config = shorthands_to_configs[model]
    
        if model_config["hosting"] == "vllm":
            # Auto-infer base_url from port if VLLM_BASE_URL is not set
            base_url = os.getenv("VLLM_BASE_URL")
            if base_url is None and "port" in model_config:
                base_url = f"http://localhost:{model_config['port']}/v1"
                print("inferred base_url from model_configs.json")
            else:
                print("Using VLLM server from environment variable")
            print("Using VLLM server at: ", base_url)
            
            agent = lam.prepare_vllm_model(**model_config["kwargs"], base_url=base_url)
        elif model_config["hosting"] == "gemini":
            agent = lam.prepare_gemini(**model_config["kwargs"])
        elif model_config["hosting"] == "openai":
            agent = lam.prepare_gpt(**model_config["kwargs"])
        else:
            raise ValueError(f"Unknown hosting: {model_config['hosting']}")
    
        agent_args.append(agent)
    
    # 3. Set up study
    n_relaunch = 10
    parallel_backend = args.parallel
    n_jobs = args.n_jobs
    
    reproducibility_mode = True
    strict_reproducibility = False
    
    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]
    
    if args.relaunch is not None:
        # if it's a path, we take the name of the last directory
        if Path(args.relaunch).is_dir():
            args.relaunch = str(Path(args.relaunch).name)
        print("Relaunching study from directory containing:", args.relaunch)
        study = Study.load_most_recent(
            contains=args.relaunch, root_dir=Path(os.environ["AGENTLAB_EXP_ROOT"])
        )
        # Override the benchmark's reset_instance setting from the command line
        study.benchmark.reset_instance = not args.disable_reset
        study.find_incomplete(include_errors=True)
    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.INFO)  # type: ignore
    
    # 4. Run study
    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=strict_reproducibility,
        n_relaunch=n_relaunch,
    )
    
    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=strict_reproducibility)


if __name__ == "__main__":
    main()
