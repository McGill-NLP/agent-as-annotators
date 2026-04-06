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

def infer_vision_args(model_name: str, shorthands_to_configs: dict):
    """Infer whether to use vision args based on model config."""
    if model_name not in shorthands_to_configs:
        raise ValueError(f"Unknown model: {model_name}")
    return shorthands_to_configs[model_name]["kwargs"].get("use_vision", False)
    
def get_a3_synth_benchmark(
    reset_instance=True, config_name="a3_synth.tasks.json", config_anchor="agent_as_annotators.configs.a3_synth", llm_judge=None, use_screenshot=True, use_axtree=True
):
    os.environ["WEBSYNTH_CONFIG_NAME"] = config_name
    os.environ["WEBSYNTH_CONFIG_ANCHOR"] = config_anchor
    os.environ["WEBSYNTH_LLM_JUDGE_USE_SCREENSHOT"] = str(use_screenshot)
    os.environ["WEBSYNTH_LLM_JUDGE_USE_AXTREE"] = str(use_axtree)
    
    if llm_judge is not None:
        os.environ["WEBSYNTH_LLM_JUDGE_MODEL"] = llm_judge
    else:
        raise ValueError("llm_judge must be set")

    from agent_as_annotators.benchmarks.a3_synth import (
        get_task_ids,
        WebSynthBenchmark,
        get_a3_synth_config_path,
        make_env_args_list_from_fixed_seeds_parallel,
    )
    config_path = get_a3_synth_config_path()
    task_ids = get_task_ids(config_path)
    logging.info(f'Using {len(task_ids)} tasks from config file "{config_path}"')

    if not config_name.startswith("a3_synth."):
        benchmark_name = f"a3_synth_{config_name.split('.')[0]}"
    else:
        benchmark_name = config_name

    b = WebSynthBenchmark(
        name=benchmark_name,
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=[],
        env_args_list=make_env_args_list_from_fixed_seeds_parallel(
            task_list=[f"a3_synth.{i}" for i in task_ids],
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=None,
        reset_instance=reset_instance,
    )

    return b



def main():
    # from agentlab.experiments.study import Study
    from agent_as_annotators.benchmarks.a3_synth.study import WebSynthStudy
    import agent_as_annotators.utils as lau
    import agent_as_annotators.modeling as lam
    
    this_file = Path(__file__)
    model_configs_path = this_file.parent / "configs" / "model_configs.json"
    shorthands_to_configs = lau.load_all_model_configs(model_configs_path)
    
    logging.getLogger().setLevel(logging.INFO)
    
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
            "a3_synth",
        ],
        default="a3_synth",
        help="Select the benchmark to run on",
    )
    
    parser.add_argument(
        "-m",
        "--models",
        choices=list(shorthands_to_configs.keys()),
        default=["qwen3-vl-32b-thinking"],
        help="Select the model to use (at least one)",
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        default=2,
        help="Number of parallel jobs to run",
    )
    parser.add_argument(
        "--parallel",
        choices=["ray", "joblib", "sequential"],
        default="ray",
        help="Select the parallel backend to use",
    )
    parser.add_argument(
        "--use-axtree",
        choices=["True", "False"],
        default="True",
        help="Use the axtree for the llm judge",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="a3_synth.tasks.json",
        help="Select the config name to use for the A3-Synth benchmark",
    )
    parser.add_argument(
        "--disable-reset",
        action="store_true",
        help="Disable the instance reset for the A3-Synth benchmark",
    )
    parser.add_argument(
        "--config-anchor",
        type=str,
        default="agent_as_annotators.configs.a3_synth",
        help="Select the config anchor to use for the A3-Synth benchmark",
    )
    parser.add_argument(
        "--llm-judge-model",
        type=str,
        choices=[
            "same-as-models",
            *[model for model in shorthands_to_configs.keys()],
        ],
        default="same-as-models",
        help="Select the model to use for the llm judge (if more than one model is selected, only the first one will be used)",
    )
    
    args = parser.parse_args()
    # 1. select benchmark:
    if args.benchmark == "a3_synth":
        if args.llm_judge_model == "same-as-models":
            llm_judge = args.models[0]
            print(f"Using {llm_judge} as the llm judge")
        else:
            llm_judge = args.llm_judge_model
        benchmark = get_a3_synth_benchmark(
            config_name=args.config_name, 
            config_anchor=args.config_anchor,
            reset_instance=not args.disable_reset, 
            llm_judge=shorthand_to_model(llm_judge), 
            use_screenshot=infer_vision_args(llm_judge, shorthands_to_configs), 
            use_axtree=args.use_axtree,
        )
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
        print("Relaunching study from directory containing:", args.relaunch)
        study = WebSynthStudy.load_most_recent(
            contains=args.relaunch, root_dir=Path(os.environ["AGENTLAB_EXP_ROOT"])
        )
        # study = Study.load_most_recent(
        #     contains=args.relaunch, root_dir=Path(os.environ["AGENTLAB_EXP_ROOT"])
        # )
        # Override the benchmark's reset_instance setting from the command line
        study.benchmark.reset_instance = not args.disable_reset
        study.find_incomplete(include_errors=True)
    else:
        study = WebSynthStudy(agent_args, benchmark, logging_level_stdout=logging.INFO)  # type: ignore
        # study = Study(agent_args, benchmark, logging_level_stdout=logging.INFO)  # type: ignore
    
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
