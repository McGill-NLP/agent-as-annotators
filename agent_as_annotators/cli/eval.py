import argparse
import json
import os
import logging
import tempfile
from pathlib import Path

import agent_as_annotators.browser_stealth  # noqa: F401 — anti-detection patches for headless browser
import agent_as_annotators.obs_timeout  # noqa: F401 — timeout for screenshot extraction (prevents 39-min hangs)

# Isolate Ray clusters so concurrent experiments don't share GCS.
# Without this, ray.shutdown() in one experiment can kill another's GCS.
import ray
_original_ray_init = ray.init
def _isolated_ray_init(*args, **kwargs):
    if "_temp_dir" not in kwargs:
        # Use /tmp/ray_scratch (symlink to scratch) to keep paths short for AF_UNIX sockets (107 byte limit)
        ray_tmpdir = "/tmp/ray_scratch" if os.path.isdir("/tmp/ray_scratch") else tempfile.gettempdir()
        kwargs["_temp_dir"] = tempfile.mkdtemp(prefix="ray_wa_", dir=ray_tmpdir)
    return _original_ray_init(*args, **kwargs)
ray.init = _isolated_ray_init

def get_visualwebarena_benchmark(split="train"):
    allowed_splits = ["train", "valid", "test", "full"]
    if split not in allowed_splits:
        raise ValueError(f"split must be one of {allowed_splits}, got {split}")

    from browsergym.experiments.benchmark.base import Benchmark
    from browsergym.experiments.benchmark.configs import (
        DEFAULT_HIGHLEVEL_ACTION_SET_ARGS,
    )
    from browsergym.experiments.benchmark.utils import (
        make_env_args_list_from_fixed_seeds,
    )
    from browsergym.experiments.benchmark.metadata.utils import (
        task_metadata,
        task_list_from_metadata,
    )
    tm = task_metadata("visualwebarena")

    bcnk = Benchmark(
        name=f"visualwebarena_{split}",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["visualwebarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["visualwebarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=task_list_from_metadata(tm),
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=tm,
    )

    b_split = bcnk if split == "full" else bcnk.subset_from_split(split)

    return b_split

def get_webarena_benchmark(split="train"):
    allowed_splits = ["train", "valid", "test", "full"]
    if split not in allowed_splits:
        raise ValueError(f"split must be one of {allowed_splits}, got {split}")

    from browsergym.experiments.benchmark.base import Benchmark
    from browsergym.experiments.benchmark.configs import (
        DEFAULT_HIGHLEVEL_ACTION_SET_ARGS,
    )
    from browsergym.experiments.benchmark.utils import (
        make_env_args_list_from_fixed_seeds,
    )
    from browsergym.experiments.benchmark.metadata.utils import (
        task_metadata,
        task_list_from_metadata,
    )

    tm = task_metadata("webarena")

    bcnk = Benchmark(
        name=f"webarena_{split}",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=False,
        backends=["webarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=task_list_from_metadata(tm),
            max_steps=30,
            fixed_seeds=[0],
        ),
        task_metadata=tm,
    )

    b_split = bcnk if split == "full" else bcnk.subset_from_split(split)

    return b_split

def get_assistantbench_benchmark(split="valid"):
    allowed_splits = ["train", "valid", "test", "full"]
    if split not in allowed_splits:
        raise ValueError(f"split must be one of {allowed_splits}, got {split}")
    from browsergym.experiments.benchmark.configs import DEFAULT_BENCHMARKS
    bcnk = DEFAULT_BENCHMARKS["assistantbench"]()
    return bcnk if split == "full" else bcnk.subset_from_split(split)

def get_miniwob_benchmark(split="full"):
    allowed_splits = ["train", "test", "full"]
    if split not in allowed_splits:
        raise ValueError(f"split must be one of {allowed_splits}, got {split}")
    from browsergym.experiments.benchmark.configs import DEFAULT_BENCHMARKS
    bcnk = DEFAULT_BENCHMARKS["miniwob"]()
    return bcnk if split == "full" else bcnk.subset_from_split(split)

def get_workarena_benchmark(split="test", level="l1"):
    allowed_splits = ["train", "valid", "test", "full"]
    allowed_levels = ["l1", "l2"]
    if split not in allowed_splits:
        raise ValueError(f"split must be one of {allowed_splits}, got {split}")
    if level not in allowed_levels:
        raise ValueError(f"level must be one of {allowed_levels}, got {level}")

    from browsergym.experiments.benchmark.base import Benchmark
    from browsergym.experiments.benchmark.configs import (
        DEFAULT_HIGHLEVEL_ACTION_SET_ARGS,
    )
    from browsergym.experiments.benchmark.utils import (
        make_env_args_list_from_fixed_seeds,
    )
    from browsergym.experiments.benchmark.metadata.utils import (
        task_metadata,
        task_list_from_metadata,
    )

    tm = task_metadata("workarena")

    if level == "l1":
        action_set_key = "workarena"
        is_multi_tab = False
        max_steps = 15
    else:
        action_set_key = "workarena++"
        is_multi_tab = True
        max_steps = 50

    bcnk = Benchmark(
        name=f"workarena_{level}_{split}",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS[action_set_key],
        is_multi_tab=is_multi_tab,
        supports_parallel_seeds=False,
        backends=["workarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=task_list_from_metadata(tm, filter={"level": f"^{level}$"}),
            max_steps=max_steps,
            fixed_seeds=list(range(10)) if level == "l1" else [0],
        ),
        task_metadata=tm,
    )

    b_split = bcnk if split == "full" else bcnk.subset_from_split(split)

    return b_split

def main():
    from agentlab.experiments.study import Study
    
    import agent_as_annotators.modeling as lam
    import agent_as_annotators.utils
    
    logging.getLogger().setLevel(logging.INFO)
    
    this_file = Path(__file__)
    model_configs_path = this_file.parent / "configs" / "model_configs.json"
    shorthands_to_configs = agent_as_annotators.utils.load_all_model_configs(model_configs_path)
    
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
            "webarena_test",
            "webarena_train",
            "visualwebarena_test",
            "visualwebarena_train",
            "workarena_l1",
            "workarena_l2_test",
            "assistantbench",
            "assistantbench_valid",
            "miniwob",
            "miniwob_test",
            "miniwob_train",
        ],
        default="webarena_test",
        help="Select the benchmark to run on",
    )
    
    parser.add_argument(
        "-m",
        "--models",
        choices=list(shorthands_to_configs.keys()),
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
        "--n-relaunch",
        type=int,
        default=10,
        help="Maximum number of relaunches for failed tasks",
    )
    
    args = parser.parse_args()
    # 1. select benchmark:
    if args.benchmark == "webarena_test":
        benchmark = get_webarena_benchmark(split="test")
    elif args.benchmark == "webarena_train":
        benchmark = get_webarena_benchmark(split="train")
    elif args.benchmark == "visualwebarena_test":
        benchmark = get_visualwebarena_benchmark(split="test")
    elif args.benchmark == "visualwebarena_train":
        benchmark = get_visualwebarena_benchmark(split="train")
    elif args.benchmark == "workarena_l1":
        benchmark = get_workarena_benchmark(split="full", level="l1")
    elif args.benchmark == "workarena_l2_test":
        benchmark = get_workarena_benchmark(split="test", level="l2")
    elif args.benchmark == "assistantbench":
        benchmark = get_assistantbench_benchmark(split="full")
    elif args.benchmark == "assistantbench_valid":
        benchmark = get_assistantbench_benchmark(split="valid")
    elif args.benchmark == "miniwob":
        benchmark = get_miniwob_benchmark(split="full")
    elif args.benchmark == "miniwob_test":
        benchmark = get_miniwob_benchmark(split="test")
    elif args.benchmark == "miniwob_train":
        benchmark = get_miniwob_benchmark(split="train")
    else:
        raise ValueError(f"Invalid benchmark: {args.benchmark}")
    
    # 2. Select model
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
    
    # 2.5. AssistantBench: instruct agent to avoid Google (CAPTCHAs block headless browsers)
    # Note: start_url is patched directly in browsergym/assistantbench/task.py to DuckDuckGo
    if args.benchmark in ("assistantbench", "assistantbench_valid"):
        for a in agent_args:
            a.flags.extra_instructions = (
                "IMPORTANT: Do NOT use Google Search — it will block you with a CAPTCHA. "
                "Use https://duckduckgo.com for all web searches."
            )
    
    # 3. Set up study
    # reproducibility_mode = False if any("gpt-5" in model for model in args.models) else True
    reproducibility_mode = False
    strict_reproducibility = False
    
    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]
    
    if args.relaunch is not None:
        print("Relaunching study from directory containing:", args.relaunch)
        study = Study.load_most_recent(
            contains=args.relaunch, root_dir=Path(os.environ["AGENTLAB_EXP_ROOT"])
        )
        study.find_incomplete(include_errors=True)
    else:
        study = Study(agent_args, benchmark, logging_level_stdout=logging.INFO)  # type: ignore
    
    # Increase step timeout (default 60s is too tight for slower models)
    if "workarena" in args.benchmark:
        study.avg_step_timeout = 300
    elif args.benchmark in ("assistantbench", "assistantbench_valid"):
        study.avg_step_timeout = 120
    
    # 4. Run study
    # Write a .preparing marker so status tools can detect massage/reset phase
    preparing_marker = None
    if study.dir is not None:
        preparing_marker = Path(study.dir) / ".preparing"
        preparing_marker.write_text(f"started: {__import__('datetime').datetime.now().isoformat()}\n")
    
    try:
        study.run(
            n_jobs=args.n_jobs,
            parallel_backend=args.parallel,
            strict_reproducibility=strict_reproducibility,
            n_relaunch=args.n_relaunch,
        )
    finally:
        # Remove preparing marker when done (whether success or failure)
        if preparing_marker is not None:
            preparing_marker.unlink(missing_ok=True)
        elif study.dir is not None:
            # Study dir was created during run — clean up marker if it was written
            Path(study.dir, ".preparing").unlink(missing_ok=True)
    
    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=strict_reproducibility)


if __name__ == "__main__":
    main()
