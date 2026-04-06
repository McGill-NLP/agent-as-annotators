import logging
from pathlib import Path
from functools import partial
from copy import deepcopy
import time
from typing import Any, Type, Union

import gymnasium as gym
import nltk
from browsergym.core.task import AbstractBrowserTask
from browsergym.core.registration import frozen_partial

from . import GenericWebSynthTask, get_task_ids, get_a3_synth_config_path
from .env import BrowserEnvWebSynth
from agent_as_annotators.utils.a3_synth import print_last_step_output, LastStepOutputType
logger = logging.getLogger(__name__)


# download necessary tokenizer resources
# note: deprecated punkt -> punkt_tab https://github.com/nltk/nltk/issues/3293
try:
    nltk.data.find("tokenizers/punkt_tab")
except Exception as e:
    print(e, 'failed to find punkt_tab. downloading...')
    nltk.download("punkt_tab", quiet=True, raise_on_error=True)

def register_task_a3_synth(
    id: str,
    task_class: Type[AbstractBrowserTask],
    task_kwargs: dict = {},
    default_task_kwargs: dict = {},
    nondeterministic: bool = True,
    *args,
    **kwargs,
):
    """
    Registers a browser task as a gym environment with its unique id.

    Args:
        id: the id of the task to register (will be prepended by "browsergym/").
        task_class: the task class to register.
        task_kwargs: frozen task arguments (can not be overloaded at environment creation time).
        task_kwargs_default: default task arguments (can be overloaded at environment creation time).
        nondeterministic: whether the task cannot be guaranteed deterministic transitions.
        *args: additional sequential arguments for either the gym or the browsergym environment.
        *kwargs: additional keyword arguments for either the gym or the browsergym environment.
    
    Copied from browsergym.core.registration.register_task with minor modifications to allow BrowserEnv
    to be customized.
    """
    
    if task_kwargs and default_task_kwargs:
        # check overlap between frozen and default task_kwargs
        clashing_kwargs = set(task_kwargs) & set(default_task_kwargs)  # key set intersection
        if clashing_kwargs:
            raise ValueError(
                f"Illegal attempt to register Browsergym environment {id} with both frozen and default values for task parameters {clashing_kwargs}."
            )

    task_entrypoint = task_class

    # freeze task_kwargs (cannot be overriden at environment creation)
    task_entrypoint = frozen_partial(task_class, **task_kwargs)

    # pre-set default_task_kwargs (can be overriden at environment creation)
    task_entrypoint = partial(task_entrypoint, **default_task_kwargs)

    gym.register(
        id=f"browsergym/{id}",
        entry_point=lambda *env_args, **env_kwargs: BrowserEnvWebSynth(
            task_entrypoint, *env_args, **env_kwargs
        ),
        nondeterministic=nondeterministic,
        *args,
        **kwargs,
    )


ALL_WEBSYNTH_GYM_IDS = []
ALL_WEBSYNTH_TASK_IDS = []

config_path: Path = get_a3_synth_config_path()

# register all WebArena benchmark
for task_id in get_task_ids(config_path):
    gym_id = f"a3_synth.{task_id}"
    register_task_a3_synth(
        gym_id,
        GenericWebSynthTask,
        task_kwargs={"task_id": task_id, 'config_path': config_path},
    )
    ALL_WEBSYNTH_GYM_IDS.append(gym_id)
    ALL_WEBSYNTH_TASK_IDS.append(task_id)

logger.info(f"Registered {len(ALL_WEBSYNTH_GYM_IDS)} WebSynth tasks")
logger.info(f"Sample Task IDs: {ALL_WEBSYNTH_GYM_IDS[0:200:20]}")