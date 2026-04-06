import logging
from pathlib import Path

import nltk
from browsergym.core.registration import register_task

from . import GenericExplorationTask, get_task_ids

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# download necessary tokenizer resources
# note: deprecated punkt -> punkt_tab https://github.com/nltk/nltk/issues/3293
try:
    nltk.data.find("tokenizers/punkt_tab")
except:
    nltk.download("punkt_tab", quiet=True, raise_on_error=True)

ALL_EXPLORE_GYM_IDS = []
ALL_EXPLORE_TASK_IDS = []


def _register_tasks():
    """Register exploration tasks. Re-reads CONFIG_PATH from the parent module
    to support worker processes that set env vars before importing."""
    global ALL_EXPLORE_GYM_IDS, ALL_EXPLORE_TASK_IDS

    # Import CONFIG_PATH fresh (may have been updated by EnvArgsParallel.make_env)
    from . import CONFIG_PATH
    config_path = CONFIG_PATH

    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found, skipping registration")
        return

    if ALL_EXPLORE_TASK_IDS:
        logger.info(f"Tasks already registered ({len(ALL_EXPLORE_TASK_IDS)} tasks), skipping")
        return

    logger.info(f"Registering exploration tasks from config file: {config_path}")
    for i, task_id in enumerate(get_task_ids(config_path)):
        gym_id = f"exploration.{task_id}"
        register_task(
            gym_id,
            GenericExplorationTask,
            task_kwargs={"task_id": task_id, 'config_path': config_path},
        )
        ALL_EXPLORE_GYM_IDS.append(gym_id)
        ALL_EXPLORE_TASK_IDS.append(task_id)

    logger.info(f"Registered {len(ALL_EXPLORE_GYM_IDS)} exploration tasks")
    logger.info(f"Sample Task IDs: {ALL_EXPLORE_GYM_IDS[0:200:20]}")


# Register tasks on import (works in main process where env vars are set)
_register_tasks()
