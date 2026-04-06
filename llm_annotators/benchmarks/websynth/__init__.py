import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import importlib.resources
import tempfile
import urllib.parse

from webarena.browser_env.actions import Action
from webarena.browser_env.utils import StateInfo
import playwright.sync_api
from browsergym.core.task import AbstractBrowserTask
from browsergym.experiments.benchmark.base import Benchmark
from browsergym.experiments.loop import EnvArgs
import requests

from llm_annotators.utils.websynth import print_last_step_output, LastStepOutputType

Trajectory = list[Union[Action, StateInfo]]

logger = logging.getLogger(__name__)

ENV_VARS = ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE")

def get_websynth_config_path():
    if "WEBSYNTH_CONFIG_NAME" not in os.environ:
        raise ValueError("WEBSYNTH_CONFIG_NAME is not set. Please set it before running this script.")

    CONFIG_FILE_NAME = os.environ["WEBSYNTH_CONFIG_NAME"]
    CONFIG_ANCHOR = os.getenv("WEBSYNTH_CONFIG_ANCHOR", "llm_annotators.configs.websynth")

    CONFIG_PATH: Path = importlib.resources.files(CONFIG_ANCHOR).joinpath(CONFIG_FILE_NAME)

    if not CONFIG_FILE_NAME.endswith(".json"):
        # warn user that the config file is not a json file
        logger.warning(f"Config file {CONFIG_FILE_NAME} is not a json file. This will likely cause errors later.")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file {CONFIG_PATH} not found")

    return CONFIG_PATH

class WebSynthInstance:
    """
    Utility class to access a WebArena instance.

    """

    RESET_URL_VAR = "WA_FULL_RESET"  # used by full_reset()

    def __init__(
        self,
    ) -> None:

        # setup webarena environment variables (webarena will read those on import)
        append_wa = lambda x: f"WA_{x}" # noqa: E731
        for key in ENV_VARS:
            assert append_wa(key) in os.environ, (
                f"Environment variable {append_wa(key)} missing.\n"
                + "Please set the following environment variables to use WebArena through BrowserGym:\n"
                + "\n".join([append_wa(x) for x in ENV_VARS])
            )
            os.environ[key] = os.environ[append_wa(key)]

        # import webarena on instanciation
        from webarena.browser_env.env_config import (
            ACCOUNTS,
            GITLAB,
            HOMEPAGE,
            MAP,
            REDDIT,
            SHOPPING,
            SHOPPING_ADMIN,
            WIKIPEDIA,
        )

        self.urls = {
            "reddit": REDDIT,
            "gitlab": GITLAB,
            "shopping": SHOPPING,
            "shopping_admin": SHOPPING_ADMIN,
            "wikipedia": WIKIPEDIA,
            "map": MAP,
        }
        self.home_url = HOMEPAGE

        self.credentials = ACCOUNTS

    def full_reset(self, skip_if_not_set: bool = True):
        base_url = os.environ.get(self.RESET_URL_VAR, None)

        if not base_url:
            # check for reset URL
            logger.error(
                f"Environment variable {self.RESET_URL_VAR} is missing or empty, required for a full instance reset."
            )
            if skip_if_not_set:
                logger.warning(
                    "Skipping automated reset. Make sure the instance has been manually reset."
                )
            else:
                raise RuntimeError("Could not reset instance, aborting.")

        else:
            # reset the instance
            reset_url = f"{base_url}/reset"
            status_url = f"{base_url}/status"

            logger.info(
                f"Initiating {self.__class__.__name__} instance reset on URL {reset_url}. Should take between 200 - 500 seconds to restart."
            )

            # trigger instance reset
            response = requests.get(reset_url)
            match response.status_code:
                case 200:
                    logger.info("Reset started.")
                case 418:
                    logger.warning("Reset was already running.")
                case _:
                    raise Exception(
                        f"{self.__class__.__name__} reset request {reset_url} failed ({response.status_code}): {response.text}"
                    )

            # wait until reset complete
            retry_after = 20  # 20 seconds wait between status checks
            timeout = 10 * 60  # 10 minutes timeout
            start_time = time.time()
            while True:
                # request instance status
                response = requests.get(status_url)
                # check for server error
                if response.status_code != 200:
                    raise Exception(
                        f"{self.__class__.__name__} status request {status_url} failed ({response.status_code}): {response.text}"
                    )
                # check for readiness
                if response.text == "Ready for duty!":
                    break
                # check for timeout
                time_elapsed = time.time() - start_time
                logger.info(f"Reset still running after {time_elapsed:.0f} seconds...")
                if time_elapsed > timeout:
                    raise Exception(
                        f"Reset still running after {time_elapsed} seconds (> {timeout}), aborting."
                    )
                # wait a bit before next retry
                time.sleep(retry_after)

        # warm-start the instance (navigate to every domain)
        retries_left = 3
        while retries_left:
            retries_left -= 1
            try:
                self._check_is_reachable(
                    timeout=60
                )  # 60 seconds, warming up after reset might be slow
                break
            except Exception as e:
                if not retries_left:
                    raise
                logger.info(
                    f"Instance unresponsive after reset, retrying ({retries_left} retries left)\n{e}"
                )

    def check_status(self):
        """
        Check the status of the instance. Raises an error if the instance is not ready to be used.

        """
        self._check_is_reachable(timeout=10)  # 10 seconds

    def _check_is_reachable(self, timeout: int):
        """
        Test that every website is reachable.

        """
        for site, url in self.urls.items():
            try:
                requests.get(url, timeout=timeout)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                raise RuntimeError(
                    f'WebArena site "{site}" ({url}) is not reacheable. Please check the URL.'
                )

    def ui_login(self, site: str, page: playwright.sync_api.Page):
        """
        Should only be called once per site (expects user to be logged out).
        """

        url = self.urls[site]

        # open a new page (tab) to perform the login
        page = page.context.new_page()

        match site:
            case "reddit":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(f"{url}")
                page.get_by_role("link", name="Log in").click()
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Log in").click()

            case "gitlab":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(f"{url}/users/sign_in")
                page.get_by_label("Username or email").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Sign in").click()

            case "shopping":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(f"{url}/customer/account/login/")
                page.get_by_label("Email", exact=True).fill(username)
                page.get_by_label("Password", exact=True).fill(password)
                page.get_by_role("button", name="Sign In").click()

            case "shopping_admin":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(url)
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Sign in").click()

            case "wikipedia":
                page.goto(url)

            case "map":
                page.goto(url)

            case _:
                raise ValueError

        # release login page
        page.close()

class GenericWebSynthTask(AbstractBrowserTask):
    """
    Base class for all WebArena tasks.
    """
    def __init__(
        self,
        seed: int,
        task_id: Optional[int] = None,
        intent_template_id: Optional[int] = None,
        with_na_hint: bool = False,
        with_homepage_hint: bool = False,
        config_path: str = None,
        use_screenshot: bool = True,
        use_axtree: bool = True,
    ) -> None:
        super().__init__(seed)

        if config_path is None:
            raise ValueError(
                "No config_path provided. Please provide a path to the WebArena config file."
            )
        if not os.path.exists(config_path):
            raise ValueError(
                f"Config file {config_path} does not exist. Please provide a valid path to the WebArena config file."
            )
        
        # task properties, will be used to set up the browsergym environment
        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 1000  # ms
        self.timeout = 10000  # ms

        self.webarena_instance = WebSynthInstance()
        self.config_file: str = None
        self.with_na_hint = with_na_hint
        self.with_homepage_hint = with_homepage_hint
        self.use_screenshot = use_screenshot
        self.use_axtree = use_axtree

        # one and only one of task id and template id must be provided
        if (task_id is None) == (intent_template_id is None):
            raise ValueError(
                f"One and only one of 'task_id' and 'intent_template_id' must be provided (task_id={task_id}, intent_template_id={intent_template_id})."
            )

        # read the list of all webarena task configs
        import webarena # noqa: F401

        # all_configs_str = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
        all_configs_str = Path(config_path).read_text()

        # substitute URLs
        for pattern, url_key in {
            "__GITLAB__": "gitlab",
            "__REDDIT__": "reddit",
            "__SHOPPING__": "shopping",
            "__SHOPPING_ADMIN__": "shopping_admin",
            "__WIKIPEDIA__": "wikipedia",
            "__MAP__": "map",
        }.items():
            all_configs_str = all_configs_str.replace(pattern, self.webarena_instance.urls[url_key])

        # load all task configs to JSON
        all_configs = json.loads(all_configs_str)

        # keep only the desired task configs
        if intent_template_id is not None:
            task_configs = [
                conf for conf in all_configs if conf["intent_template_id"] == intent_template_id
            ]
            if not task_configs:
                raise ValueError(
                    f"Could not find any task config with intent_template_id={intent_template_id}."
                )

        elif task_id is not None:
            task_configs = [conf for conf in all_configs if conf["task_id"] == task_id]
            if not task_configs:
                raise ValueError(
                    f"Could not find any task config with task_id={intent_template_id}."
                )

        self.task_configs = task_configs

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        # import our custom webarena-derived evaluator on instanciation
        from llm_annotators.benchmarks.websynth.evaluators import evaluator_router_websynth

        # pick a task at random
        self.config = self.random.choice(self.task_configs)

        # hack: dynamically build a config file to read from
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            json.dump(self.config, f)
            f.flush()
            self.config_file = f.name

        # build the evaluator
        self.evaluator = evaluator_router_websynth(self.config_file)

        # authenticate
        for site in self.config["sites"]:
            self.webarena_instance.ui_login(site=site, page=page)

        # set geolocation
        page.context.set_geolocation(self.config["geolocation"])

        # navigate to the starting url(s) (might need several pages)
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/browser_env/envs.py#L150
        if self.config["start_url"]:
            start_urls = self.config["start_url"].split(" |AND| ")
            for i, url in enumerate(start_urls):
                page.goto(url)
                if i < len(start_urls) - 1:
                    page = page.context.new_page()

        # recover goal
        goal = self.config["intent"]

        # This note is present in all webarena's agent prompts
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/agent/prompts/raw/p_cot_id_actree_2s.py#L34
        if self.with_homepage_hint:
            goal += f"""

(Note: if you want to visit other websites, check out the homepage at {self.webarena_instance.home_url}. It has a list of websites you can visit. {self.webarena_instance.home_url}/password.html lists all the account name and password for the websites. You can use them to log in to the websites.)
"""

        # This note is present in some of webarena's agent prompts
        if self.with_na_hint:
            goal += """\

If you believe the task is impossible to complete, provide the answer "N/A".
"""

        return goal, {}

    def cheat(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> None:
        raise NotImplementedError

    @classmethod
    def get_task_id(cls):
        """
        Generic class for several task ids, this way of obtaining the task id is not compatible for now.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        # Nothing to be done here
        # https://github.com/web-arena-x/webarena/blob/c6475f0e9affe5252a2966e26b8cb4c834a4ae40/browser_env/envs.py#L227
        pass

    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list[str], last_step_output: LastStepOutputType
    ) -> Tuple[float, bool, str, dict]:

        # safeguard: check that all open tabs are either blank or within the list of WebArena URLs
        authorized_locations = ["newtab", ""] + [
            urllib.parse.urlparse(url).netloc
            for url in [*self.webarena_instance.urls.values(), self.webarena_instance.home_url]
        ]
        for open_page in page.context.pages:
            page_location = urllib.parse.urlparse(open_page.url).netloc
            if page_location not in authorized_locations:
                return 0, True, "", {"error": "Unauthorized url, terminating task"}

        # import webarena dynamically
        from webarena.browser_env.actions import ActionTypes

        # if any, use the last assistant message as the stop answer for webarena
        if chat_messages and chat_messages[-1]["role"] == "assistant":
            last_action = {"action_type": ActionTypes.STOP, "answer": chat_messages[-1]["message"]}
        elif chat_messages and chat_messages[-1]["role"] == "infeasible":
            last_action = {"action_type": ActionTypes.STOP, "answer": "N/A"}
        else:
            last_action = {"action_type": ActionTypes.NONE, "answer": ""}
            # llm_fuzzy_match() bugfix
            last_action["answer"] = "whatever"

        # hack: fake trajectory for evaluation (only last_action["answer"] is used in the webarena evaluation codebase)
        trajectory = [{}, last_action]  # list[str], Action
        # MODIFIED: trajectory is now a list of chat_messages and the last action, instead of a hack
        # trajectory = [chat_messages, last_action]  # list[str], Action

        # call the evaluator
        use_screenshot = os.getenv("WEBSYNTH_LLM_JUDGE_USE_SCREENSHOT", "True").lower() == "true"
        use_axtree = os.getenv("WEBSYNTH_LLM_JUDGE_USE_AXTREE", "True").lower() == "true"

        try:
            print_last_step_output(last_step_output, func_name='GenericWebSynthTask.validate')
            score = self.evaluator(
                trajectory=trajectory,
                config_file=self.config_file,
                page=page,
                client=None,  # none of webarena's evaluators requires a cdp session
                last_step_output=last_step_output,
                use_screenshot=use_screenshot,
                use_axtree=use_axtree,
            )
        # llm_fuzzy_match() bugfix (assert "correct" in response)
        except AssertionError:
            logger.debug(
                "llm_fuzzy_match() bugfix applied: AssertionError in evaluator, using score = 0.0"
            )
            score = 0.0

        if score > 0 or last_action["action_type"] == ActionTypes.STOP:
            return score, True, "", {}
        else:
            return score, False, "", {}
        

class EnvArgsParallel(EnvArgs):
    """
    NOTE: This is needed in order to run the parallel runs with the websynth benchmark, i.e. via "ray" with 4 jobs.
    """
    def make_env(self, action_mapping, exp_dir, exp_task_kwargs: dict = {}):
        logger.info("Importing llm_annotators.benchmarks.websynth.register inside EnvArgsParallel to ensure tasks are registered...")
        from . import register  # noqa: F401

        return super().make_env(action_mapping, exp_dir, exp_task_kwargs)

def make_env_args_list_from_fixed_seeds_parallel(
    task_list: list[str], max_steps: int, fixed_seeds: list[int]
):
    """
    Generates a list of `len(task_list)` time `n_repeats` environments arguments, using randomly generated seeds.
    NOTE: This is needed in order to run the parallel runs with the websynth benchmark, i.e. via "ray" with 4 jobs.
    """
    env_args_list = []
    for task in task_list:
        for seed in fixed_seeds:
            env_args_list.append(
                EnvArgsParallel(
                    task_name=task,
                    task_seed=int(seed),
                    max_steps=max_steps,
                    headless=True,
                    record_video=False,
                    wait_for_user_message=False,
                    viewport=None,
                    slow_mo=None,
                    storage_state=None,
                    task_kwargs=None,
                )
            )

    return env_args_list

def get_task_ids(config_path):
    """
    Get the number of task ids from the config file.
    """
    all_configs = json.loads(Path(config_path).read_text())

    return [conf["task_id"] for conf in all_configs]


def list_config_files(anchor="llm_annotators.configs"):
    base_configs = importlib.resources.files(anchor)
    return [f.name for f in base_configs.iterdir() if f.is_file() and f.suffix == ".json"]

def prepare_websynth_backend(backend, reset_instance=True):
    # register environments
    import llm_annotators.benchmarks.websynth.register  # noqa: F401
    
    # backend is not used, but is required by the original benchmark class, so we pass it in
    # as a convention
    if backend != "websynth":
        raise ValueError(f"Invalid backend: {backend}. Only 'websynth' is supported.")
    
    # full reset the instance (requires environment variables properly set up)
    from browsergym.webarena.instance import WebArenaInstance
    from browsergym.experiments.benchmark.utils import massage_tasks

    default_instance = WebArenaInstance()
    if reset_instance:
        logging.info("Resetting websynth instance...")
        default_instance.full_reset()
    else:
        logging.info("Resetting websynth instance has been disabled.")

    logging.info(
        f"Initiating {backend} instance warm-up. Some tasks will be pre-loaded (massaged) to trigger some caching mechanisms and make the server more responsive."
    )
    task_ids = get_task_ids(get_websynth_config_path())
    # N = len(task_ids)
    
    massage_tasks(
        [
            f"websynth.{id}"
            for id in [
                task_ids[0],
                # TODO: uncomment these after testing
                # task_ids[N//4],
                # task_ids[N//2],
                # task_ids[3*N//4],
                # task_ids[N-1],
            ]
        ]
    )

@dataclass
class WebSynthBenchmark(Benchmark):
    reset_instance: bool = True
    
    def prepare_backends(self):
        backend = "websynth"
        
        logger.info(f"Preparing {backend} backend...")
        prepare_websynth_backend(backend, reset_instance=self.reset_instance)
        logger.info(f"{backend} backend ready")
