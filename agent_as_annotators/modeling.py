# flake8: noqa: E402
from functools import partial
from pathlib import Path
import os
from dataclasses import dataclass
import logging

# this needs to be set before importing agentlab
default_exp_root = str(Path(__file__).parent / "agentlab_results")
os.environ["AGENTLAB_EXP_ROOT"] = os.getenv("AGENTLAB_EXP_ROOT", default_exp_root)

from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgent
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.chat_api import ChatModel
import agentlab.llm.tracking as tracking
from agentlab.llm.llm_utils import AIMessage
from agentlab.llm.chat_api import handle_error, RetryError, OpenRouterError
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_pricing_custom():
    """Returns a dictionary of model pricing for OpenAI models."""
    try:
        cost_dict = MODEL_COST_PER_1M_TOKENS
    except Exception as e:
        logging.warning(
            f"Failed to get OpenAI pricing: {e}. "
            "Please install langchain-community or use LiteLLM API for pricing information."
        )
        return {}
    cost_dict = {k: v / 1000000 for k, v in cost_dict.items()}
    res = {}
    for k in cost_dict:
        if k.endswith("-completion"):
            continue
        prompt_key = k
        completion_key = k + "-completion"
        if completion_key in cost_dict:
            res[k] = {
                "prompt": cost_dict[prompt_key],
                "completion": cost_dict[completion_key],
            }
    return res

MODEL_COST_PER_1M_TOKENS = {
    # GEMINI MODELS
    "gemini-3-pro-preview": 2.0,
    "gemini-3-pro-preview-completion": 12.0,
    "gemini-3.1-pro-preview": 2.0,
    "gemini-3.1-pro-preview-completion": 12.0,
    "gemini-2.5-flash": 0.3,
    "gemini-2.5-flash-completion": 1.0,
    "gemini-3-flash-preview": 0.5,
    "gemini-3-flash-preview-completion": 3.0,
    "gemini-3.1-flash-lite-preview": 0.25,
    "gemini-3.1-flash-lite-preview-completion": 1.5,
    # GPT MODELS
    "gpt-5.1": 1.25,
    "gpt-5.1-completion": 10.0,
    "gpt-5": 1.25,
    "gpt-5-completion": 10.0,
    "gpt-5-mini": 0.25,
    "gpt-5-mini-completion": 2.0,
    "gpt-5-nano": 0.05,
    "gpt-5-nano-completion": 0.4,
    "gpt-4.1": 2.0,
    "gpt-4.1-completion": 8.0,
    "gpt-4.1-mini": 0.4,
    "gpt-4.1-mini-completion": 1.6,
    "gpt-4.1-nano": 0.1,
    "gpt-4.1-nano-completion": 0.4,
    "gpt-4o": 2.5,
    "gpt-4o-completion": 10.0,
    "gpt-4o-2024-05-13": 5.0,
    "gpt-4o-2024-05-13-completion": 15.0,
    "gpt-4o-mini": 0.15,
    "gpt-4o-mini-completion": 0.6,
    "gpt-5.1-codex-mini": 0.25,
    "gpt-5.1-codex-mini-completion": 2.0,
    "codex-mini-latest": 1.5,
    "codex-mini-latest-completion": 6.0,
    "gpt-5-search-api": 1.25,
    "gpt-5-search-api-completion": 10.0,
    "gpt-4o-mini-search-preview": 0.15,
    "gpt-4o-mini-search-preview-completion": 0.6,
    "gpt-4o-search-preview": 2.5,
    "gpt-4o-search-preview-completion": 10.0,
    "computer-use-preview": 3.0,
    "computer-use-preview-completion": 12.0,
}

class WebsynthAgentArgs(GenericAgentArgs):
    def make_agent(self):
        return WebsynthAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )

class WebsynthAgent(GenericAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Make thinking and chat messages added back to the prompt
        
class OpenAICompatibleChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        api_key_env_var,
        base_url_env_var,
        api_key=None,
        base_url=None,
        temperature=0.5,
        max_tokens=1024,
        max_retry=4,
        min_retry_wait_time=60,
        extra_body=None,
    ):
        import agentlab.llm.tracking as tracking

        if not api_key_env_var in os.environ:
            raise ValueError(f"{api_key_env_var} must be set in the environment")

        # Only require base_url_env_var if base_url is not explicitly provided
        if base_url is None:
            if not base_url_env_var in os.environ:
                raise ValueError(f"{base_url_env_var} must be set in the environment (or pass base_url explicitly)")
            base_url = os.environ[base_url_env_var]

        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var=api_key_env_var,
            client_class=OpenAI,
            client_args={
                "base_url": base_url,
            },
            log_probs=None,
            pricing_func=get_pricing_custom,
        )
        self.extra_body = extra_body

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        e = None
        for itr in range(self.max_retry):
            self.retries += 1
            temperature = temperature if temperature is not None else self.temperature
            try:
                # Build kwargs - exclude logprobs for Gemini as it's not supported
                create_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "n": n_samples,
                    "temperature": temperature,
                    "max_completion_tokens": self.max_tokens,
                }
                if self.extra_body is not None:
                    create_kwargs["extra_body"] = self.extra_body
                if not self.model_name.startswith("gemini"):
                    create_kwargs["logprobs"] = self.log_probs
                completion = self.client.chat.completions.create(**create_kwargs)

                if completion.usage is None:
                    raise OpenRouterError(
                        "The completion object does not contain usage information. This is likely a bug in the OpenRouter API."
                    )

                self.success = True
                break
            except openai.OpenAIError as e:
                error_type = handle_error(e, itr, self.min_retry_wait_time, self.max_retry)
                self.error_types.append(error_type)

        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {error_type}"
            )

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance, tracking.LLMTracker
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        if n_samples == 1:
            res = AIMessage(completion.choices[0].message.content)
            if self.log_probs:
                res["log_probs"] = completion.choices[0].log_probs
            return res
        else:
            return [AIMessage(c.message.content) for c in completion.choices]


@dataclass
class VllmModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an OpenAI
    model."""

    def set_base_url(self, base_url):
        self.base_url = base_url

    def set_api_key(self, api_key):
        self.api_key = api_key

    def make_model(self):
        base_url = None if not hasattr(self, "base_url") else self.base_url
        api_key = None if not hasattr(self, "api_key") else self.api_key

        return OpenAICompatibleChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            api_key_env_var="VLLM_API_KEY",
            base_url_env_var="VLLM_BASE_URL",
            base_url=base_url,
            api_key=api_key,
        )

@dataclass
class GeminiModelArgs(BaseModelArgs):
    extra_body: dict = None

    def set_base_url(self, base_url):
        self.base_url = base_url

    def set_api_key(self, api_key):
        self.api_key = api_key

    def make_model(self):
        base_url = None if not hasattr(self, "base_url") else self.base_url
        api_key = None if not hasattr(self, "api_key") else self.api_key

        return OpenAICompatibleChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            api_key_env_var="GEMINI_API_KEY",
            base_url_env_var="GEMINI_BASE_URL",
            base_url=base_url,
            api_key=api_key,
            extra_body=self.extra_body,
        )

def get_agent_occam_action_set_args():
    from browsergym.core.action.highlevel import HighLevelActionSet
    from browsergym.core.action.functions import (
        click,
        fill,
        go_back,
        keyboard_press,
        send_msg_to_user,
    )
    custom_actions = [
        #                   #     code      |      paper       |      prompt
        keyboard_press,  #    KEY_PRESS     | press(key_comb)  | press [key_comb]
        #                     MOUSE_CLICK   |                  |
        #                     KEYBOARD_TYPE |                  |
        #                     MOUSE_HOVER   |                  |
        click,  #             CLICK         | click(elem)      | click [id]
        fill,  #              TYPE          | type(elem, text) | type [id] [content]
        go_back,  #           GO_BACK       | go_back()        | go_back
        #                     CHECK         |                  |
        send_msg_to_user,  #  STOP          | stop(answer)     | stop [answer]
        # note() is not supported in BrowserGym, so we don't include it.
        # stop() is not supported in BrowserGym, so we don't include it. send_msg_to_user is used instead
        # go_home() is not supported in BrowserGym, so we don't include it.

        # THE FOLLOWING ACTIONS HAVE BEEN REMOVED FROM THE ACTION SET
        # report_infeasible,  ## explicit unachievable action, equivalent STOP "N/A"
        # hover,  #             HOVER         | hover(elem)      | hover [id]
        # go_forward,  #        GO_FORWARD    | go_forward()     | go_forward
        # goto,  #              GOTO_URL      | goto(url)        | goto [url]
        # tab_focus,  #         PAGE_FOCUS    | tab_focus(index) | tab_focus [tab_index]
        # tab_close,  #         PAGE_CLOSE    | tab_close()      | close_tab
        # new_tab,  #           NEW_TAB       | new_tab()        | new_tab
        # select_option,  #     SELECT_OPTION |                  |
        # scroll,  #            SCROLL        | scroll(dir)      | scroll [down|up]
    ]
    return HighLevelActionSet(
        subsets=["custom"],
        custom_actions=custom_actions,
        multiaction=False,
        strict=False,
        retry_with_force=False,
        demo_mode=None,
    )

def get_flags(
    max_prompt_tokens,
    use_screenshot,
    use_som,
    use_ax_tree,
    enable_chat=False,
    use_agent_occam_actions=False,
):

    from agentlab.agents import dynamic_prompting as dp
    from agentlab.agents.generic_agent.generic_agent import GenericPromptFlags
    from browsergym.experiments.benchmark import HighLevelActionSetArgs

    if use_agent_occam_actions:
        print("USING AGENT OCCAM ACTIONS")
        action_set = get_agent_occam_action_set_args()
    else:
        action_set = HighLevelActionSetArgs(
            subsets=("bid",),
            multiaction=False,
        )

    flags = GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=use_ax_tree,
            use_focused_element=True,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=False,
            use_action_history=True,
            use_think_history=False,
            use_diff=False,
            html_type="pruned_html",
            use_screenshot=use_screenshot,
            use_som=use_som,
            extract_visible_tag=True,
            extract_clickable_tag=True,
            extract_coords="False",
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            action_set=action_set,
            long_description=False,
            individual_examples=False,
        ),
        use_plan=False,
        use_criticise=False,
        use_thinking=True,
        use_memory=False,
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=True,
        enable_chat=enable_chat,
        max_prompt_tokens=max_prompt_tokens,
        be_cautious=True,
        extra_instructions=None,
    )

    return flags

def prepare_vllm_model(
    model_name,
    max_new_tokens,
    max_prompt_tokens,
    max_total_tokens,
    temperature,
    use_vision,
    use_ax_tree,
    enable_chat=False,
    base_url=None,
    api_key=None,
    agent_name_suffix=None,
):
    # the base url and api key are set in VllmModelArgs's make_model,
    # so it is not necessary to set them here, but it is possible if needed
    model_args = VllmModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
        max_new_tokens=max_new_tokens,
        vision_support=use_vision,
        temperature=temperature,
    )
    if base_url is not None:
        model_args.set_base_url(base_url)
    if api_key is not None:
        model_args.set_api_key(api_key)

    agent_args = GenericAgentArgs(
        chat_model_args=model_args,
        flags=get_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_som=use_vision,
            use_screenshot=use_vision,
            enable_chat=enable_chat,
            use_ax_tree=use_ax_tree,
        ),
    )

    if agent_name_suffix:
        agent_args.agent_name = f"{agent_args.agent_name}_{agent_name_suffix}"

    return agent_args


def prepare_gemini(
    model_name,
    max_new_tokens,
    max_prompt_tokens,
    max_total_tokens,
    temperature,
    use_vision,
    use_ax_tree,
    extra_body=None,
    enable_chat=False,
    base_url=None,
    api_key=None,
    agent_name_suffix=None,
):
    # the base url and api key are set in GeminiModelArgs's make_model,
    # so it is not necessary to set them here, but it is possible if needed
    model_args = GeminiModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
        max_new_tokens=max_new_tokens,
        vision_support=use_vision,
        temperature=temperature,
        extra_body=extra_body,
    )
    if base_url is not None:
        model_args.set_base_url(base_url)
    if api_key is not None:
        model_args.set_api_key(api_key)

    agent_args = GenericAgentArgs(
        chat_model_args=model_args,
        flags=get_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_som=use_vision,
            use_screenshot=use_vision,
            enable_chat=enable_chat,
            use_ax_tree=use_ax_tree,
        ),
    )

    if agent_name_suffix:
        agent_args.agent_name = f"{agent_args.agent_name}_{agent_name_suffix}"

    return agent_args


def prepare_gpt(
    model_name,
    max_new_tokens,
    max_prompt_tokens,
    max_total_tokens,
    temperature,
    use_vision,
    use_ax_tree,
    enable_chat=False,
    agent_name_suffix=None,
):
    from agentlab.llm.chat_api import OpenAIModelArgs

    agent_arg = GenericAgentArgs(
        chat_model_args=OpenAIModelArgs(
            model_name=model_name,
            max_total_tokens=max_total_tokens,
            max_input_tokens=max_total_tokens - max_new_tokens,
            max_new_tokens=max_new_tokens,
            vision_support=use_vision,
            temperature=temperature,
        ),
        flags=get_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_screenshot=use_vision,
            use_som=use_vision,
            enable_chat=enable_chat,
            use_ax_tree=use_ax_tree,
        ),
    )

    if agent_name_suffix:
        agent_arg.agent_name = f"{agent_arg.agent_name}_{agent_name_suffix}"

    return agent_arg

# TODO: remove this function
def prepare_azure_gpt_41_mini():
    from agentlab.llm.chat_api import AzureModelArgs
    from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
    from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_4o

    model_config = AzureModelArgs(
            deployment_name="gpt-4.1-mini-2025-04-14",
            model_name="gpt-4.1-mini",
            max_new_tokens=16_384,
            max_input_tokens=40_000,
            max_total_tokens=40_000,
            vision_support=True,
            temperature=0.0,
        )

    FLAGS_GPT_4o = FLAGS_GPT_4o.copy()
    FLAGS_GPT_4o.obs.use_think_history = True
    GENERIC_AGENT_4_1_MINI = GenericAgentArgs(
        chat_model_args=model_config,
        flags=FLAGS_GPT_4o,
    )

    return GENERIC_AGENT_4_1_MINI

# TODO: remove this function
def prepare_vllm_webarena_prompted(
    model_name,
    max_new_tokens,
    max_prompt_tokens,
    max_total_tokens,
    temperature,
    use_vision,
    use_ax_tree,
    enable_chat=False,
    base_url=None,
    api_key=None,
):
    from .agents import WebArenaAgentArgs

    # the base url and api key are set in VllmModelArgs's make_model,
    # so it is not necessary to set them here, but it is possible if needed
    model_args = VllmModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
        max_new_tokens=max_new_tokens,
        vision_support=use_vision,
        temperature=temperature,
    )
    if base_url is not None:
        model_args.set_base_url(base_url)
    if api_key is not None:
        model_args.set_api_key(api_key)

    agent_args = WebArenaAgentArgs(
        chat_model_args=model_args,
        flags=get_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_som=use_vision,
            use_screenshot=use_vision,
            enable_chat=enable_chat,
            use_ax_tree=use_ax_tree,
        ),
    )

    return agent_args


# TODO: remove this function
def prepare_vllm_agent_occam_actions(
    model_name,
    max_new_tokens,
    max_prompt_tokens,
    max_total_tokens,
    temperature,
    use_vision,
    use_ax_tree,
    enable_chat=False,
    base_url=None,
    api_key=None,
):
    # the base url and api key are set in VllmModelArgs's make_model,
    # so it is not necessary to set them here, but it is possible if needed
    model_args = VllmModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
        max_new_tokens=max_new_tokens,
        vision_support=use_vision,
        temperature=temperature,
    )
    if base_url is not None:
        model_args.set_base_url(base_url)
    if api_key is not None:
        model_args.set_api_key(api_key)

    agent_args = GenericAgentArgs(
        chat_model_args=model_args,
        flags=get_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_som=use_vision,
            use_screenshot=use_vision,
            enable_chat=enable_chat,
            use_agent_occam_actions=True,
            use_ax_tree=use_ax_tree,
        ),
    )

    agent_args.agent_name = f"OccamActionsAgent-{model_args.model_name}".replace("/", "_")

    return agent_args

# TODO: remove this function
def prepare_vllm_webarena_prompted_url(
    model_name,
    max_new_tokens,
    max_prompt_tokens,
    max_total_tokens,
    temperature,
    use_vision,
    use_ax_tree,
    enable_chat=False,
    base_url=None,
    api_key=None,
):
    from .agents import WebArenaURLAgentArgs

    # the base url and api key are set in VllmModelArgs's make_model,
    # so it is not necessary to set them here, but it is possible if needed
    model_args = VllmModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
        max_new_tokens=max_new_tokens,
        vision_support=use_vision,
        temperature=temperature,
    )
    if base_url is not None:
        model_args.set_base_url(base_url)
    if api_key is not None:
        model_args.set_api_key(api_key)

    agent_args = WebArenaURLAgentArgs(
        chat_model_args=model_args,
        flags=get_flags(
            max_prompt_tokens=max_prompt_tokens,
            use_som=use_vision,
            use_screenshot=use_vision,
            enable_chat=enable_chat,
            use_ax_tree=use_ax_tree,
        ),
    )

    return agent_args