from textwrap import dedent
from typing import Union
from pathlib import Path
import json
import os

from beartype import beartype
from playwright.sync_api import CDPSession, Page
from webarena.browser_env.actions import Action
from webarena.browser_env.utils import StateInfo
from webarena.evaluation_harness.evaluators import Evaluator, StringEvaluator, URLEvaluator, HTMLContentEvaluator
import openai

from llm_annotators.utils import (
    select_client_config,
)
from llm_annotators.utils.websynth import print_last_step_output, LastStepOutputType
import llm_annotators.judge.utils as judge_utils

ChatMessage = dict[str, str]
Trajectory = list[Union[Action, StateInfo, ChatMessage]]

if "WEBSYNTH_LLM_JUDGE_MODEL" not in os.environ:
    raise ValueError("WEBSYNTH_LLM_JUDGE_MODEL is not set. Please set it before running this script.")
LLM_JUDGE_MODEL = os.environ["WEBSYNTH_LLM_JUDGE_MODEL"]

def is_step_info_valid(step_info) -> bool:
    if step_info.obs is None:
        return False
    return True

def format_step_info(step_info, use_som=False) -> str:
    from copy import deepcopy
    from browsergym.utils.obs import flatten_axtree_to_str

    screenshot = step_info.obs['screenshot_som'] if use_som else step_info.obs['screenshot']
    screenshot_base64 = judge_utils.convert_numpy_array_to_base64(screenshot)
    
    goal = step_info.obs['goal']
    open_pages_urls = step_info.obs['open_pages_urls']
    active_page_index = step_info.obs['active_page_index']
    url = step_info.obs['url']
    axtree_txt = step_info.obs['axtree_txt']
    axtree_obj = step_info.obs['axtree_object']
    extra_element_properties = step_info.obs['extra_element_properties']
    axtree_pruned = flatten_axtree_to_str(
        deepcopy(axtree_obj), 
        filter_visible_only=True, 
        filter_with_bid_only=True, 
        extra_properties=extra_element_properties
    )
    reasoning = step_info.agent_info['think']
    
    return {
        "goal": goal,
        "url": url,
        "action": step_info.action,
        "reasoning": reasoning,
        "screenshot_base64": screenshot_base64,
        "axtree_txt": axtree_txt,
        "axtree_pruned": axtree_pruned,
        "open_pages_urls": open_pages_urls,
        "active_page_index": active_page_index,
    }

def format_trajectory(episode_info: list[dict]) -> dict:
    return {
        'goal': episode_info[0].obs['goal'],
        'steps': [format_step_info(step_info) for step_info in episode_info],
    }

def format_persona(persona: dict) -> str:
    template = dedent("""\
    Name: {name}
    Skills: {skills}
    Interests: {interests}
    Description: {description}
    """)
    name = persona["name"]
    skills = ", ".join(persona["skills"])
    interests = ", ".join(persona["interests"])
    description = persona["description"]
    return template.format(name=name, skills=skills, interests=interests, description=description)

def format_intent_context(intent: str, persona: str, hint: str) -> str:
    template = dedent("""\
    <intent>
    {intent}
    </intent>
    <persona>
    {persona}
    </persona>
    <hint>
    {hint}
    </hint>
    """)
    return template.format(intent=intent, hint=hint, persona=persona)

def remove_invalid_step_infos(last_step_output: LastStepOutputType) -> list[dict]:
    return [step_info for step_info in last_step_output["episode_info"] if is_step_info_valid(step_info)]

def llm_judge_match_with_hint(
    client: openai.OpenAI, 
    model: str,
    hint: str, 
    intent: str, 
    last_step_output: LastStepOutputType,
    persona: dict,
    enable_thinking=True,
    use_screenshot=True,
    use_axtree=True,
    max_tokens=4096,
) -> dict:
    from llm_annotators.judge import create_chat_messages_from_trajectory, get_content_inside_tag
    from llm_annotators.utils import get_completion_kwargs

    print_last_step_output(last_step_output, func_name='llm_judge_match_with_hint')
    if last_step_output is None:
        return {"score": 0.0, "response": None}

    if len(last_step_output["episode_info"]) <= 0:
        print("[llm_judge_match_with_hint] Length of episode_info is 0")
        return {"score": 0.0, "response": None}

    # remove invalid step infos
    step_infos = remove_invalid_step_infos(last_step_output)
    if len(step_infos) <= 0:
        print("[llm_judge_match_with_hint] Length of step_infos is 0")
        return {"score": 0.0, "response": None}

    trajectory = format_trajectory(step_infos)
    
    goal = format_intent_context(intent=intent, persona=persona, hint=hint)

    messages = create_chat_messages_from_trajectory(
        trajectory=trajectory, goal=goal, use_screenshot=use_screenshot, use_axtree=use_axtree
    )
    # if it uses qwen3, we need to add the extra_body
    kwargs = get_completion_kwargs(model=model, enable_thinking=enable_thinking)
    try:
        msg = messages['regular']
        response = client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            **kwargs,
        )
    except Exception as e:
        print("[llm_judge_match_with_hint] Error in llm_judge_match_with_hint: {e}")
        print("[llm_judge_match_with_hint] Trying with pruned messages")
        msg = messages['pruned']
        response = client.chat.completions.create(
            model=model,
            messages=msg,
            max_tokens=max_tokens,
            **kwargs,
        )

    response_dict = response.model_dump()
    
    response_msg = response.choices[0].message.content.lower()
    answer = get_content_inside_tag("answer", response_msg, lower=True)
    print(f"[llm_judge_match_with_hint] Answer: {answer}")

    # save the response to a json file
    step_info = last_step_output["episode_info"][-1]
    
    with open(Path(step_info.exp_dir, f"llm_judge_step_{step_info.step}.json"), "w") as f:
        json.dump({
            "step": step_info.step - 1,
            "response": response_dict,
            "answer": answer,
            "score": float(answer == "successful"),
            "input_messages": msg,
        }, f, indent=4)
    
    return {"score": float(answer == "successful"), "response": response_dict}


class LLMJudgeWithHintEvaluator(StringEvaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """
    def judge_match_with_hint(self, hint: str, pred: str, intent: str, persona: str, trajectory: Trajectory, last_step_output: LastStepOutputType, max_tokens: int = 4096, use_screenshot: bool = True, use_axtree: bool = True) -> float:
        enable_thinking = True
        model = LLM_JUDGE_MODEL
        config = select_client_config(model)
        client = openai.OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        print_last_step_output(last_step_output, func_name='judge_match_with_hint')
        # we ignore the pred, trajectory here, as it is simply a placeholder by browsergym
        results = llm_judge_match_with_hint(
            client=client, model=model, hint=hint, intent=intent, 
            enable_thinking=enable_thinking, persona=persona, use_screenshot=use_screenshot, 
            use_axtree=use_axtree, max_tokens=max_tokens,
            last_step_output=last_step_output,
        )

        return results["score"]

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | None = None,
        client: CDPSession | None = None,
        last_step_output: LastStepOutputType = None,
        max_tokens: int = 4096,
        use_screenshot: bool = True,
        use_axtree: bool = True,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        last_action = self.get_last_action(trajectory)
        pred = self.clean_answer(last_action["answer"])

        score = 1.0
        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)

                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        score *= self.exact_match(ref=value, pred=pred)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if score != 1:
                            score = 1.0 * self.ua_match(
                                intent=configs["intent"],
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                    else:
                        assert isinstance(value, list)
                        for reference in value:
                            score *= self.fuzzy_match(
                                ref=reference, pred=pred, intent=intent
                            )
                case "hint":
                    score *= self.judge_match_with_hint(
                        hint=configs["eval"]["reference_answers"]["hint"],
                        pred=pred,
                        intent=configs["intent"],
                        persona=configs["persona"],
                        trajectory=trajectory,
                        last_step_output=last_step_output,
                        max_tokens=max_tokens,
                        use_screenshot=use_screenshot,
                        use_axtree=use_axtree,
                    )
        return score


class EvaluatorCombWebsynth:
    def __init__(self, evaluators: list[Evaluator]) -> None:
        self.evaluators = evaluators

    @beartype
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | None,
        client: CDPSession | None,
        last_step_output: LastStepOutputType,
        use_screenshot: bool = True,
        use_axtree: bool = True,
    ) -> float:
        score = 1.0
        for evaluator in self.evaluators:
            if type(evaluator) in (HTMLContentEvaluator, URLEvaluator, StringEvaluator):
                cur_score = evaluator(trajectory, config_file, page, client)
            else:
                # for other evaluators like LLMJudgeWithHintEvaluator, pass the last_step_output 
                # as kwargs
                print_last_step_output(last_step_output, func_name='EvaluatorCombWebsynth.__call__')
                cur_score = evaluator(trajectory, config_file, page, client, last_step_output=last_step_output, use_screenshot=use_screenshot, use_axtree=use_axtree)
            score *= cur_score
        return score

@beartype
def evaluator_router_websynth(config_file: Path | str):
    """Router to get the evaluator class"""
    with open(config_file, "r") as f:
        configs = json.load(f)

    eval_types = configs["eval"]["eval_types"]
    evaluators: list[Evaluator] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator())
            case "url_match":
                evaluators.append(URLEvaluator())
            case "program_html":
                evaluators.append(HTMLContentEvaluator())
            case "llm_judge_with_hint":
                evaluators.append(LLMJudgeWithHintEvaluator())
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorCombWebsynth(evaluators)