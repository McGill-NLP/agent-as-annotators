import logging
import os
import json
from pathlib import Path

# Valid thinking levels for Gemini models
GEMINI_THINKING_LEVELS = {
    "gemini-3-flash": ["minimal", "low", "medium", "high"],
    "gemini-3-pro": ["low", "medium", "high"],
    # Default for other/older Gemini models
    "default": ["low", "medium", "high"],
}

def _validate_gemini_thinking_level(shorthand: str, model_name: str, thinking_level: str):
    """Validate that thinking_level is valid for the given Gemini model."""
    # Determine which model type this is
    model_key = "default"
    if "gemini-3-flash" in shorthand or "gemini-3-flash" in model_name:
        model_key = "gemini-3-flash"
    elif "gemini-3-pro" in shorthand or "gemini-3-pro" in model_name:
        model_key = "gemini-3-pro"

    valid_levels = GEMINI_THINKING_LEVELS[model_key]
    if thinking_level not in valid_levels:
        raise ValueError(
            f"Invalid thinking_level '{thinking_level}' for model '{shorthand}'. "
            f"Valid options are: {valid_levels}"
        )

def _apply_default_kwargs(model_config: dict, defaults: dict[str, dict]) -> dict:
    """
    Merge default kwargs into a model config, with model-specific kwargs taking precedence.

    Args:
        model_config: A single model's config entry (must contain a "default_kwargs" key
                      whose value is a key into `defaults`).
        defaults: The top-level "default_kwargs" mapping from model_configs.json.

    Returns:
        A new model config dict with merged kwargs.
    
    Example:
    >>> model_config = {
    ...     "default_kwargs": "qwen-vl-sft-short",
    ...     "kwargs": {
    ...         "temperature": 0.7
    ...     }
    ... }
    >>> defaults = {
    ...     "qwen-vl-sft-short": {
    ...         "temperature": 0.6
    ...     }
    ... }
    >>> _apply_default_kwargs(model_config, defaults)
    {
    ...     "default_kwargs": "qwen-vl-sft-short",
    ...     "kwargs": {
    ...         "temperature": 0.7
    ...     }
    ... }
    """
    defaults_key = model_config["default_kwargs"] # e.g., "qwen-vl-sft-short"
    merged_kwargs = defaults[defaults_key].copy() # e.g., {"temperature": 0.6}
    merged_kwargs.update(model_config["kwargs"]) # e.g., {"temperature": 0.7}
    merged_config = model_config.copy() # e.g., {"default_kwargs": "qwen-vl-sft-short", "kwargs": {"temperature": 0.7}}
    merged_config["kwargs"] = merged_kwargs
    return merged_config # e.g., {"default_kwargs": "qwen-vl-sft-short", "kwargs": {"temperature": 0.7}}


def load_all_model_configs(path: Path | str = "configs/model_configs.json") -> dict:
    """
    This function loads all model configs from the given path and give a shorthand to model config mapping.
    """
    path = Path(path)
    with open(path, "r") as f:
        model_configs = json.load(f)
    shorthands_to_configs = {}
    for model_config in model_configs["models"].values():
        if "default_kwargs" in model_config:
            model_config = _apply_default_kwargs(model_config, model_configs["default_kwargs"])

        # For Gemini models, validate and convert thinking_level to extra_body
        shorthand = model_config["shorthand"]
        if "gemini" in shorthand.lower():
            model_name = model_config.get("kwargs", {}).get("model_name", "")
            thinking_level = model_config.get("kwargs", {}).get("thinking_level")
            if thinking_level is not None:
                _validate_gemini_thinking_level(shorthand, model_name, thinking_level)
                # Convert thinking_level to extra_body (nested structure required by Gemini API)
                model_config["kwargs"]["extra_body"] = {
                    "extra_body": {
                        "google": {
                            "thinking_config": {
                                "thinking_level": thinking_level,
                                "include_thoughts": True
                            }
                        }
                    }
                }
                # Remove thinking_level from kwargs since it's now in extra_body
                del model_config["kwargs"]["thinking_level"]

        shorthands_to_configs[model_config["shorthand"]] = model_config
    return shorthands_to_configs

def select_client_config(model, model_config: dict | None = None):
    """
    This function selects an API key and base URL from the environment variables.

    For VLLM-hosted models, if VLLM_BASE_URL is not set, the port will be automatically
    inferred from the model_config (from configs/model_configs.json).

    Args:
        model: The model name (can be shorthand or full name)
        model_config: Optional model configuration dict containing 'hosting' and 'port' keys.
                     If not provided and VLLM_BASE_URL is not set, the function will attempt
                     to load the config from model_configs.json.
    """
    model = model.lower()

    if "gpt" in model:
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
        }
    elif "qwen" in model or "gemma" in model or "olmo" in model:
        base_url = os.getenv("VLLM_BASE_URL")

        # If VLLM_BASE_URL is not set, try to infer from model_config
        if base_url is None:
            port = None
            if model_config is not None and "port" in model_config:
                port = model_config["port"]
            else:
                # Try to load config from model_configs.json
                try:
                    all_configs = load_all_model_configs()
                    # Search by shorthand or model name
                    for shorthand, config in all_configs.items():
                        if shorthand.lower() == model or model in config.get("kwargs", {}).get("model_name", "").lower():
                            if "port" in config:
                                port = config["port"]
                                break
                except Exception:
                    pass  # If loading fails, port remains None

            if port is not None:
                base_url = f"http://localhost:{port}/v1"
                print(f"Using VLLM server at {base_url} (inferred from model_configs.json)")
        else:
            print(f"Using VLLM server at {base_url} (from environment variable)")

        return {
            "api_key": os.getenv("VLLM_API_KEY"),
            "base_url": base_url,
        }
    elif "gemini" in model:
        return {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "base_url": os.getenv("GEMINI_BASE_URL"),
        }
    else:
        raise ValueError(f"Model {model} not supported")


def get_qwen_completion_kwargs(enable_thinking: bool):
    """
    This function returns the parameters for the OpenAI API.
    """
    if enable_thinking:
        # For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0 (the default setting in generation_config.json). 
        # DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            # "top_k": 20,
            # "min_p": 0,
        }
    else:
        # For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
        return {
            "temperature": 0.7,
            "top_p": 0.8,
            # "top_k": 20,
            # "min_p": 0,
        }

def get_completion_kwargs(model: str, enable_thinking: bool, model_config: dict | None = None):
    """
    This function returns the parameters for the OpenAI API for qwen and other models

    Args:
        model: The model name
        enable_thinking: Enable or disable thinking mode (for Qwen models)
        model_config: Model configuration dict from model_configs.json.
                      For Gemini models, extra_body is already prepared by load_all_model_configs.
    """
    model = model.lower()
    if "qwen" in model:
        qwen_params = get_qwen_completion_kwargs(enable_thinking)
        qwen_params['extra_body'] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        return qwen_params
    elif "gemini" in model:
        params = {
            "temperature": 1.0, "top_p": 0.95
        }
        # extra_body is prepared by load_all_model_configs
        if model_config and "kwargs" in model_config:
            extra_body = model_config["kwargs"].get("extra_body")
            if extra_body:
                params['extra_body'] = extra_body
        return params
    else:
        return {}

def parse_completion_content(message: str):
    """
    The content between the <think> and </think> tags from the message
    is the reasoning of the model, after that it's the results
    """
    if "<think>" and "</think>" not in message:
        return {"reasoning": "", "message": message}
    
    # first, split by <think> and take the content after it
    think_start = message.find("<think>")
    think_end = message.find("</think>")
    start_token_len = len("<think>")
    end_token_len = len("</think>")

    if think_end == -1:
        # assume no thinking
        reasoning = ""
        message = message.strip()
        return {"reasoning": reasoning, "message": message}
    elif think_start == -1:
        # <think> tag is not present, but </think> is present, so we take the content before </think>
        reasoning = message[0 : think_end].strip()
        message = message[think_end + end_token_len :].strip()
        return {"reasoning": reasoning, "message": message}
    else:
        reasoning = message[think_start + start_token_len : think_end].strip()
        message = message[think_end + end_token_len :].strip()
        return {"reasoning": reasoning, "message": message}


def shorthand_to_model(shorthand: str):
    """
    This function converts a shorthand to a model name.
    """
    shorthands_to_configs = load_all_model_configs()
    if shorthand not in shorthands_to_configs:
        raise ValueError(f"Unknown shorthand: {shorthand}")
    return shorthands_to_configs[shorthand]["kwargs"]["model_name"]