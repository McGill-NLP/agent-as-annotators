from typing import Any, Union

LastStepOutputType = Union[dict[str, Any], list[dict[str, Any]], Any]

def print_last_step_output(last_step_output: LastStepOutputType, func_name, padding_len=80):
    text = func_name + ': last_step_output keys'
    text_len = len(text)
    left_padding = (padding_len - text_len) // 2
    right_padding = padding_len - text_len - left_padding
    bottom_padding = padding_len

    print('='*left_padding + text + '='*right_padding)
    if last_step_output is not None:
        if isinstance(last_step_output, dict):
            print(last_step_output.keys())
        elif isinstance(last_step_output, list):
            print([item.keys() for item in last_step_output])
        else:
            print("Found unexpected type of last_step_output:", type(last_step_output))
    else:
        print("None")
    print('-'*bottom_padding)