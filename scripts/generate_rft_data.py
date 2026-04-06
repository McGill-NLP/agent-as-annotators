import base64
from copy import deepcopy
import os
from pathlib import Path
import orjson

from tqdm import tqdm


def encode_image(img_path):
    ext = Path(img_path).suffix[1:]
    if ext in ['jpg', 'jpeg']:
        ext = 'jpeg'
    elif ext == 'png':
        ext = 'png'
    else:
        ext = 'jpeg'  # default to jpeg
    
    # Read the actual image file bytes
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    
    b64_code = base64.b64encode(img_bytes).decode('utf-8')
    encoded_img = f"data:image/{ext};base64,{b64_code}"

    return encoded_img

def remove_axtree(record):
    """
    Remove everything after `## AXTree:` and before `## Focused element:`
    """
    record = deepcopy(record)
    modified_record = []

    for r in record:
        text = r['content'][0]['text']
        start_idx = text.find("## AXTree:")
        end_idx = text.find("## Focused element:")
        modified_text = text[:start_idx] + text[end_idx:]
        r['content'][0]['text'] = modified_text
        modified_record.append(r)
    return modified_record

def generate_rft_data(
    cleaned_base_dir,
    model_name,
    benchmark="websynth",
    base_save_dir="trajectories/rft_data",
    delete_existing_train_file=None,
):
    # load from trajectories/cleaned and prepare the data for sft
    cleaned_parent_dir = cleaned_base_dir / benchmark / model_name
    save_dir = Path(base_save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    train_file = save_dir / "train.jsonl"
    train_no_axtree_file = save_dir / "train_no_axtree.jsonl"

    num_successful = 0
    num_total = 0

    print(f"Found {len(list(cleaned_parent_dir.glob('**/*.json')))} cleaned files for {model_name} in: {cleaned_parent_dir}")

    # if both train files exist, skip
    if not (train_file.exists() and train_no_axtree_file.exists()):
        print(f"No training files found at {save_dir}")
        delete_existing_train_file = False
    elif delete_existing_train_file is None:
        if train_file.exists():
            print(f"Train file found at {train_file}")
        elif train_no_axtree_file.exists():
            print(f"Train no axtree file found at {train_no_axtree_file}")
        
        # only prompt if the train files do not exist
        if not train_file.exists() and not train_no_axtree_file.exists():
            delete_existing_train_file_resp = input(f"Do you want to delete existing train files for {model_name}? (y/n): ").strip().lower()
            if delete_existing_train_file_resp == 'y':
                delete_existing_train_file = True
            elif delete_existing_train_file_resp == 'n':
                delete_existing_train_file = False
            else:
                raise ValueError("Invalid input. Please enter 'y' or 'n'.")
        else:
            delete_existing_train_file = False
    else:
        print(f"delete_existing_train_file was set in parameters to: {delete_existing_train_file}")
        if delete_existing_train_file in ['y', 'yes', 'true', '1', True, 1]:
            print(f"Deleting existing train files for {model_name}")
            delete_existing_train_file = True
        else:
            print(f"Not deleting existing train files for {model_name}")
            delete_existing_train_file = False

    if delete_existing_train_file:
        print(f"Deleting existing train files for {model_name}")
        if train_file.exists():
            os.remove(train_file)
        if train_no_axtree_file.exists():
            os.remove(train_no_axtree_file)
    else:
        print(f"Not deleting existing train files for {model_name}")

    pbar = tqdm(list(cleaned_parent_dir.glob("**/*.json")))
    for cleaned_path in pbar:
        num_total += 1
        # load the cleaned file
        with open(cleaned_path, "r") as f:
            cleaned = orjson.loads(f.read())

        success = cleaned['summary_info']['cum_reward'] > 0
        
        # if the trajectory is unsuccessful, skip
        if not success:
            continue
        
        num_successful += 1

        for step in cleaned["steps"]:
            msgs = step["chat_messages"]
            prefix = f"Skipping step {step['num']} of {cleaned_path.name} because it's not a"
            if len(msgs) != 3:
                tqdm.write(f"{prefix} valid message (expected 3 messages, got {len(msgs)})")
                continue
            
            sysp, user, asst = msgs
            if sysp["role"] != "system":
                tqdm.write(f"{prefix} system message")
                continue
            if user["role"] != "user":
                tqdm.write(f"{prefix} user message")
                continue
            if asst["role"] != "assistant":
                tqdm.write(f"{prefix} assistant message")
                continue
            
            if isinstance(user['content'], str):
                # in the first case, we have a text-only message
                # we need to convert the string to a list
                user['content'] = [
                    {"type": "text", "text": user['content']},
                    {"type": "image_url", "image_url": {"url": os.path.relpath(step['screenshot_path'], start='.')}}
                ]
                img_rec = user['content'][1]
            else:
                # if the user message is a list, we need to check if the second element is an image
                # then replace the image url with the relative path
                img_rec = user['content'][1]
                if img_rec['type'] != 'image_url':
                    # skip if the assistant message is not an image
                    tqdm.write(f"{prefix} image message")
                    continue
                
                # load image from step['screenshot_path']
                # img_rec['image_url']['url'] = encode_image(step['screenshot_path'])
                img_rec['image_url']['url'] = os.path.relpath(step['screenshot_path'], start='.')

            record = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sysp['content']}],
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': user['content'][0]['text']}, 
                        {'type': 'image', 'image': img_rec['image_url']['url']}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": asst['content']}],
                },
            ]
            
            with open(save_dir / "train.jsonl", "a") as f:
                f.write(orjson.dumps(record).decode('utf-8') + "\n")
            
            record_without_axtree = remove_axtree(record)
            
            with open(save_dir / "train_no_axtree.jsonl", "a") as f:
                f.write(orjson.dumps(record_without_axtree).decode('utf-8') + "\n")
            
        pbar.set_description(f"successful: {num_successful} / {num_total}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate RFT data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cleaned_base_dir", type=str, default="trajectories/cleaned", help="The base directory where the cleaned trajectories are stored",
    )
    parser.add_argument(
        "--base_save_dir", type=str, default="trajectories/rft_data", help="The base directory where the RFT data will be saved",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="The model name to use",
    )
    parser.add_argument(
        "--benchmark", type=str, default="websynth", help="The benchmark to use",
    )
    parser.add_argument(
        "--delete_existing_train_file", type=str, default=None, help="Delete existing train files? (y/n)",
    )
    args = parser.parse_args()
    generate_rft_data(
        cleaned_base_dir=Path(args.cleaned_base_dir),
        base_save_dir=Path(args.base_save_dir),
        model_name=args.model_name,
        benchmark=args.benchmark,
        delete_existing_train_file=args.delete_existing_train_file,
    )
