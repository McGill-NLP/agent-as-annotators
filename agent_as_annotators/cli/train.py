import json
import os
import sys
import shutil
import uuid
from datetime import datetime
import torch
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator
from tqdm import tqdm

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return int(os.environ.get("RANK", 0)) == 0

def print_main(*args, **kwargs):
    """Print only on the main process"""
    if is_main_process():
        print(*args, **kwargs)

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Qwen2.5-VL model with SFT')
    parser.add_argument('--config', type=str, default='configs/3b.json',
                       help='Path to configuration JSON file')
    return parser.parse_args()

def list_checkpoint_dirs(output_dir):
    """find all subdir that match the pattern "checkpoint-*"""
    p = Path(output_dir)
    return [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]

def find_latest_valid_checkpoint(output_dir):
    """Find latest checkpoint that has trainer_state.json."""
    if not os.path.exists(output_dir):
        return None
    candidates = []
    for ckpt_dir in list_checkpoint_dirs(output_dir):
        if (ckpt_dir / "trainer_state.json").exists():
            candidates.append(ckpt_dir)
    if not candidates:
        return None
    def step_num(path):
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1
    return max(candidates, key=step_num)

def update_config_architecture(config_path):
    """Update the architecture in config.json from FSDP to regular version"""
    with open(config_path, "r") as f:
        config = json.load(f)
    for i, arch in enumerate(config["architectures"]):
        if arch.startswith("FSDP"):
            config["architectures"][i] = arch.replace("FSDP", "")
    with open(config_path, "w") as f:
        json.dump(config, f)

def filter_by_token_length(dataset, processor, max_length, buffer_ratio=0.95):
    """
    Filter dataset to remove samples that exceed max_length tokens.

    For VLMs, we cannot use truncation because it may remove image placeholder tokens
    while the actual images are still provided, causing a mismatch error.
    Instead, we pre-filter samples that are too long.

    Args:
        dataset: List of examples with 'messages' key
        processor: The model processor for tokenization
        max_length: Maximum allowed token length
        buffer_ratio: Keep samples up to this ratio of max_length (default 0.97)
                     to account for image token expansion

    Returns:
        Filtered dataset
    """
    if not dataset:
        print_main("Warning: Empty dataset provided to filter_by_token_length")
        return dataset

    effective_max = int(max_length * buffer_ratio)
    filtered = []
    skipped = 0
    errors = 0

    is_interactive = sys.stdout.isatty()
    pbar = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc="Filtering by token length",
        disable=not (is_main_process() and is_interactive),
    )
    for i, example in pbar:
        try:
            # Apply chat template to get text
            text = processor.apply_chat_template(example['messages'], tokenize=False)
            # Tokenize without images to get approximate text token count
            tokens = processor.tokenizer(text, add_special_tokens=False)
            token_count = len(tokens['input_ids'])

            if token_count <= effective_max:
                filtered.append(example)
            else:
                skipped += 1
        except Exception as e:
            # If we can't process the sample, skip it to avoid crashes during training
            errors += 1
            if errors <= 5:  # Only print first 5 errors to avoid spam
                print_main(f"Warning: Failed to process sample {i}, skipping: {e}")

    total = len(dataset)
    kept = len(filtered)
    pct = 100 * kept / total if total > 0 else 0
    print_main(f"Filtered dataset: kept {kept}/{total} samples ({pct:.1f}%)")
    print_main(f"  Skipped {skipped} samples exceeding {effective_max} tokens (max_length={max_length} * buffer={buffer_ratio})")
    if errors > 0:
        print_main(f"  Skipped {errors} samples due to processing errors")

    if not filtered:
        raise ValueError(
            f"All {total} samples were filtered out! "
            f"Try increasing max_length (current: {max_length}) or check your data format."
        )

    return filtered

# Create a data collator to encode text and image pairs
def create_data_collator(processor, max_length=16384, mask_before_image_tokens=False):
    """
    If mask_before_image_tokens is True, we mask the image tokens in the labels before the image tokens.
    This is useful when the text content before the image may not be assistant response we want
    the model to learn, e.g. it may be the system prompt or the user prompt. Generally, in a single
    turn conversation scenario, the text content before the image is not part of the assistant response.
    """
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example['messages'], tokenize=False) 
            for example in examples
        ]  # Prepare texts for processing
        image_inputs = [process_vision_info(example['messages'])[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        # Note: truncation=True is NOT supported for VLMs as it can remove image tokens
        # Instead, we pre-filter the dataset to remove samples exceeding max_length
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding='max_length', max_length=max_length
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, AutoProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch
        
        if mask_before_image_tokens:
            image_tokens_indices = []
            for image_token_id in image_tokens:
                # Find all occurrences of this image token
                indices = torch.where(labels == image_token_id)[0]
                # append to image_tokens_indices
                image_tokens_indices.append(indices.cpu())
            # concatenate all the indices
            image_tokens_indices = torch.cat(image_tokens_indices)
            
            if image_tokens_indices.numel() > 0:
                # make everything before the last image token in the labels -100
                last_image_token_index = image_tokens_indices.max()
                labels[:last_image_token_index] = -100

        return batch  # Return the prepared batch
    
    return collate_fn

# Parse command line arguments and load configuration

def main():
    args = parse_args()
    config = load_config(args.config)

    print_main("=" * 60)
    print_main(f"Loaded config from {args.config}:")
    print_main("-" * 60)
    print_main(json.dumps(config, indent=2))
    print_main("=" * 60)

    # Extract configuration values
    max_length = config['max_length']
    model_id = config['model_id']
    data_path = config['data_path']
    save_steps = config['save_steps']
    learning_rate = config['learning_rate']
    logging_steps = config['logging_steps']
    num_train_epochs = config['num_train_epochs']
    batch_size_per_gpu = config['batch_size_per_gpu']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    max_grad_norm = config['max_grad_norm']
    warmup_ratio = config['warmup_ratio']
    weight_decay = config['weight_decay']
    seed = config['seed']
    resume_from_checkpoint = config.get('resume_from_checkpoint', "auto")
    resumed_job_id = config.get('resumed_job_id', None)
    save_total_limit = config.get('save_total_limit', 2)
    run_name = config.get('run_name', "Web-SFT")
    ckpt_dir = config.get('ckpt_dir', "checkpoints")

    job_id = resumed_job_id or os.environ.get("SLURM_JOB_ID", str(uuid.uuid4())[:10])
    model_name_slug = model_id.replace("/","-").replace(".","_")
    output_dir = f'{ckpt_dir}/{model_name_slug}-{run_name}-{job_id}'
    min_pixels = 3136
    max_pixels = 1080 * 1920
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    print_main(f"Output directory: {output_dir}")
    # Resolve resume_from_checkpoint safely to avoid missing trainer_state.json
    latest_valid_checkpoint = find_latest_valid_checkpoint(output_dir)
    if resume_from_checkpoint == "auto":
        if latest_valid_checkpoint is not None:
            resume_from_checkpoint = str(latest_valid_checkpoint)
            print_main(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            resume_from_checkpoint = False
            print_main(f"No valid checkpoint found in {output_dir}, starting from scratch")
    elif resume_from_checkpoint is True:
        if latest_valid_checkpoint is not None:
            resume_from_checkpoint = str(latest_valid_checkpoint)
            print_main(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            resume_from_checkpoint = False
            print_main(f"No valid checkpoint found in {output_dir}, starting from scratch")
    elif isinstance(resume_from_checkpoint, str):
        requested_ckpt = Path(resume_from_checkpoint)
        if not (requested_ckpt / "trainer_state.json").exists():
            if latest_valid_checkpoint is not None:
                resume_from_checkpoint = str(latest_valid_checkpoint)
                print_main(
                    "Requested checkpoint missing trainer_state.json; "
                    f"falling back to {resume_from_checkpoint}"
                )
            else:
                resume_from_checkpoint = False
                print_main(
                    "Requested checkpoint missing trainer_state.json and no valid "
                    f"checkpoint found in {output_dir}, starting from scratch"
                )

    with open(data_path, "r") as f:
        train_dataset = [{'messages': json.loads(line)} for line in f]

    # Filter out samples that exceed max_length to prevent VLM truncation errors
    print_main(f"Filtering dataset by token length (max_length={max_length})...")
    train_dataset = filter_by_token_length(train_dataset, processor, max_length)

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=output_dir,  # Directory to save the model
        num_train_epochs=num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=batch_size_per_gpu,  # Batch size for training
        per_device_eval_batch_size=batch_size_per_gpu,  # Batch size for evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,  # Steps to accumulate gradients
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=learning_rate,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=logging_steps,  # Steps interval for logging
        eval_steps=None,  # Steps interval for evaluation
        eval_strategy="no",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=save_steps,  # Steps interval for saving
        save_total_limit=save_total_limit,
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=False,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=max_grad_norm,  # Maximum norm for gradient clipping
        warmup_ratio=warmup_ratio,  # Ratio of total steps for warmup
        weight_decay=weight_decay,  # Weight decay for training
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to='none',  # Reporting tool for tracking metrics. "none" means we disable everything
        padding_free=False,  # works with fa2, could try later
        # Gradient checkpointing settings
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        seed=seed,  # Seed for reproducibility
        assistant_only_loss=False,  # I don't think this flag works
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    accelerator = Accelerator()
    # if this script was run with fsdp, we need to set the device map to the correct device
    if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
        print_main("Using FSDP's device map via accelerate")
        device_map = {"": accelerator.process_index}
    else:
        print_main("Using auto device map'")
        device_map = "auto"

    if "qwen2.5" in model_id.lower():
        print_main(f"Using Qwen2.5 family: {model_id}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
            )
    elif "qwen3" in model_id.lower():
        print_main(f"Using Qwen3 family: {model_id}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    else:
        print_main(f"Using non-Qwen2.5 family: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # type: ignore
        data_collator=create_data_collator(processor, max_length=max_length, mask_before_image_tokens=True),
        peft_config=None,
        processing_class=processor.tokenizer, # type: ignore
    )

    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    print_main("Training finished at", datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    # save processor
    processor.save_pretrained(output_dir)

    # save log history
    with open(os.path.join(output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f)
    print_main(f"Log history saved to {os.path.join(output_dir, 'log_history.json')}")

    # copy slurm logs to output dir, if they exist
    # find all files where the name contains <job_id> and copy them to output dir
    slurm_logs = [f for f in os.listdir("slurm_logs") if f.startswith(f"{job_id}")]
    for log_file in slurm_logs:
        log_file_with_timestamp = f"{log_file}-ended-at-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        shutil.copy(f"slurm_logs/{log_file}", os.path.join(output_dir, log_file_with_timestamp))
    print_main(f"Slurm logs copied to {output_dir}")

    # for each checkpoint dir, save the processor, and inside the config.json, replace key "architectures"
    # from FSDPQwen* to Qwen*
    failures = 0
    for ckpt_dir in [d for d in list_checkpoint_dirs(output_dir) if d.name.split("-")[-1].isdigit()]:
        try:
            processor.save_pretrained(ckpt_dir)
            update_config_architecture(os.path.join(ckpt_dir, "config.json"))
        except Exception as e:
            print_main(f"Failed to update config.json for {ckpt_dir}: {e}")
            failures += 1

    if failures > 0:
        print_main(f"Failed to update config.json for {failures} checkpoints")
    else:
        print_main("Finished updating config.json for all checkpoints without errors")

    # Create checkpoint-latest symlink pointing to the last checkpoint
    checkpoint_dirs = list_checkpoint_dirs(output_dir)
    # Filter out non-numeric checkpoints (e.g., checkpoint-latest symlink)
    checkpoint_dirs = [d for d in checkpoint_dirs if d.name.split("-")[-1].isdigit()]
    if checkpoint_dirs:
        # Sort by checkpoint number (extract number from checkpoint-<num>)
        checkpoint_dirs.sort(key=lambda d: int(d.name.split("-")[-1]))
        latest_checkpoint = checkpoint_dirs[-1]
        latest_symlink = Path(output_dir) / "checkpoint-latest"

        # Remove existing symlink if it exists
        if latest_symlink.is_symlink() or latest_symlink.exists():
            latest_symlink.unlink()

        # Create relative symlink
        latest_symlink.symlink_to(latest_checkpoint.name)
        print_main(f"Created symlink: {latest_symlink} -> {latest_checkpoint.name}")

    # # save the accelerator state since trainer.save_model doesn't work for fsdp
    # if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
    #     if accelerator.is_main_process:
    #         print(f"Saving on the main process {accelerator.process_index}")
    #         try:
    #             # save as model
    #             trainer.accelerator.save_model(model=model, save_directory=output_dir)
    #         except Exception as e:
    #             print(f"Error saving accelerator state: {e}")
    #             print("Using a different method to save the accelerator state")
    #             trainer.accelerator.save_state(output_dir=output_dir)
    #     else:
    #         print(f"Skipping saving on non-main process {accelerator.process_index}. Waiting for main process to save.")
    #         accelerator.wait_for_everyone()
    #         print("Main process has saved the model.")
    # else:
    #     trainer.save_model(output_dir=output_dir)



if __name__ == "__main__":
    main()
