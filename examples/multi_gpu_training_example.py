#!/usr/bin/env python3
"""
Example script demonstrating Unsloth multi-GPU training.

This script shows how to use Unsloth with multiple GPUs for distributed training.
"""

# Example 0: Auto-detection multi-GPU training (NEW - improved auto-detection)
def example_auto_detection_multi_gpu():
    """Example of auto-detection multi-GPU training."""
    print("Example 0: Auto-detection multi-GPU training")
    print("-" * 50)
    
    import os
    
    # Enable multi-GPU training via environment variable
    os.environ["UNSLOTH_ENABLE_MULTIGPU"] = "1"
    
    # Import unsloth - it will now auto-detect and setup multi-GPU
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer, SFTConfig
    
    print(f"Available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    # Load model - multi-GPU will be auto-detected and configured
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        # No need to specify enable_multi_gpu=True, it's auto-detected from environment
        # device_map will be automatically set to "auto" for multi-GPU
    )
    
    print("Model loaded successfully with auto-detected multi-GPU configuration")
    
    # Continue with normal training setup...
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    
    print("LoRA adapters configured for multi-GPU training")
    return model, tokenizer

# Example 1: Single-process multi-GPU training (uses multiple GPUs in one process)
def example_single_process_multi_gpu():
    """Example of single-process multi-GPU training."""
    print("Example 1: Single-process multi-GPU training")
    print("-" * 50)
    
    # Import unsloth
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    
    # Load model with multi-GPU support
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        enable_multi_gpu = True,  # Enable multi-GPU support
        device_map = "auto",      # Automatically distribute across GPUs
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    # Dataset
    dataset = load_dataset("alpaca", split = "train")
    
    # Training
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    
    trainer.train()


# Example 2: Distributed Data Parallel (DDP) training
def example_distributed_training():
    """Example script for distributed training with torchrun."""
    print("Example 2: Distributed training setup")
    print("-" * 50)
    print("To run with distributed training, use:")
    print("torchrun --nproc_per_node=2 your_training_script.py")
    print("")
    print("Your training script should contain:")
    print('''
import os
import torch
import torch.distributed as dist
from unsloth import FastLanguageModel, UnslothTrainer
from trl import SFTConfig

# Unsloth will automatically detect distributed environment
# and initialize accordingly when using UnslothTrainer

# Load model - each process will handle one GPU
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
    enable_multi_gpu = True,  # Enable multi-GPU support
    # device_map will be automatically set to current GPU
)

# Setup LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Use UnslothTrainer which has built-in distributed training support
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = your_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
        # DDP specific settings
        ddp_find_unused_parameters = False,
    ),
)

trainer.train()
''')


# Example 3: Environment variable configuration
def example_environment_setup():
    """Example of environment variable setup."""
    print("Example 3: Environment variable configuration")
    print("-" * 50)
    print("You can enable multi-GPU training using environment variables:")
    print("")
    print("export UNSLOTH_ENABLE_MULTIGPU=1")
    print("python your_training_script.py")
    print("")
    print("Or for distributed training:")
    print("export UNSLOTH_ENABLE_MULTIGPU=1")
    print("torchrun --nproc_per_node=2 your_training_script.py")
    print("")


def main():
    """Main function to demonstrate examples."""
    print("=" * 60)
    print("Unsloth Multi-GPU Training Examples")
    print("=" * 60)
    print("")
    
    example_environment_setup()
    print("")
    
    try:
        # New auto-detection example
        print("Running auto-detection example...")
        example_auto_detection_multi_gpu()
        print("Auto-detection example completed successfully!")
    except Exception as e:
        print(f"Auto-detection example failed (expected in environments without GPUs): {e}")
    print("")
    
    example_single_process_multi_gpu()
    print("")
    
    example_distributed_training()
    print("")
    
    print("=" * 60)
    print("Notes:")
    print("- NEW: Auto-detection now works by setting UNSLOTH_ENABLE_MULTIGPU=1")
    print("- NEW: Improved distributed training initialization")
    print("- Single-process multi-GPU (device_map='auto') works best for inference")
    print("- Distributed training (DDP) is recommended for training large models")
    print("- Unsloth automatically detects your environment and configures accordingly")
    print("- Make sure all GPUs have sufficient VRAM for your model")
    print("=" * 60)


if __name__ == "__main__":
    main()