#!/usr/bin/env python3
"""
Test script to validate enhanced DDP gradient checkpointing fix.

Usage:
    # Single GPU test
    python test_ddp_fix.py
    
    # Multi-GPU DDP test
    torchrun --nproc_per_node=2 test_ddp_fix.py
"""

import os
import torch
import torch.distributed as dist

def test_ddp_static_graph_setup():
    """Test that UnslothTrainer correctly sets up DDP static graph and handles autograd hooks."""
    
    # Check if we're in a distributed environment
    is_distributed = (
        os.environ.get("LOCAL_RANK") is not None or
        os.environ.get("WORLD_SIZE") is not None or
        os.environ.get("RANK") is not None
    )
    
    print(f"Distributed environment detected: {is_distributed}")
    
    if is_distributed:
        # Initialize distributed training
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        
        print(f"Rank {local_rank}: Using device {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Single GPU mode: Using device {device}")
    
    # Import unsloth after distributed setup
    from unsloth import FastLanguageModel, UnslothTrainer
    from trl import SFTConfig
    from datasets import Dataset
    
    # Load a small model for testing
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True,
        # Enable multi-GPU support if distributed
        enable_multi_gpu = is_distributed,
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    
    # Create a minimal dataset for testing
    dataset = Dataset.from_dict({
        "text": [
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n2+2 equals 4.<|im_end|>",
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>",
        ] * 10  # Repeat to have enough data
    })
    
    # Create UnslothTrainer (this should setup DDP static graph automatically)
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 512,
        dataset_num_proc = 1,
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 2,
            warmup_steps = 1,
            max_steps = 3,  # Very few steps for testing
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            output_dir = "/tmp/test_ddp_output",
            # DDP specific settings
            ddp_find_unused_parameters = False,  # Important for avoiding autograd hook errors
            report_to = [],  # Disable wandb/tensorboard logging
        ),
    )
    
    print("UnslothTrainer created successfully")
    
    # Test that DDP static graph setup doesn't cause errors
    try:
        # This should trigger the lazy DDP setup
        trainer._setup_ddp_static_graph_lazy(trainer.model)
        print("DDP static graph setup completed without errors")
    except Exception as e:
        print(f"DDP static graph setup failed: {e}")
        return False
    
    # Test reducer preparation
    try:
        trainer._prepare_ddp_reducer_for_training(trainer.model)
        print("DDP reducer preparation completed without errors")
    except Exception as e:
        print(f"DDP reducer preparation failed: {e}")
        return False
    
    # Run a few training steps to test gradient checkpointing
    try:
        print("Starting training test...")
        trainer.train()
        print("Training completed successfully!")
        return True
    except RuntimeError as e:
        error_msg = str(e)
        if "Expected to mark a variable ready only once" in error_msg:
            print(f"DDP 'parameter marked ready twice' error still occurs: {e}")
            return False
        elif "expect_autograd_hooks_" in error_msg:
            print(f"DDP 'expect_autograd_hooks_' error still occurs: {e}")
            return False
        else:
            print(f"Different training error occurred: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error during training: {e}")
        return False

if __name__ == "__main__":
    success = test_ddp_static_graph_setup()
    
    if success:
        print("\n✅ Test PASSED: Enhanced DDP gradient checkpointing fix works!")
        print("Both 'parameter marked ready twice' and 'expect_autograd_hooks_' errors are handled.")
    else:
        print("\n❌ Test FAILED: DDP gradient checkpointing issue persists")
    
    # Clean up distributed process group if initialized
    if dist.is_initialized():
        dist.destroy_process_group()