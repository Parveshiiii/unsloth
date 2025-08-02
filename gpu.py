
# --- Multi-GPU DDP Training Fix for Unsloth ---
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Disable TorchDynamo to avoid recompile_limit errors
os.environ["UNSLOTH_ENABLE_MULTIGPU"] = "1"  # Enable Unsloth multi-GPU support
os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous CUDA operations for debugging
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # Enable distributed debugging

# Import unsloth first to automatically patch SFTTrainer with DDP support
import unsloth
from unsloth import FastLanguageModel
import torch

fourbit_models = [
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", # Qwen 14B 2x faster
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507",
    max_seq_length = 128000,   # Context length - can be longer, but uses more memory
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = False,     # A bit more accurate, uses 2x memory
    full_finetuning = False,  # We have full finetuning now!
    device_map=None,          # Let DDP handle device placement
    use_cache=False,          # Disable use_cache for compatibility with gradient checkpointing
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "True", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

# Data Prep
from datasets import load_dataset, concatenate_datasets
reasoning_dataset = load_dataset("HelpingAI/Dhanishtha-2.0-SUPERTHINKER", split = "train")
# another_dataset = load_dataset("Abhaykoul/DH2.0-1_8", split="train")
# merged_dataset = concatenate_datasets([reasoning_dataset, another_dataset])

# Function to convert to conversation format
def generate_conversation(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    conversations = []
    for instruction, output in zip(instructions, outputs):
        conversations.append([
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ])
    return {"conversations": conversations}

# Convert reasoning_dataset to conversation format using generate_conversation
reasoning_conversations_data = generate_conversation(reasoning_dataset)
reasoning_conversations = tokenizer.apply_chat_template(
    reasoning_conversations_data["conversations"],
    tokenize=False,
)

print(reasoning_conversations[3])

# Now let's see how long the dataset is:
print(len(reasoning_conversations))

# Use the full reasoning dataset for training
import pandas as pd
data = pd.Series(reasoning_conversations)
data.name = "text"

from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)

# Train the model using UnslothTrainer for better DDP support
import unsloth
from unsloth import UnslothTrainer
from trl import SFTConfig
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 800,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
        ddp_find_unused_parameters = False,  # Critical for DDP performance and stability
        bf16 = True,  # Use bfloat16 for better performance on H100
        dataloader_pin_memory = False,  # Helps with multi-GPU training
        gradient_checkpointing = True,  # Enable gradient checkpointing for DDP
        ddp_backend = "nccl",  # Use NCCL backend for multi-GPU
        ddp_bucket_cap_mb = 25,  # Reduce bucket size for better memory management
        save_strategy = "steps",  # Save checkpoints periodically
        output_dir = "./trainer_output",  # Output directory for checkpoints
    ),
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Let's train the model! To resume a training run, set trainer.train(resume_from_checkpoint = True)
trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Inference
messages = [
    {"role" : "system", "content" : "You are HelpingAI a emotional AI always answer my question in HelpingAI style"},
    {"role" : "user", "content" : "How many planets are there in the solar system?"},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
    enable_thinking = False, # Disable thinking
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 256, # Increase for longer outputs!
    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
    streamer = TextStreamer(tokenizer, skip_prompt = True),
    use_cache=False,  # Disable use_cache for generation as well
)

model.push_to_hub_merged("Vortexjr/Qwen3-open-dhanishta", tokenizer, save_method = "merged_16bit", token = "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")
