# Unsloth Multi-GPU Training Support

This document describes the new multi-GPU training support in Unsloth, allowing you to train models on multiple GPUs for faster training and larger model support.

## Features

- **Automatic multi-GPU detection**: Unsloth automatically detects available GPUs and configures accordingly
- **Multiple training modes**: Support for both single-process multi-GPU and distributed data parallel (DDP) training
- **Backward compatibility**: Single-GPU training remains the default for maximum compatibility
- **Easy configuration**: Enable with a simple parameter or environment variable

## Quick Start

### 1. Enable Multi-GPU Training (NEW - Improved)

#### Method 1: Environment Variable (Recommended)
```bash
export UNSLOTH_ENABLE_MULTIGPU=1
python your_training_script.py
```

The system will now automatically:
- Detect available GPUs
- Configure optimal device mapping
- Set up distributed training environment if needed
- Provide helpful guidance for distributed training setup

#### Method 2: Function Parameter
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    enable_multi_gpu=True,  # Enable multi-GPU support
    device_map="auto",      # Automatically distribute across GPUs (optional, auto-detected)
)
)
```

### 2. Single-Process Multi-GPU Training

Best for models that fit in total GPU memory:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch

# Load model with automatic GPU distribution
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    enable_multi_gpu=True,
    device_map="auto",  # Distribute model across available GPUs
)

# Setup LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Train with SFTTrainer (automatically gets DDP support when unsloth is imported)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=60,
        learning_rate=2e-4,
        output_dir="outputs",
    ),
)

trainer.train()
```

### 3. Distributed Data Parallel (DDP) Training

Best for large models and maximum training speed:

```bash
# Launch with torchrun for 2 GPUs
export UNSLOTH_ENABLE_MULTIGPU=1
torchrun --nproc_per_node=2 your_training_script.py
```

Your training script:
```python
from unsloth import FastLanguageModel, UnslothTrainer
from trl import SFTConfig

# Unsloth automatically detects distributed environment
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    enable_multi_gpu=True,
)

# Setup LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Use SFTTrainer - automatically gets DDP support when unsloth is imported
# Or use UnslothTrainer explicitly if preferred
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=60,
        learning_rate=2e-4,
        output_dir="outputs",
        ddp_find_unused_parameters=False,
    ),
)

trainer.train()
```

## Configuration Options

### Environment Variables

- `UNSLOTH_ENABLE_MULTIGPU`: Set to "1" to enable multi-GPU support
- `LOCAL_RANK`: Automatically set by torchrun for distributed training
- `WORLD_SIZE`: Automatically set by torchrun for distributed training

### Function Parameters

- `enable_multi_gpu`: Boolean to enable/disable multi-GPU support
- `device_map`: Device mapping strategy
  - `"sequential"`: Default single-GPU behavior
  - `"auto"`: Automatic distribution across available GPUs
  - `"cuda:0"`, `"cuda:1"`, etc.: Specific GPU assignment

## Training Modes Comparison

| Mode | Use Case | Memory | Speed | Setup | Trainer |
|------|----------|---------|-------|-------|---------|
| Single-GPU | Small models, limited hardware | Low | Baseline | Default | `SFTTrainer` |
| Single-Process Multi-GPU | Medium models, inference | Medium | 1.5-2x | `device_map="auto"` | `SFTTrainer` (auto-patched) |
| Distributed (DDP) | Large models, maximum speed | High | 2-4x | `torchrun` | `SFTTrainer` (auto-patched) or `UnslothTrainer` |

**Note**: When you import unsloth, `SFTTrainer` is automatically patched with DDP support to avoid gradient checkpointing issues.

## Performance Tips

1. **Memory Optimization**:
   - Use `load_in_4bit=True` for better memory efficiency
   - Adjust `per_device_train_batch_size` based on GPU memory
   - Use gradient checkpointing: `use_gradient_checkpointing="unsloth"`

2. **Speed Optimization**:
   - Use `bf16=True` on supported hardware
   - Set appropriate `gradient_accumulation_steps`
   - Use `optim="adamw_8bit"` for memory efficiency

3. **Distributed Training**:
   - Ensure all GPUs have similar memory capacity
   - Use `ddp_find_unused_parameters=False` for better performance
   - Balance batch size across devices

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable 4-bit quantization: `load_in_4bit=True`

2. **Multi-GPU Training Not Working (Fixed in this version)**:
   - **Symptom**: "Num GPUs = 4" shown but training uses only 1 GPU
   - **Solution**: Set `UNSLOTH_ENABLE_MULTIGPU=1` environment variable
   - **Auto-fix**: System now automatically detects and configures multi-GPU
   - **Verification**: Look for "Multi-GPU training enabled" message in output
   - **Alternative**: Use explicit `enable_multi_gpu=True` parameter

3. **Distributed Training Hangs**:
   - Check network connectivity between nodes
   - Verify all processes can access the same dataset
   - Ensure consistent environment across all processes

4. **Model Loading Errors**:
   - Verify all GPUs have sufficient memory
   - Check CUDA compatibility
   - Try `device_map="sequential"` for debugging

### Debug Commands

```bash
# Check GPU availability
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Test multi-GPU configuration
python -c "
import os
os.environ['UNSLOTH_ENABLE_MULTIGPU'] = '1'
from unsloth.models._utils import get_multi_gpu_config
print(get_multi_gpu_config())
"

# Verify distributed setup
torchrun --nproc_per_node=2 -c "
import torch.distributed as dist
print(f'Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}')
"
```

### DDP Gradient Checkpointing Issues (FIXED)

**Issue**: When using distributed training with gradient checkpointing, you might encounter:
```
RuntimeError: Expected to mark a variable ready only once. 
Parameter base_model.model.model.layers.X.mlp.gate_proj.lora_A.default.weight has been marked as ready twice.
```

**Solution**: Import unsloth before using SFTTrainer (automatic fix) or use UnslothTrainer explicitly:

```python
# ✅ Option 1: Automatic fix (recommended)
import unsloth  # This patches SFTTrainer automatically
from trl import SFTTrainer
trainer = SFTTrainer(...)

# ✅ Option 2: Explicit UnslothTrainer usage
from unsloth import UnslothTrainer
trainer = UnslothTrainer(...)

# ❌ This can cause DDP issues:
from trl import SFTTrainer  # Without importing unsloth first
trainer = SFTTrainer(...)
```

**Technical Details**: 
- When unsloth is imported, `SFTTrainer` is automatically patched with DDP static graph optimization
- This tells PyTorch that the model structure doesn't change during training
- Safe for fine-tuning scenarios and improves DDP performance
- Can be disabled with `UNSLOTH_DISABLE_DDP_STATIC_GRAPH=1` if needed

**Test the fix**:
```bash
# Run the included test
python test_ddp_fix.py

# Test with multiple GPUs
torchrun --nproc_per_node=2 test_ddp_fix.py
```

For more details, see `DDP_GRADIENT_CHECKPOINTING_FIX.md`.

## Examples

See the `examples/multi_gpu_training_example.py` file for complete working examples.

## Requirements

- PyTorch with CUDA support
- Multiple NVIDIA GPUs with CUDA Compute Capability 7.0+
- Sufficient GPU memory for your model
- For distributed training: Same CUDA version across all GPUs

## Backward Compatibility

All existing single-GPU code continues to work without changes. Multi-GPU support is opt-in via the `enable_multi_gpu` parameter or environment variable.