# DDP Gradient Checkpointing Fix

## Problem

When using Unsloth with distributed training (`torchrun --nproc_per_node=N`), users may encounter one of these errors:

```
RuntimeError: Expected to mark a variable ready only once. This error is caused by one of the following reasons: 
1) Use of a module parameter outside the `forward` function. 
2) Reused parameters in multiple reentrant backward passes.
Parameter at index 498 with name base_model.model.model.layers.35.mlp.gate_proj.lora_A.default.weight has been marked as ready twice.
```

```
RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/reducer.cpp":1633, please report a bug to PyTorch.
```

## Root Cause

These errors occur because:

1. **Gradient Checkpointing Re-execution**: Unsloth uses gradient checkpointing to save memory by re-computing forward passes during the backward pass.

2. **DDP Parameter Tracking**: PyTorch's DistributedDataParallel (DDP) tracks when each parameter is used during the backward pass to synchronize gradients across processes.

3. **Autograd Hook Conflicts**: DDP relies on autograd hooks to track parameter usage, but gradient checkpointing can interfere with the normal hook registration and execution process.

4. **Conflict**: When gradient checkpointing re-runs the forward pass during backward, it can cause the same LoRA parameters to be "used" multiple times, violating DDP's expectation that each parameter is marked ready exactly once per iteration.

## Solution

The fix implements the following changes in `UnslothTrainer`:

### 1. DDP Static Graph Optimization

```python
def _setup_ddp_static_graph(self):
    """Setup DDP static graph to fix gradient checkpointing issues."""
    # Find DDP-wrapped model and call _set_static_graph()
    ddp_model._set_static_graph()
```

The `_set_static_graph()` method tells DDP that the computation graph structure is static (doesn't change between iterations), which is true for most fine-tuning scenarios. This allows DDP to optimize its parameter tracking and handle gradient checkpointing correctly.

### 2. DDP Reducer Preparation

```python
def _prepare_ddp_reducer_for_training(self):
    """Prepare DDP reducer to avoid expect_autograd_hooks_ errors."""
    # Force DDP reducer initialization and autograd hook setup
    reducer._rebuild_buckets()
```

This ensures that DDP's internal reducer component is properly initialized with all necessary autograd hooks before the first training step, preventing internal assertion failures.

### 3. Enhanced Parameter Validation

The fix now includes additional checks for:
- `find_unused_parameters` setting (recommends False for better performance)
- DDP reducer state validation
- Early autograd hook registration
- Parameter registration verification

### 2. Robust DDP Model Detection

```python
def _find_ddp_model(self, model):
    """Recursively search for DDP-wrapped model in the model hierarchy."""
    # Check multiple levels of nesting: model.module, model.base_model, etc.
```

Models can be wrapped in multiple layers (PEFT, Accelerate, DDP), so we recursively search the model hierarchy to find the actual DDP-wrapped model.

### 3. Lazy Setup with Reducer Preparation

```python
def training_step(self, model, inputs, num_items_in_batch=None):
    """Override training_step to handle DDP gradient checkpointing issues."""
    self._setup_ddp_static_graph_lazy(model)
    self._prepare_ddp_reducer_for_training(model)
    return super().training_step(model, inputs, num_items_in_batch)
```

The DDP setup is performed lazily just before the first training step, ensuring that all model wrapping (by Accelerate, PEFT, etc.) has been completed. Additionally, the reducer is properly prepared to prevent autograd hook errors.

## Usage

### Option 1: Automatic SFTTrainer Patching (Recommended)

When you import unsloth, SFTTrainer is automatically patched with DDP support:

```python
import unsloth  # This automatically patches SFTTrainer with DDP support
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    # ... other args
    args=SFTConfig(
        # ... training args
        ddp_find_unused_parameters=False,  # Recommended for better DDP performance
    )
)
trainer.train()  # Now works correctly with distributed training!
```

### Option 2: Explicit UnslothTrainer Usage

You can still use UnslothTrainer explicitly if preferred:

```python
from unsloth import UnslothTrainer

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    # ... other args
    args=SFTConfig(
        # ... training args
        ddp_find_unused_parameters=False,  # Recommended for better DDP performance
    )
)
trainer.train()  # Works correctly with distributed training
```

### Before (causes DDP errors):

```python
from trl import SFTTrainer  # Without importing unsloth first

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    # ... other args
)
trainer.train()  # Causes "parameter marked ready twice" error
```

## Running Distributed Training

```bash
# Multi-GPU training
torchrun --nproc_per_node=4 your_training_script.py

# Or with Accelerate
accelerate launch --multi_gpu your_training_script.py
```

## Safety

The fix is safe because:

1. **Static Graph Assumption**: Fine-tuning scenarios typically have static computation graphs (same model architecture, same forward pass every iteration).

2. **Conditional Application**: The fix only applies when DDP is detected and only affects the DDP synchronization behavior.

3. **Graceful Fallback**: If the fix cannot be applied, training continues with warning messages but may still encounter the original error.

4. **No Performance Impact**: The static graph optimization can actually improve DDP performance by reducing synchronization overhead.

## Testing

Run the test script to validate the fix:

```bash
# Single GPU test
python test_ddp_fix.py

# Multi-GPU DDP test  
torchrun --nproc_per_node=2 test_ddp_fix.py
```

## References

- [PyTorch DDP Static Graph Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel._set_static_graph)
- [Gradient Checkpointing Documentation](https://pytorch.org/docs/stable/checkpoint.html)
- [Related PyTorch Issues](https://github.com/pytorch/pytorch/issues/62719)