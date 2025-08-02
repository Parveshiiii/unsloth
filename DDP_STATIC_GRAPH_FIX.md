# DDP Static Graph Compatibility Fix for Gradient Checkpointing

## Problem

When using Unsloth with distributed training and gradient checkpointing, you may encounter this error:

```
RuntimeError: Your training graph has changed in this iteration, e.g., one parameter is unused in first iteration, but then got used in the second iteration. this is not compatible with static_graph set to True.
```

## Root Cause

The error occurs because:

1. **DDP Static Graph Optimization**: PyTorch's DistributedDataParallel (DDP) enables static graph optimization to improve performance
2. **Gradient Checkpointing Variability**: Gradient checkpointing can cause slight variations in the computation graph between iterations
3. **Incompatibility**: Static graph optimization requires an identical computation graph every iteration, but gradient checkpointing can violate this assumption

## Solution

Unsloth now automatically detects when gradient checkpointing is being used and disables DDP static graph optimization to ensure compatibility.

### Automatic Detection

The trainer automatically detects gradient checkpointing in these scenarios:
- `use_gradient_checkpointing = "unsloth"` in PEFT configuration
- Standard PyTorch gradient checkpointing enabled
- Unsloth-specific gradient checkpointing patterns
- Environment variable `UNSLOTH_USE_GRADIENT_CHECKPOINTING=1`

### Expected Output

When the fix is applied, you'll see these messages:

```
Unsloth: Gradient checkpointing detected - disabling DDP static graph to prevent graph change errors
Unsloth: This ensures compatibility between gradient checkpointing and DDP
```

If no gradient checkpointing is detected:

```
Unsloth: Enabled DDP static graph optimization (no gradient checkpointing detected)
```

## Manual Control

If automatic detection doesn't work or you need manual control:

### Force Disable Static Graph for Gradient Checkpointing
```bash
export UNSLOTH_DISABLE_DDP_STATIC_GRAPH_FOR_GRAD_CHECKPOINT=1
```

### Completely Disable Static Graph Optimization
```bash
export UNSLOTH_DISABLE_DDP_STATIC_GRAPH=1
```

### Force Enable Gradient Checkpointing Detection
```bash
export UNSLOTH_USE_GRADIENT_CHECKPOINTING=1
```

## Training Configuration

Ensure your training arguments are configured correctly:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # ... other arguments ...
    ddp_find_unused_parameters=False,  # Critical for stability
    # Don't manually set ddp_static_graph when using gradient checkpointing
)
```

## Additional Fixes

This update also includes:
- Fixed deprecated `torch.cuda.amp.autocast` → `torch.amp.autocast('cuda')`
- Enhanced gradient checkpointing detection for Unsloth-specific implementations
- Improved DDP reducer initialization for better autograd hook management

## Verification

To verify the fix is working:
1. Look for the gradient checkpointing detection messages in your training logs
2. Training should proceed without the "static_graph" RuntimeError
3. If issues persist, try the manual environment variable controls above

## Related Issues

- [PyTorch DDP Static Graph Documentation](https://pytorch.org/docs/stable/ddp_comm_hooks.html#static-graph)
- [PyTorch Issue #62719](https://github.com/pytorch/pytorch/issues/62719)
