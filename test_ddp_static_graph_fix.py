#!/usr/bin/env python3
"""
Test script to verify DDP static graph fix for gradient checkpointing compatibility.

This script demonstrates that the trainer correctly detects gradient checkpointing
and disables static graph optimization to prevent runtime errors.
"""

import os
import tempfile
import torch
from unittest.mock import Mock, patch


def test_gradient_checkpointing_detection():
    """Test that gradient checkpointing detection works correctly."""
    
    # Import the trainer module
    try:
        from unsloth.trainer import _setup_ddp_static_graph
        print("✅ Successfully imported _setup_ddp_static_graph")
    except ImportError as e:
        print(f"❌ Failed to import trainer module: {e}")
        return False
    
    # Mock a model with gradient checkpointing enabled
    mock_model = Mock()
    mock_model.gradient_checkpointing = True
    
    # Mock DDP model
    mock_ddp_model = Mock()
    mock_ddp_model._static_graph = False
    mock_ddp_model.find_unused_parameters = False
    
    # Mock the environment to simulate distributed training
    env_vars = {
        'LOCAL_RANK': '0',
        'WORLD_SIZE': '2'
    }
    
    with patch.dict(os.environ, env_vars):
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('unsloth.trainer._find_ddp_model', return_value=mock_ddp_model):
                # Test that gradient checkpointing is detected and static graph is disabled
                result = _setup_ddp_static_graph(mock_model)
                
                if result is False:
                    print("✅ Gradient checkpointing correctly detected - static graph disabled")
                    return True
                else:
                    print("❌ Gradient checkpointing not detected - static graph enabled incorrectly")
                    return False


def test_no_gradient_checkpointing():
    """Test that static graph is enabled when no gradient checkpointing is detected."""
    
    try:
        from unsloth.trainer import _setup_ddp_static_graph
    except ImportError as e:
        print(f"❌ Failed to import trainer module: {e}")
        return False
    
    # Mock a model without gradient checkpointing
    mock_model = Mock()
    mock_model.gradient_checkpointing = False
    
    # Mock DDP model
    mock_ddp_model = Mock()
    mock_ddp_model._static_graph = False
    mock_ddp_model.find_unused_parameters = False
    
    env_vars = {
        'LOCAL_RANK': '0',
        'WORLD_SIZE': '2'
    }
    
    with patch.dict(os.environ, env_vars):
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('unsloth.trainer._find_ddp_model', return_value=mock_ddp_model):
                # Test that static graph is enabled when no gradient checkpointing
                result = _setup_ddp_static_graph(mock_model)
                
                if result is True:
                    print("✅ No gradient checkpointing detected - static graph enabled")
                    return True
                else:
                    print("❌ Static graph not enabled when no gradient checkpointing")
                    return False


def test_environment_variable_override():
    """Test that environment variables correctly override behavior."""
    
    try:
        from unsloth.trainer import _setup_ddp_static_graph
    except ImportError as e:
        print(f"❌ Failed to import trainer module: {e}")
        return False
    
    mock_model = Mock()
    
    # Test UNSLOTH_DISABLE_DDP_STATIC_GRAPH=1
    env_vars = {
        'UNSLOTH_DISABLE_DDP_STATIC_GRAPH': '1',
        'LOCAL_RANK': '0',
        'WORLD_SIZE': '2'
    }
    
    with patch.dict(os.environ, env_vars):
        with patch('torch.distributed.is_initialized', return_value=True):
            result = _setup_ddp_static_graph(mock_model)
            if result is False:
                print("✅ Environment variable UNSLOTH_DISABLE_DDP_STATIC_GRAPH correctly disables static graph")
            else:
                print("❌ Environment variable override not working")
                return False
    
    # Test UNSLOTH_DISABLE_DDP_STATIC_GRAPH_FOR_GRAD_CHECKPOINT=1
    env_vars = {
        'UNSLOTH_DISABLE_DDP_STATIC_GRAPH_FOR_GRAD_CHECKPOINT': '1',
        'LOCAL_RANK': '0',
        'WORLD_SIZE': '2'
    }
    
    with patch.dict(os.environ, env_vars):
        with patch('torch.distributed.is_initialized', return_value=True):
            result = _setup_ddp_static_graph(mock_model)
            if result is False:
                print("✅ Environment variable UNSLOTH_DISABLE_DDP_STATIC_GRAPH_FOR_GRAD_CHECKPOINT correctly disables static graph")
                return True
            else:
                print("❌ Gradient checkpointing environment variable override not working")
                return False


def main():
    """Run all tests."""
    print("🧪 Testing DDP Static Graph Fix for Gradient Checkpointing")
    print("=" * 60)
    
    tests = [
        ("Gradient Checkpointing Detection", test_gradient_checkpointing_detection),
        ("No Gradient Checkpointing", test_no_gradient_checkpointing),
        ("Environment Variable Override", test_environment_variable_override),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The DDP static graph fix is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    main()
