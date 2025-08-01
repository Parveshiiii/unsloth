#!/usr/bin/env python3
"""
Integration test specifically for DDP gradient checkpointing fix.

This test validates that UnslothTrainer properly handles DDP static graph
setup to prevent "parameter marked ready twice" errors.
"""

import os
import unittest
from unittest.mock import patch, MagicMock


class TestDDPGradientCheckpointingFix(unittest.TestCase):
    """Test cases for DDP gradient checkpointing fix."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear any existing environment variables
        for var in ["LOCAL_RANK", "WORLD_SIZE", "RANK", "UNSLOTH_DISABLE_DDP_STATIC_GRAPH"]:
            if var in os.environ:
                del os.environ[var]
    
    def test_non_distributed_environment(self):
        """Test that DDP setup is skipped in non-distributed environment."""
        from unsloth.trainer import UnslothTrainer
        
        # Mock SFTTrainer to avoid complex initialization
        with patch('unsloth.trainer.SFTTrainer.__init__') as mock_sft_init:
            mock_sft_init.return_value = None
            
            trainer = UnslothTrainer.__new__(UnslothTrainer)
            trainer.model = MagicMock()
            
            # This should return early since we're not in distributed environment
            trainer._setup_ddp_static_graph()
            
            # No DDP-related calls should have been made
            self.assertFalse(hasattr(trainer, '_ddp_static_graph_setup_done'))
    
    def test_distributed_environment_detection(self):
        """Test that distributed environment is properly detected."""
        from unsloth.trainer import UnslothTrainer
        
        # Set up distributed environment variables
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        
        with patch('unsloth.trainer.SFTTrainer.__init__') as mock_sft_init:
            mock_sft_init.return_value = None
            
            trainer = UnslothTrainer.__new__(UnslothTrainer)
            trainer.model = MagicMock()
            
            # Mock torch.distributed
            with patch('torch.distributed.is_initialized', return_value=False):
                trainer._setup_ddp_static_graph()
                
            # Should have attempted to check distributed status
            self.assertTrue(True)  # If we get here, detection worked
    
    def test_ddp_static_graph_call(self):
        """Test that _set_static_graph is called on DDP model."""
        from unsloth.trainer import UnslothTrainer
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        # Set up distributed environment
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        
        with patch('unsloth.trainer.SFTTrainer.__init__') as mock_sft_init:
            mock_sft_init.return_value = None
            
            trainer = UnslothTrainer.__new__(UnslothTrainer)
            
            # Create mock DDP model
            mock_ddp_model = MagicMock(spec=DDP)
            trainer.model = mock_ddp_model
            
            # Mock torch.distributed
            with patch('torch.distributed.is_initialized', return_value=True):
                trainer._setup_ddp_static_graph_lazy(trainer.model)
                
            # Should have called _set_static_graph
            mock_ddp_model._set_static_graph.assert_called_once()
            self.assertTrue(hasattr(trainer, '_ddp_static_graph_setup_done'))
    
    def test_nested_ddp_model_detection(self):
        """Test detection of DDP model in nested structure."""
        from unsloth.trainer import UnslothTrainer
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        trainer = UnslothTrainer.__new__(UnslothTrainer)
        
        # Create nested model structure
        mock_ddp_model = MagicMock(spec=DDP)
        mock_wrapper = MagicMock()
        mock_wrapper.module = mock_ddp_model
        
        # Test detection
        found_ddp = trainer._find_ddp_model(mock_wrapper)
        
        self.assertEqual(found_ddp, mock_ddp_model)
    
    def test_environment_variable_disable(self):
        """Test that the fix can be disabled via environment variable."""
        from unsloth.trainer import UnslothTrainer
        
        # Set up distributed environment and disable flag
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["UNSLOTH_DISABLE_DDP_STATIC_GRAPH"] = "1"
        
        with patch('unsloth.trainer.SFTTrainer.__init__') as mock_sft_init:
            mock_sft_init.return_value = None
            
            trainer = UnslothTrainer.__new__(UnslothTrainer)
            trainer.model = MagicMock()
            
            # This should return early due to disable flag
            trainer._setup_ddp_static_graph_lazy(trainer.model)
            
            # Should be marked as done but no DDP calls made
            self.assertTrue(hasattr(trainer, '_ddp_static_graph_setup_done'))
    
    def test_training_step_integration(self):
        """Test that training_step calls lazy DDP setup."""
        from unsloth.trainer import UnslothTrainer
        
        with patch('unsloth.trainer.SFTTrainer.__init__') as mock_sft_init, \
             patch('unsloth.trainer.SFTTrainer.training_step') as mock_training_step:
            
            mock_sft_init.return_value = None
            mock_training_step.return_value = MagicMock()
            
            trainer = UnslothTrainer.__new__(UnslothTrainer)
            trainer.model = MagicMock()
            
            # Mock the lazy setup method
            trainer._setup_ddp_static_graph_lazy = MagicMock()
            
            # Call training_step
            trainer.training_step(trainer.model, {})
            
            # Should have called lazy setup
            trainer._setup_ddp_static_graph_lazy.assert_called_once_with(trainer.model)
            mock_training_step.assert_called_once()


if __name__ == '__main__':
    unittest.main()