#!/usr/bin/env python
"""Test script to verify Qwen3 model loading."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../fake_quant'))

import torch
import model_utils

class Args:
    """Mock args for testing."""
    def __init__(self):
        self.online_hadamard = False
        self.a_bits = 16
        self.w_bits = 2
        self.v_bits = 16
        self.k_bits = 16
        self.a_asym = False
        self.w_asym = False
        self.v_asym = False
        self.k_asym = False
        self.w_groupsize = -1
        self.a_groupsize = -1
        self.k_groupsize = -1
        self.v_groupsize = -1
        self.a_clip_ratio = 1.0
        self.k_clip_ratio = 1.0
        self.v_clip_ratio = 1.0
        self.a_dynamic_method = 'pertensor'

def test_qwen3_load():
    """Test loading Qwen3 model."""
    model_path = '/dataset/common/pretrain_model/Qwen3-30B-A3B-Base'
    args = Args()

    print(f"Loading model from {model_path}...")
    try:
        model = model_utils.get_qwen(model_path, None, args)
        print(f"✓ Successfully loaded model: {model.__class__.__name__}")
        print(f"✓ Model type detected: {model_utils.get_model_type(model).__name__}")
        print(f"✓ Number of layers: {len(model.model.layers)}")

        # Check first layer structure
        first_layer = model.model.layers[0]
        print(f"✓ First layer class: {first_layer.__class__.__name__}")
        print(f"✓ MLP class: {first_layer.mlp.__class__.__name__}")
        print(f"✓ Number of experts: {len(first_layer.mlp.experts)}")
        print(f"✓ Has shared_expert: {hasattr(first_layer.mlp, 'shared_expert')}")

        # Test model type checking functions
        print(f"✓ _is_qwen_moe_model check: {model_utils._is_qwen_moe_model(model)}")
        model_type = model_utils.get_model_type(model)
        print(f"✓ _is_qwen_moe_type check: {model_utils._is_qwen_moe_type(model_type)}")

        print("\n✓✓✓ All tests passed! ✓✓✓")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_qwen3_load()
    sys.exit(0 if success else 1)
