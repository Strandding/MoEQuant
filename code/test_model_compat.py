#!/usr/bin/env python
"""Test script to check model compatibility."""
import sys

def test_model_structure(model_path, model_name):
    """Test if a model can be loaded and check its structure."""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print('='*80)

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        print(f"✓ Config loaded successfully")
        print(f"  - Architecture: {config.architectures}")
        print(f"  - Model type: {config.model_type}")

        # Check MoE specific attributes
        moe_attrs = {}
        for attr in ['num_experts', 'n_routed_experts', 'num_experts_per_tok',
                     'moe_intermediate_size', 'n_shared_experts',
                     'shared_expert_intermediate_size']:
            if hasattr(config, attr):
                moe_attrs[attr] = getattr(config, attr)

        if moe_attrs:
            print(f"  - MoE attributes:")
            for k, v in moe_attrs.items():
                print(f"    - {k}: {v}")

        # Try to understand the layer structure by checking class name patterns
        arch_name = config.architectures[0] if config.architectures else "Unknown"
        print(f"  - Expected layer class: {arch_name.replace('ForCausalLM', 'DecoderLayer')}")

        return True, config

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Test both models."""
    models = [
        ("/dataset/common/pretrain_model/DeepSeek-V2-Lite", "DeepSeek-V2-Lite"),
        ("/dataset/common/pretrain_model/Qwen3-Next-80B-A3B-Instruct", "Qwen3-Next-80B"),
    ]

    results = {}
    for path, name in models:
        success, config = test_model_structure(path, name)
        results[name] = (success, config)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    for name, (success, config) in results.items():
        status = "✓ Compatible" if success else "✗ Incompatible"
        print(f"{name}: {status}")

        if success and config:
            arch = config.architectures[0] if config.architectures else "Unknown"
            # Check if it matches known patterns
            if 'Qwen' in arch and 'Moe' not in arch and 'Next' in arch:
                print(f"  ⚠️  Warning: {arch} is Qwen3-Next, not standard Qwen MoE")
            elif 'DeepseekV2' in arch:
                print(f"  ⚠️  Warning: {arch} is DeepSeek V2, different from V1")

if __name__ == '__main__':
    main()
