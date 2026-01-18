#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

print("Testing FLA availability...")
try:
    from transformers.utils.import_utils import is_flash_linear_attention_available
    print(f"is_flash_linear_attention_available: {is_flash_linear_attention_available()}")
except Exception as e:
    print(f"Error checking FLA: {e}")

print("\nTesting direct import...")
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM
    print(f"Direct import successful! Class: {Qwen3NextForCausalLM}")
except Exception as e:
    print(f"Direct import failed: {e}")

print("\nTesting AutoModel import...")
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("/dataset/common/pretrain_model/Qwen3-Next-80B-A3B-Instruct", trust_remote_code=True)
    print(f"Config loaded: {type(config).__name__}")
    print(f"Model type: {config.model_type}")
except Exception as e:
    print(f"Config load failed: {e}")
