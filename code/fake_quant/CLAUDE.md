# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoEQuant is a post-training quantization (PTQ) framework specifically designed for Mixture-of-Experts (MoE) large language models. It addresses two key challenges in MoE quantization:
- **Inter-expert imbalance**: Uneven sample distribution across experts
- **Intra-expert imbalance**: Varying sample-expert correlations due to MoE's aggregation mechanism

## Build and Setup

```bash
# Install the package (from code/ directory)
pip install -e .

# Install fast-hadamard-transform (required dependency)
# Clone to third-party/ directory and compile
```

## Running Quantization

Main entry point is `fake_quant/main.py`. Example command:

```bash
CUDA_VISIBLE_DEVICES=0 python fake_quant/main.py \
    --model /path/to/model \
    --w_bits 4 \
    --fp32_had \
    --rotate \
    --AGQ_GPTQ \
    --EBSS_calib --calib_path ./EBSS_data/ebss_file.jsonl \
    --quant_test \
    --save_qmodel_path ./output.pth
```

Key flags:
- `--AGQ_GPTQ`: Use Affinity-Guided Quantization with GPTQ
- `--EBSS_calib`: Use Expert-Balanced Self-Sampling calibration data
- `--rotate`: Apply Hadamard rotation to weights
- `--quant_test`: Run evaluation after quantization
- `--load_qmodel_path`: Load a pre-quantized model for evaluation

## Architecture

### Core Components

**main.py**: Entry point orchestrating model loading, quantization, and evaluation pipeline.

**model_utils.py**: Model loading and type detection. Supports multiple MoE architectures:
- Qwen MoE (1.5, 2, 3, 3-Next) via `_is_qwen_moe_model()`
- DeepSeek (V1, V2) via `_is_deepseek_model()`
- Mixtral via local `mixtral_model/`
- Uses `device_map='auto'` for multi-GPU support

**gptq_utils_moe.py**: GPTQ quantization with MoE-specific extensions:
- `GPTQ` class: Standard GPTQ with Hessian accumulation
- `add_batch_score()`: Incorporates routing scores (AGQ technique)
- `add_batch_shared_score()`: Handles shared experts

**quant_layers/**: Quantized layer implementations
- `quant_layer.py`: `QuantDecoderLayer`, `QuantAttention`, `QuantMoeBlock`, `QuantMLP`
- `quant_ops.py`: Quantized operations (linear, matmul, softmax, etc.)
- `quantizer.py`: Low-level quantization logic

### Model-Specific Implementations

Each MoE architecture has its own modeling file:
- `deepseek_moe_16b_chat/`: DeepSeek V1 model
- `mixtral_model/`: Mixtral model
- `qwen_moe_14b_chat/`: Qwen MoE model

These contain full model implementations with custom attention and MoE routing.

### Evaluation

**evaluation/**: Benchmark evaluation modules
- `evaluate_mmlu.py`, `evaluate_humaneval.py`, `evaluate_gsm8k.py`
- `eval_lm.py`: General LM evaluation via lm_eval harness

**lm_eval/**: Local copy of lm-evaluation-harness for task evaluation

## Key Quantization Parameters

From `utils.py` parser:
- `--w_bits`: Weight bit-width (default 16)
- `--a_bits`: Activation bit-width (default 16)
- `--v_bits`, `--k_bits`: KV-cache quantization
- `--w_groupsize`, `--a_groupsize`: Quantization group sizes
- `--nsamples`: Number of calibration samples for GPTQ

## EBSS Data

Pre-computed Expert-Balanced Self-Sampling calibration data in `EBSS_data/`:
- `ebss_qwen15_*.jsonl`: Qwen 1.5 MoE
- `ebss_qwen3_*.jsonl`: Qwen 3 MoE
- `EBSS_mixtral.jsonl`: Mixtral
- `EBSS_deepseek.jsonl`: DeepSeek

## Multi-Model Compatibility Notes

When modifying `QuantDecoderLayer` or `QuantAttention`:
- Qwen3/Qwen3-Next use `position_embeddings` (cos, sin tuple) instead of `position_ids`
- Qwen3 expects single tensor return from decoder layers; Qwen2 expects tuple
- DeepSeek V2 and Mixtral use their own local model implementations, not quantized wrappers
- Check `model_utils._is_qwen_moe_model()` and similar functions for model type detection
