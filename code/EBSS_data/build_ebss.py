#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EBSS (Expert-Balanced Self-Sampling) builder — paper-faithful workflow + engineering optimizations.

Engineering upgrades applied (minimal algorithm change):
1) Prefer native router/gate logits from model outputs if available; otherwise fallback to hooks.
2) NO GPU->CPU sync in routing capture (hooks keep tensors on GPU).
3) counts + sigma computed on GPU (vectorized scatter_add), avoid Python loops.
4) Sentence-level batching: generate K sentences in parallel, each with beam width w,
   so each step forwards batch = K*w tokens (fills H100).
5) Constant-shape attention_mask per step (no per-item loops).
6) bfloat16 weights + try FlashAttention2 + torch.inference_mode().
7) Keep EBSS "deferred imbalance": score uses sigma(parent), not sigma(parent||v).

Output jsonl: keep EXACT {"text": "..."} per line (as in your current writer).
"""

import os
import re
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from tqdm import tqdm

# Disable torch.compile to avoid Triton 3.x incompatibility with PyTorch 2.2.1
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'

import torch
import torch.nn as nn

# -----------------------------
# Suppress FLA/Triton warnings
# -----------------------------
import warnings
warnings.filterwarnings('ignore', message='Triton is not supported on current platform')
warnings.filterwarnings('ignore', message='Current Triton version')

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Pre-import Qwen3Next to ensure it's registered with AutoModel
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM
    # Manually register if not already registered
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    if hasattr(AutoModelForCausalLM, '_model_mapping'):
        if Qwen3NextConfig not in AutoModelForCausalLM._model_mapping._model_mapping:
            AutoModelForCausalLM._model_mapping._model_mapping[Qwen3NextConfig] = Qwen3NextForCausalLM
except Exception as e:
    print(f"[EBSS][WARN] Could not pre-register Qwen3NextForCausalLM: {e}")

try:
    from transformers.cache_utils import Cache, DynamicCache
except Exception:
    Cache = None
    DynamicCache = None

# Try to import Qwen3NextDynamicCache for special handling
Qwen3NextDynamicCache = None
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDynamicCache
except Exception:
    pass

# -----------------------------
# Compatibility patch for DeepSeek-V2 and other models using old cache API
# -----------------------------
if DynamicCache is not None:
    # Add get_usable_length as an alias for get_seq_length for backward compatibility
    if not hasattr(DynamicCache, 'get_usable_length'):
        def _get_usable_length_compat(self, seq_len=None, layer_idx=0):
            """Backward compatibility wrapper for get_usable_length -> get_seq_length."""
            # DeepSeek-V2 and older models call: get_usable_length(seq_len, layer_idx)
            # Newer transformers use: get_seq_length(layer_idx)
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = _get_usable_length_compat

# -----------------------------
# 1) Router/Gate tracker via hooks (NO CPU sync)
# -----------------------------
class ExpertTracker:
    """
    Hook candidate router/gate Linear layers. Keep captured outputs on GPU (detach only),
    and validate shapes at runtime. This avoids GPU->CPU sync.
    """

    def __init__(
        self,
        model: nn.Module,
        router_regex: str = r"(router|routing|gate|moe)",
        exclude_regex: str = r"(gate_proj|up_proj|down_proj)",
        max_router_out: int = 256,
        verbose: bool = True,
    ):
        self.model = model
        self.router_re = re.compile(router_regex, re.IGNORECASE) if router_regex else None
        self.exclude_re = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None
        self.max_router_out = max_router_out
        self.verbose = verbose

        self.layer_names: List[str] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        # collected per forward (GPU tensors)
        self.gate_outputs: List[torch.Tensor] = []

        self._register_hooks()

    def _register_hooks(self):
        # Special hook for DeepSeek-V2's MoEGate that computes logits from input
        def deepseek_gate_hook(_module, _inp, out):
            # DeepSeek-V2 MoEGate returns (topk_weight, topk_idx, aux_loss)
            # We recompute logits from input: F.linear(hidden_states, module.weight)
            import torch.nn.functional as F
            if len(_inp) > 0 and torch.is_tensor(_inp[0]):
                hidden_states = _inp[0]
                if hasattr(_module, 'weight') and torch.is_tensor(_module.weight):
                    h = hidden_states.shape[-1]
                    hidden_flat = hidden_states.view(-1, h)
                    logits = F.linear(hidden_flat.float(), _module.weight.float(), None)
                    # Reshape back to [bsz, seq_len, n_experts]
                    if hidden_states.ndim == 3:
                        bsz, seq_len, _ = hidden_states.shape
                        logits = logits.view(bsz, seq_len, -1)
                    self.gate_outputs.append(logits.detach())

        def hook_fn(_module, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            if not torch.is_tensor(out):
                return

            x = out
            if x.ndim == 0:
                return
            last_dim = x.shape[-1] if x.ndim >= 1 else 0
            if last_dim <= 1 or last_dim > self.max_router_out:
                return

            # Keep on GPU; detach only.
            self.gate_outputs.append(x.detach())

        for name, module in self.model.named_modules():
            # Support both nn.Linear and custom gate modules (like DeepSeek-V2's MoEGate)
            is_linear = isinstance(module, nn.Linear)
            is_custom_gate = (
                type(module).__name__ in ["MoEGate", "Gate", "Router", "Routing"] or
                (hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter) and
                 name.endswith('.gate') and 'mlp' in name)
            )

            if not (is_linear or is_custom_gate):
                continue
            if self.exclude_re is not None and self.exclude_re.search(name):
                continue
            if self.router_re is not None and not self.router_re.search(name):
                continue

            # Heuristic: router out_features should be small
            if is_linear:
                if getattr(module, "out_features", None) is not None:
                    if module.out_features <= 1 or module.out_features > self.max_router_out:
                        continue

            self.layer_names.append(name)
            # Use special hook for DeepSeek-V2's MoEGate
            if type(module).__name__ == "MoEGate":
                self._handles.append(module.register_forward_hook(deepseek_gate_hook))
            else:
                self._handles.append(module.register_forward_hook(hook_fn))

        if self.verbose:
            print(f"[EBSS] Hooked candidate router/gate Linear layers: {len(self.layer_names)}")
            if self.layer_names:
                preview = "\n".join(self.layer_names[:10])
                more = "" if len(self.layer_names) <= 10 else f"\n... (+{len(self.layer_names) - 10} more)"
                print(f"[EBSS] Example layer names:\n{preview}{more}")
            else:
                print("[EBSS][WARN] No router/gate layers detected. You may need to tune --router_regex/--exclude_regex/--max_router_out.")

    def clear(self):
        self.gate_outputs.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# -----------------------------
# 2) Candidate state (counts on GPU)
# -----------------------------
@dataclass
class Candidate:
    tokens: List[int]                      # generated tokens (excluding BOS)
    rs: float                              # cumulative logprob (Eq.11-like)
    counts: torch.Tensor                   # [L,E] int32 on GPU
    length: int                            # len(tokens)
    past_key_values: Optional[Any]       # HF cache on device (batch=1)
    next_logits: Optional[torch.Tensor]    # [vocab] on device


# -----------------------------
# 3) Device helpers + KV batching
# -----------------------------
def _get_input_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _is_hf_cache(pkv: Any) -> bool:
    """Check if pkv is a HF Cache object (by instance check or duck typing)."""
    if pkv is None:
        return False
    # Instance check
    if (Cache is not None) and isinstance(pkv, Cache):
        return True
    # Check for Qwen3NextDynamicCache
    if (Qwen3NextDynamicCache is not None) and isinstance(pkv, Qwen3NextDynamicCache):
        return True
    # Duck typing: check for Cache-like attributes/methods
    # This handles custom Cache classes (e.g., HybridCache in Qwen3-Next)
    if hasattr(pkv, "get_seq_length") and hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return True
    return False


def _is_qwen3_next_cache(pkv: Any) -> bool:
    """Check if pkv is a Qwen3NextDynamicCache (has conv_states/recurrent_states)."""
    if pkv is None:
        return False
    if (Qwen3NextDynamicCache is not None) and isinstance(pkv, Qwen3NextDynamicCache):
        return True
    # Duck typing: check for Qwen3Next-specific attributes
    if hasattr(pkv, "conv_states") and hasattr(pkv, "recurrent_states") and hasattr(pkv, "layer_types"):
        return True
    return False


def _safe_cat_tensors(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Safely concatenate tensors, handling None and empty tensors.
    For Qwen3NextDynamicCache, some layers have empty tensors (linear attention layers
    have empty key_cache/value_cache, while attention layers have empty conv_states/recurrent_states).
    """
    if not tensors:
        return None

    # Filter out None tensors
    valid_tensors = [t for t in tensors if t is not None]
    if not valid_tensors:
        return None

    # Check if all tensors are empty (numel == 0)
    non_empty_tensors = [t for t in valid_tensors if t.numel() > 0]
    if not non_empty_tensors:
        # All tensors are empty, return the first one as placeholder
        return valid_tensors[0]

    # If some are empty and some are not, we need to handle this carefully
    # For Qwen3NextDynamicCache: attention layers have real k/v but empty conv/recurrent
    # and linear layers have empty k/v but real conv/recurrent
    if len(non_empty_tensors) != len(valid_tensors):
        # Mixed case: some empty, some not
        # Return concatenated non-empty tensors
        return torch.cat(non_empty_tensors, dim=dim)

    return torch.cat(valid_tensors, dim=dim)


def _stack_qwen3_next_cache(pasts: List[Any]) -> Any:
    """
    Stack Qwen3NextDynamicCache objects across batch dimension.
    This cache has: key_cache, value_cache (for attention layers),
    conv_states, recurrent_states (for linear attention layers).
    """
    import copy

    if not pasts:
        raise ValueError("Empty past list")

    p0 = pasts[0]

    # Create new cache by deepcopy (Qwen3NextDynamicCache requires config in __init__)
    cache = copy.deepcopy(p0)

    n_layers = len(p0.key_cache)

    def _stack_tensors(tensor_list):
        """Stack tensors along batch dimension, handling None and empty tensors."""
        if not tensor_list:
            return None
        # Filter out None
        valid = [t for t in tensor_list if t is not None]
        if not valid:
            return None
        # Check if all are empty
        non_empty = [t for t in valid if t.numel() > 0]
        if not non_empty:
            # All tensors are empty, stack them by expanding batch dim
            # e.g., [(1, 0), (1, 0)] -> (2, 0)
            total_batch = sum(t.shape[0] for t in valid)
            new_shape = list(valid[0].shape)
            new_shape[0] = total_batch
            return valid[0].new_empty(new_shape)
        # All should be non-empty at this point for normal cat
        return torch.cat(non_empty, dim=0)

    # Stack key_cache and value_cache
    cache.key_cache = [_stack_tensors([p.key_cache[i] for p in pasts]) for i in range(n_layers)]
    cache.value_cache = [_stack_tensors([p.value_cache[i] for p in pasts]) for i in range(n_layers)]

    # Stack conv_states and recurrent_states
    cache.conv_states = [_stack_tensors([p.conv_states[i] for p in pasts]) for i in range(n_layers)]
    cache.recurrent_states = [_stack_tensors([p.recurrent_states[i] for p in pasts]) for i in range(n_layers)]

    return cache


def _repeat_qwen3_next_cache(past: Any, times: int) -> Any:
    """
    Repeat a batch=1 Qwen3NextDynamicCache into batch=times.
    """
    import copy

    if times <= 1:
        return past

    cache = copy.deepcopy(past)
    n_layers = len(past.key_cache)

    def _repeat_tensor(t, times):
        """Repeat a tensor along batch dimension, handling None and empty tensors."""
        if t is None:
            return None
        if t.numel() == 0:
            # For empty tensors, expand the first dimension (batch) only
            # e.g., (1, 0) -> (times, 0)
            new_shape = list(t.shape)
            new_shape[0] = times
            return t.new_empty(new_shape)
        # Normal repeat: repeat batch dimension
        return t.repeat(times, *([1] * (t.ndim - 1)))

    # Repeat key_cache and value_cache
    cache.key_cache = [_repeat_tensor(past.key_cache[i], times) for i in range(n_layers)]
    cache.value_cache = [_repeat_tensor(past.value_cache[i], times) for i in range(n_layers)]

    # Repeat conv_states and recurrent_states
    cache.conv_states = [_repeat_tensor(past.conv_states[i], times) for i in range(n_layers)]
    cache.recurrent_states = [_repeat_tensor(past.recurrent_states[i], times) for i in range(n_layers)]

    return cache


def _split_qwen3_next_cache(past_batched: Any, batch_size: int) -> List[Any]:
    """
    Split a batched Qwen3NextDynamicCache (batch=batch_size) into list of batch=1 caches.
    """
    import copy

    if batch_size <= 0:
        return []

    def _split_tensor(t, bi):
        """Split a tensor at batch index bi, handling None and empty tensors."""
        if t is None:
            return None
        if t.numel() == 0:
            # For empty tensors, return a batch=1 empty tensor with same other dims
            # e.g., (batch, 0) -> (1, 0)
            new_shape = list(t.shape)
            new_shape[0] = 1
            return t.new_empty(new_shape)
        return t[bi:bi+1].contiguous()

    pasts = []
    n_layers = len(past_batched.key_cache)

    for bi in range(batch_size):
        cache = copy.deepcopy(past_batched)

        # Split key_cache and value_cache
        cache.key_cache = [_split_tensor(past_batched.key_cache[i], bi) for i in range(n_layers)]
        cache.value_cache = [_split_tensor(past_batched.value_cache[i], bi) for i in range(n_layers)]

        # Split conv_states and recurrent_states
        cache.conv_states = [_split_tensor(past_batched.conv_states[i], bi) for i in range(n_layers)]
        cache.recurrent_states = [_split_tensor(past_batched.recurrent_states[i], bi) for i in range(n_layers)]

        pasts.append(cache)

    return pasts


def _tuple_to_dynamic_cache(past_tuple: tuple, reference_cache: Any = None) -> Any:
    """
    Convert a legacy tuple-based past_key_values to DynamicCache.
    Used for models that require Cache objects (e.g., Qwen3-Next).

    Args:
        past_tuple: tuple of (key, value) pairs per layer
        reference_cache: optional Cache object to use as template for type
    """
    if DynamicCache is None:
        return past_tuple

    if _is_hf_cache(past_tuple):
        return past_tuple

    import copy

    # Try to create a Cache object with the same type as reference_cache
    cache = None
    if reference_cache is not None and _is_hf_cache(reference_cache):
        cls = reference_cache.__class__
        # Method 1: Try to create empty instance and copy structure
        try:
            cache = cls()
        except Exception:
            pass

        # Method 2: If empty instantiation fails, try deepcopy and reset
        if cache is None:
            try:
                cache = copy.deepcopy(reference_cache)
                # Clear the cache contents
                if hasattr(cache, 'key_cache'):
                    cache.key_cache = []
                if hasattr(cache, 'value_cache'):
                    cache.value_cache = []
            except Exception:
                pass

    # Fallback to DynamicCache
    if cache is None:
        try:
            cache = DynamicCache()
        except Exception:
            return past_tuple

    # Ensure key_cache and value_cache exist
    if not hasattr(cache, 'key_cache'):
        cache.key_cache = []
    if not hasattr(cache, 'value_cache'):
        cache.value_cache = []

    seen_tokens = 0
    for layer_past in past_tuple:
        if layer_past is None:
            cache.key_cache.append(None)
            cache.value_cache.append(None)
        elif isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
            key_states, value_states = layer_past
            cache.key_cache.append(key_states)
            cache.value_cache.append(value_states)
            # Infer seen_tokens from key shape if possible
            if key_states is not None and hasattr(key_states, 'shape') and len(key_states.shape) >= 3:
                seen_tokens = max(seen_tokens, key_states.shape[-2])
        else:
            cache.key_cache.append(None)
            cache.value_cache.append(None)

    # Set _seen_tokens for models that require it
    if hasattr(cache, '_seen_tokens'):
        cache._seen_tokens = seen_tokens

    return cache


def _ensure_correct_cache_type(pkv: Any, reference_type: Any) -> Any:
    """
    Ensure past_key_values matches the expected type (Cache or tuple).
    If reference_type is a Cache and pkv is a tuple, convert it.
    If reference_type is a tuple and pkv is a Cache, keep it as Cache (model will handle).
    """
    if reference_type is None:
        return pkv

    # If reference is Cache and pkv is tuple, convert
    if _is_hf_cache(reference_type) and not _is_hf_cache(pkv):
        return _tuple_to_dynamic_cache(pkv, reference_cache=reference_type)

    return pkv


def _stack_past_key_values(pasts: List[Any]) -> Any:
    """
    Stack past_key_values across batch dimension.
    Supports:
      - New HF Cache (transformers.cache_utils.Cache / DynamicCache)
      - Qwen3NextDynamicCache (has conv_states/recurrent_states)
      - Legacy tuple(num_layers) of (k,v)
    """
    if not pasts:
        raise ValueError("Empty past list")

    p0 = pasts[0]

    # ---- Qwen3NextDynamicCache path ----
    if _is_qwen3_next_cache(p0):
        return _stack_qwen3_next_cache(pasts)

    # ---- New HF Cache path (Qwen3-MoE etc.) ----
    if _is_hf_cache(p0):
        cls = p0.__class__
        if hasattr(cls, "from_batch_splits"):
            try:
                # best: build batched cache from list of batch=1 caches
                return cls.from_batch_splits(pasts)
            except Exception:
                pass

        # Conservative fallback: if cache exposes key_cache/value_cache lists
        if hasattr(p0, "key_cache") and hasattr(p0, "value_cache"):
            import copy
            try:
                # Try to create new instance
                try:
                    cache = cls()
                except Exception:
                    # If empty instantiation fails, deepcopy and reset
                    cache = copy.deepcopy(p0)
                if hasattr(p0, "_seen_tokens"):
                    cache._seen_tokens = p0._seen_tokens
                cache.key_cache = [_safe_cat_tensors([p.key_cache[i] for p in pasts], dim=0) for i in range(len(p0.key_cache))]
                cache.value_cache = [_safe_cat_tensors([p.value_cache[i] for p in pasts], dim=0) for i in range(len(p0.value_cache))]
                return cache
            except Exception as e:
                print(f"[WARN] Cache stack failed, falling back to tuple conversion: {e}")
                pass

        # Last resort: convert all to tuples, stack, convert back
        if hasattr(p0, "key_cache") and hasattr(p0, "value_cache"):
            tuple_pasts = [tuple((k, v) for k, v in zip(p.key_cache, p.value_cache)) for p in pasts]
            stacked_tuple = _stack_past_key_values(tuple_pasts)
            return _tuple_to_dynamic_cache(stacked_tuple, reference_cache=p0)

        raise TypeError(f"Unsupported Cache type for stacking: {type(p0)}")

    # ---- Legacy tuple path ----
    n_layers = len(p0)
    stacked = []
    for li in range(n_layers):
        ks, vs = [], []
        has_none = False
        for p in pasts:
            layer_past = p[li]
            # Handle None or incomplete layer_past
            if layer_past is None:
                has_none = True
                break
            if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                k, v = layer_past
                if k is None or v is None:
                    has_none = True
                    break
                ks.append(k)
                vs.append(v)
            else:
                has_none = True
                break

        if has_none:
            # If any layer has None, keep as None
            stacked.append(None)
        else:
            stacked.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))
    return tuple(stacked)


def _repeat_past_key_values(past: Any, times: int) -> Any:
    """
    Repeat a batch=1 past into batch=times.
    Supports Cache, Qwen3NextDynamicCache, and legacy tuple.
    """
    if times <= 1:
        return past

    # Qwen3NextDynamicCache path (must check before _is_hf_cache as it also passes that check)
    if _is_qwen3_next_cache(past):
        return _repeat_qwen3_next_cache(past, times)

    # HF Cache path
    if _is_hf_cache(past):
        # Prefer built-in repeat if available
        if hasattr(past, "batch_repeat_interleave"):
            try:
                # repeats can be int or tensor-like; int works in recent HF
                return past.batch_repeat_interleave(times)
            except Exception:
                pass

        # Fallback: deep-copy + from_batch_splits
        import copy
        cls = past.__class__
        if hasattr(cls, "from_batch_splits"):
            try:
                splits = [past] + [copy.deepcopy(past) for _ in range(times - 1)]
                return cls.from_batch_splits(splits)
            except Exception:
                pass

        # Manual fallback for Cache objects: access key_cache and value_cache
        if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
            try:
                # Try to create new instance
                try:
                    new_cache = cls()
                except Exception:
                    # If empty instantiation fails, deepcopy and reset
                    new_cache = copy.deepcopy(past)

                # Copy metadata if available
                if hasattr(past, "_seen_tokens"):
                    new_cache._seen_tokens = past._seen_tokens
                # Repeat each layer's key and value tensors
                new_cache.key_cache = [k.repeat(times, 1, 1, 1) if k is not None else None
                                       for k in past.key_cache]
                new_cache.value_cache = [v.repeat(times, 1, 1, 1) if v is not None else None
                                         for v in past.value_cache]
                return new_cache
            except Exception as e:
                # If all else fails, convert to tuple, repeat, and convert back
                print(f"[WARN] Cache repeat failed, falling back to tuple conversion: {e}")
                pass

        # Last resort: convert to tuple, repeat, convert back
        if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
            tuple_past = tuple((k, v) for k, v in zip(past.key_cache, past.value_cache))
            repeated_tuple = _repeat_past_key_values(tuple_past, times)
            return _tuple_to_dynamic_cache(repeated_tuple, reference_cache=past)

        raise TypeError(f"Cache does not support repeat: {type(past)}")

    # Legacy tuple path
    out = []
    for layer_past in past:
        # Handle None or incomplete layer_past
        if layer_past is None:
            out.append(None)
            continue

        # Unpack k, v (handle case where layer_past might not be a tuple)
        if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
            k, v = layer_past
            # Handle None k or v
            if k is None or v is None:
                out.append((None, None))
            else:
                out.append((k.repeat(times, 1, 1, 1), v.repeat(times, 1, 1, 1)))
        else:
            # Unexpected format, keep as-is
            out.append(layer_past)
    return tuple(out)


def _split_past_key_values(past_batched: Any, batch_size: int) -> List[Any]:
    """
    Split a batched past (batch=batch_size) into list of batch=1 past.
    Supports Cache, Qwen3NextDynamicCache, and legacy tuple.
    """
    if batch_size <= 0:
        return []

    # Qwen3NextDynamicCache path (must check before _is_hf_cache as it also passes that check)
    if _is_qwen3_next_cache(past_batched):
        return _split_qwen3_next_cache(past_batched, batch_size)

    # HF Cache path
    if _is_hf_cache(past_batched):
        if hasattr(past_batched, "batch_split"):
            try:
                splits = past_batched.batch_split(full_batch_size=batch_size, split_size=1)
                return list(splits)
            except Exception:
                pass

        # Conservative fallback using batch_select_indices (in-place) + deepcopy
        import copy
        if hasattr(past_batched, "batch_select_indices"):
            try:
                out = []
                for bi in range(batch_size):
                    pc = copy.deepcopy(past_batched)
                    idx = torch.tensor([bi], device=next(iter(pc.key_cache)).device) if hasattr(pc, "key_cache") else torch.tensor([bi])
                    pc.batch_select_indices(idx)
                    out.append(pc)
                return out
            except Exception:
                pass

        # Manual fallback for Cache objects: slice key_cache and value_cache
        if hasattr(past_batched, "key_cache") and hasattr(past_batched, "value_cache"):
            try:
                cls = past_batched.__class__
                out = []
                for bi in range(batch_size):
                    # Try to create new instance
                    try:
                        new_cache = cls()
                    except Exception:
                        # If empty instantiation fails, deepcopy and reset
                        new_cache = copy.deepcopy(past_batched)
                    # Copy metadata if available
                    if hasattr(past_batched, "_seen_tokens"):
                        new_cache._seen_tokens = past_batched._seen_tokens
                    # Slice each layer's key and value tensors
                    new_cache.key_cache = [k[bi:bi+1].contiguous() if k is not None else None
                                          for k in past_batched.key_cache]
                    new_cache.value_cache = [v[bi:bi+1].contiguous() if v is not None else None
                                            for v in past_batched.value_cache]
                    out.append(new_cache)
                return out
            except Exception as e:
                print(f"[WARN] Cache split failed, falling back to tuple conversion: {e}")
                pass

        # Last resort: convert to tuple, split, convert back
        if hasattr(past_batched, "key_cache") and hasattr(past_batched, "value_cache"):
            tuple_past = tuple((k, v) for k, v in zip(past_batched.key_cache, past_batched.value_cache))
            split_tuples = _split_past_key_values(tuple_past, batch_size)
            return [_tuple_to_dynamic_cache(t, reference_cache=past_batched) for t in split_tuples]

        raise TypeError(f"Cache does not support split: {type(past_batched)}")

    # Legacy tuple path
    pasts = []
    for bi in range(batch_size):
        one = []
        for layer_past in past_batched:
            # Handle None or incomplete layer_past
            if layer_past is None:
                one.append(None)
                continue
            if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                k_t, v_t = layer_past
                if k_t is None or v_t is None:
                    one.append((None, None))
                else:
                    one.append((k_t[bi:bi + 1].contiguous(), v_t[bi:bi + 1].contiguous()))
            else:
                # Unexpected format, keep as-is
                one.append(layer_past)
        pasts.append(tuple(one))
    return pasts


# -----------------------------
# 3.5) Debug utilities for cache shape checking
# -----------------------------
def _shape_str(t):
    """Helper to format tensor shape info."""
    if t is None:
        return "None"
    if not torch.is_tensor(t):
        return f"{type(t).__name__}"
    return f"{list(t.shape)} {t.dtype} {t.device} numel={t.numel()}"


def debug_check_cache(pkv: Any, expect_B: int, tag: str, max_layers: int = 6, enabled: bool = False):
    """
    Print/check cache batch dimensions for debugging.
    For Qwen3NextDynamicCache: checks key_cache/value_cache/conv_states/recurrent_states.
    For HF Cache: checks key_cache/value_cache.
    For legacy tuple: checks (k,v) pairs.

    Args:
        pkv: The cache object to check
        expect_B: Expected batch size
        tag: Label for this checkpoint
        max_layers: Maximum number of layers to print (default 6 for brevity)
        enabled: Whether debug is enabled (default False)
    """
    if not enabled:
        return

    print(f"\n[DBG][{tag}] type={type(pkv).__name__} expect_B={expect_B}")

    def _check_tensor(t, name, li):
        if t is None:
            return
        if not torch.is_tensor(t):
            print(f"[DBG][{tag}] {name}[{li}] non-tensor: {type(t)}")
            return
        # batch dim usually dim0
        b = t.shape[0] if t.ndim >= 1 else None
        print(f"[DBG][{tag}] {name}[{li}] shape={list(t.shape)} numel={t.numel()}")
        if b is not None and b != expect_B:
            print(f"[DBG][{tag}][BAD] {name}[{li}] batch={b} != {expect_B}")

    # Qwen3-Next hybrid cache
    if _is_qwen3_next_cache(pkv):
        n = len(pkv.key_cache)
        for li in range(min(n, max_layers)):
            _check_tensor(pkv.key_cache[li], "key_cache", li)
            _check_tensor(pkv.value_cache[li], "value_cache", li)
            _check_tensor(pkv.conv_states[li], "conv_states", li)
            _check_tensor(pkv.recurrent_states[li], "recurrent_states", li)
        return

    # HF Cache style
    if _is_hf_cache(pkv) and hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        n = len(pkv.key_cache)
        for li in range(min(n, max_layers)):
            _check_tensor(pkv.key_cache[li], "key_cache", li)
            _check_tensor(pkv.value_cache[li], "value_cache", li)
        return

    # legacy tuple
    if isinstance(pkv, tuple):
        n = len(pkv)
        for li in range(min(n, max_layers)):
            layer = pkv[li]
            if layer is None:
                continue
            if isinstance(layer, (tuple, list)) and len(layer) == 2:
                k, v = layer
                _check_tensor(k, "k", li)
                _check_tensor(v, "v", li)
            else:
                print(f"[DBG][{tag}] layer[{li}] unexpected: {type(layer)}")
        return

    print(f"[DBG][{tag}] unknown cache format")


# -----------------------------
# 4) Sigma computation (GPU)
# -----------------------------
@torch.no_grad()
def sigma_from_counts_gpu(counts: torch.Tensor) -> torch.Tensor:
    """
    counts: [B,L,E] int32 on GPU
    returns: [B] float32 sigma
    """
    assert counts.dim() == 3
    B, L, E = counts.shape
    if E <= 1:
        return torch.zeros((B,), device=counts.device, dtype=torch.float32)

    counts_f = counts.float()
    T = counts_f.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B,L,1]
    freqs = counts_f / T
    mean = freqs.mean(dim=-1, keepdim=True)                # [B,L,1]
    var = ((freqs - mean) ** 2).sum(dim=-1) / float(E - 1) # [B,L]
    sig = torch.sqrt(var).mean(dim=-1)                     # [B]
    return sig


# -----------------------------
# 5) Routing logits extraction: native outputs preferred
# -----------------------------
def _extract_router_logits_from_outputs(out: Any) -> Optional[torch.Tensor]:
    """
    Try multiple common names in remote-code MoE implementations.
    Expected shape examples:
      [B, 1, E] or [B, S, E] or list/tuple of per-layer tensors.
    Return:
      - If tensor: returns tensor
      - If list/tuple of tensors: returns stacked list-like is handled elsewhere
    """
    # direct attributes
    for key in (
        "router_logits",
        "router_logit",
        "gating_logits",
        "gate_logits",
        "moe_router_logits",
        "moe_gate_logits",
    ):
        if hasattr(out, key):
            return getattr(out, key)

    # some models pack extras in a dict-like field
    if hasattr(out, "aux_loss") and hasattr(out, "router_logits"):
        return getattr(out, "router_logits")

    return None


def _infer_num_experts(model, gate_outs: List[torch.Tensor], max_router_out: int) -> int:
    # Try config first
    cfg = getattr(model, "config", None)
    for attr in ("num_experts", "n_experts", "num_local_experts", "moe_num_experts"):
        v = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(v, int) and v > 1 and v <= max_router_out:
            return int(v)

    # Fallback: inspect gate outputs
    expert_dims = []
    for g in gate_outs:
        if torch.is_tensor(g) and g.ndim >= 2:
            e = int(g.shape[-1])
            if 1 < e <= max_router_out:
                expert_dims.append(e)
    if expert_dims:
        # most common
        return max(set(expert_dims), key=expert_dims.count)
    return 1


# -----------------------------
# 6) Update counts (GPU scatter_add)
# -----------------------------
@torch.no_grad()
def update_counts_from_gate_outputs_batch_gpu(
    counts_b: torch.Tensor,         # [B,L,E] int32 on GPU (updated in-place)
    gate_outputs: List[torch.Tensor],
    moe_top_k: int,
):
    """
    gate_outputs: list length >= L, each tensor on GPU with shape:
      [B, 1, E] or [B, E] or [E]
    We increment top-k experts for the *last* position (one token step).
    """
    assert counts_b.dim() == 3
    B, L, E = counts_b.shape
    if B == 0 or E <= 1:
        return

    use_L = min(L, len(gate_outputs))
    for li in range(use_L):
        x = gate_outputs[li]
        if not torch.is_tensor(x):
            continue

        if x.ndim == 3:
            x = x[:, -1, :]  # [B,E']
        elif x.ndim == 2:
            pass             # [B,E']
        elif x.ndim == 1:
            x = x.unsqueeze(0).expand(B, -1)
        else:
            continue

        if x.shape[0] != B:
            continue
        E2 = x.shape[1]
        k = min(moe_top_k, E2)
        if k <= 0:
            continue

        topk_idx = torch.topk(x, k=k, dim=-1).indices  # [B,k]
        # If E2 != E, clamp indices into [0,E-1] (conservative)
        if E2 != E:
            topk_idx = topk_idx.clamp(0, E - 1)

        # Move topk_idx to the same device as counts_b (for multi-GPU models)
        if topk_idx.device != counts_b.device:
            topk_idx = topk_idx.to(counts_b.device)

        ones = torch.ones_like(topk_idx, dtype=counts_b.dtype)
        # counts_b[:, li, :] is [B,E]
        counts_b[:, li, :].scatter_add_(dim=1, index=topk_idx, src=ones)


# -----------------------------
# 7) Forward helpers (batched, cache-friendly)
# -----------------------------
@torch.no_grad()
def forward_step_batched(
    model,
    tracker: ExpertTracker,
    token_ids: torch.Tensor,            # [B,1] long on device
    attention_mask: torch.Tensor,       # [B, cur_len] long on device
    past_key_values: Optional[Any],   # batched past
    max_router_out: int,
    debug: bool = False,
) -> Tuple[torch.Tensor, Tuple, List[torch.Tensor]]:
    """
    One token step for a batch.
    Returns:
      next_logits: [B, vocab] on device
      new_past: past_key_values (batched) on device
      gate_outputs: list of tensors on device (per-layer), either from native outputs or hooks.
    """
    tracker.clear()
    try:
        out = model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )
    except Exception as e:
        print(f"[DBG][forward_step_batched] EXCEPTION: {repr(e)}")
        if past_key_values is not None:
            debug_check_cache(past_key_values, expect_B=token_ids.shape[0], tag="pkv_at_crash", enabled=True)
        raise
    logits = out.logits  # [B,1,V]
    next_logits = logits[:, -1, :].detach()
    new_past = out.past_key_values

    native = _extract_router_logits_from_outputs(out)
    gate_outs: List[torch.Tensor] = []

    if torch.is_tensor(native):
        # native might be [B,S,E] or [B,E]
        # normalize to list-per-layer is not always possible; assume single router for all layers is not correct.
        # If native is [B,1,E] and model has per-layer router outputs elsewhere, we won't have it.
        # Therefore: accept native only if it looks like a per-layer stack: [L,B,1,E] or [L,B,E].
        x = native
        if x.ndim == 4:
            # [L,B,S,E] or [B,L,S,E] — guess by small dim
            if x.shape[0] <= 128 and x.shape[-1] <= max_router_out:
                # treat as [L,B,S,E]
                for li in range(x.shape[0]):
                    gate_outs.append(x[li].detach())
            elif x.shape[1] <= 128 and x.shape[-1] <= max_router_out:
                # treat as [B,L,S,E]
                for li in range(x.shape[1]):
                    gate_outs.append(x[:, li].detach())
        elif x.ndim == 3 and x.shape[-1] <= max_router_out:
            # ambiguous: could be [B,S,E] single router — not per-layer. Use hook fallback.
            gate_outs = []
        elif x.ndim == 2 and x.shape[-1] <= max_router_out:
            gate_outs = []
        else:
            gate_outs = []

    if not gate_outs:
        gate_outs = tracker.gate_outputs[:]  # GPU tensors

    return next_logits, new_past, gate_outs


# -----------------------------
# 8) Scoring (objective switch) — deferred imbalance
# -----------------------------
def compute_score_or_cost(
    rs_new: float,
    l_new: int,
    sigma_parent: float,
    tau: float,
) -> float:
    # val = -(rs_new / l_new) + sigma_parent/tau
    return float(-(rs_new / float(max(1, l_new))) + (sigma_parent / float(tau)))


# -----------------------------
# 9) EBSS batched generation: K sentences in parallel
# -----------------------------
def generate_sent_batch_ebss(
    model,
    tokenizer,
    tracker: ExpertTracker,
    input_device: torch.device,
    seq_len: int,
    width_w: int,
    tau: float,
    moe_top_k: int,
    top_token_k: int,
    objective: str,                   # "min" or "max"
    sent_batch: int,                  # K
    banned_ids_device: torch.Tensor,
    banned_set: set,
    avoid_special: bool = True,
    avoid_eos_before_end: bool = True,
    base_seed: Optional[int] = None,
    max_router_out: int = 256,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    assert objective in ("min", "max")
    K = int(sent_batch)
    if K <= 0:
        raise ValueError("--sent_batch must be > 0")

    # -------------------------
    # Anti-repetition knobs
    # -------------------------
    repetition_penalty = 1.12
    no_repeat_ngram_size = 4
    max_repeat_run = 3

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError("Tokenizer has neither bos_token_id nor eos_token_id.")
    eos_id = tokenizer.eos_token_id

    special_ids = set(tokenizer.all_special_ids or [])
    vocab_size = int(getattr(tokenizer, "vocab_size", None) or model.get_input_embeddings().weight.shape[0])

    def _build_ngram_set(tokens: List[int], n: int) -> set:
        if n <= 1 or len(tokens) < n:
            return set()
        return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    def _would_repeat_ngram(tokens: List[int], v: int, n: int, ngram_set: set) -> bool:
        if n <= 1:
            return False
        if len(tokens) < n - 1:
            return False
        cand = tuple(tokens[-(n - 1):] + [v])
        return cand in ngram_set

    def _exceeds_repeat_run(tokens: List[int], v: int, max_run: int) -> bool:
        if max_run <= 0:
            return False
        run = 0
        for t in reversed(tokens):
            if t == v:
                run += 1
            else:
                break
        return run >= max_run

    # ---- Step 0: BOS init (once) ----
    tracker.clear()
    bos_in = torch.tensor([[bos_id]], device=input_device, dtype=torch.long)
    bos_mask = torch.ones((1, 1), device=input_device, dtype=torch.long)
    out0 = model(input_ids=bos_in, attention_mask=bos_mask, use_cache=True)
    past_bos_1 = out0.past_key_values
    next_logits_bos = out0.logits[:, -1, :].detach()  # [1,V]

    # ---- Per-sentence RNG (deterministic, no global coupling) ----
    rngs: List[random.Random] = []
    if base_seed is None:
        base_seed = 1234
    for i in range(K):
        rngs.append(random.Random(int(base_seed) + i * 1000003))

    # ---- First token: uniform random over allowed tokens ----
    first_tokens: List[int] = []
    for i in range(K):
        while True:
            v = rngs[i].randrange(0, vocab_size)
            if v in banned_set:
                continue
            if avoid_special and v in special_ids:
                continue
            if avoid_eos_before_end and eos_id is not None and v == eos_id and seq_len > 1:
                continue
            first_tokens.append(v)
            break

    # logp(first) from BOS distribution with banned ids masked
    logits0 = next_logits_bos[0].clone()
    if banned_ids_device.numel() > 0:
        logits0.index_fill_(0, banned_ids_device, -1e9)
    logp0 = torch.log_softmax(logits0, dim=-1)  # [V]
    rs_first = [float(logp0[v].item()) for v in first_tokens]

    # ---- Step 1: forward the first token in batch K ----
    # past_bos batched by repeating along batch dim
    # (repeat is safe; it's just the BOS cache)
    # past_bos = []
    # for (k, v) in past_bos_1:
    #     past_bos.append((k.repeat(K, 1, 1, 1), v.repeat(K, 1, 1, 1)))
    # past_bos = tuple(past_bos)

    # Checkpoint A: Check BOS cache before and after repeat
    debug_check_cache(past_bos_1, expect_B=1, tag="past_bos_1", enabled=debug)
    past_bos = _repeat_past_key_values(past_bos_1, K)
    debug_check_cache(past_bos, expect_B=K, tag="past_bos_repeated", enabled=debug)

    tok1 = torch.tensor(first_tokens, device=input_device, dtype=torch.long).view(K, 1)
    am1 = torch.ones((K, 2), device=input_device, dtype=torch.long)

    next_logits_1, past_1, gate_outs_1 = forward_step_batched(
        model=model,
        tracker=tracker,
        token_ids=tok1,
        attention_mask=am1,
        past_key_values=past_bos,
        max_router_out=max_router_out,
        debug=debug,
    )

    inferred_E = _infer_num_experts(model, gate_outs_1, max_router_out=max_router_out)
    L = max(1, len(tracker.layer_names))
    if inferred_E <= 1:
        inferred_E = max(2, inferred_E)

    counts1 = torch.zeros((K, L, inferred_E), device=input_device, dtype=torch.int32)
    update_counts_from_gate_outputs_batch_gpu(counts1, gate_outs_1, moe_top_k=moe_top_k)

    # split past_1 (batched K) into per-sentence batch-1 past
    # pasts_1: List[Tuple] = []
    # for i in range(K):
    #     one = []
    #     for (k_t, v_t) in past_1:
    #         one.append((k_t[i:i + 1].contiguous(), v_t[i:i + 1].contiguous()))
    #     pasts_1.append(tuple(one))
    pasts_1 = _split_past_key_values(past_1, K)

    # Initialize per-sentence candidates (beam size = 1 initially)
    beams: List[List[Candidate]] = []
    for i in range(K):
        beams.append([
            Candidate(
                tokens=[first_tokens[i]],
                rs=rs_first[i],
                counts=counts1[i],
                length=1,
                past_key_values=pasts_1[i],
                next_logits=next_logits_1[i],
            )
        ])

    # ---- Main loop (j from 1 to seq_len-1) ----
    for j in range(1, seq_len):
        # For each sentence, build expansions, then keep top-w per sentence.
        selected_global = []  # list of dict with sentence index, parent idx, new token, rs_new, val
        for si in range(K):
            candidates = beams[si]
            expanded = []  # (val, pi, v, rs_new)
            for pi, cand in enumerate(candidates):
                # deferred sigma on parent
                sigma_parent = float(sigma_from_counts_gpu(cand.counts.unsqueeze(0))[0].item())

                ngram_set = _build_ngram_set(cand.tokens, no_repeat_ngram_size) if no_repeat_ngram_size > 1 else set()

                logits = cand.next_logits.clone()

                # hard ban
                if banned_ids_device.numel() > 0:
                    logits.index_fill_(0, banned_ids_device, -1e9)

                # repetition penalty
                if repetition_penalty and repetition_penalty > 1.0:
                    used = set(cand.tokens)
                    used = [t for t in used if 0 <= t < logits.numel()]
                    if used:
                        idx = torch.tensor(used, device=logits.device, dtype=torch.long)
                        vals = logits.index_select(0, idx)
                        vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
                        logits.index_copy_(0, idx, vals)

                log_probs = torch.log_softmax(logits, dim=-1)
                k = min(top_token_k, log_probs.numel())
                topv = torch.topk(log_probs, k=k, dim=-1).indices.tolist()

                for v in topv:
                    if v in banned_set:
                        continue
                    if avoid_special and v in special_ids:
                        continue
                    if avoid_eos_before_end and eos_id is not None and v == eos_id and j < (seq_len - 1):
                        continue
                    if max_repeat_run > 0 and _exceeds_repeat_run(cand.tokens, v, max_repeat_run):
                        continue
                    if no_repeat_ngram_size > 1 and _would_repeat_ngram(cand.tokens, v, no_repeat_ngram_size, ngram_set):
                        continue

                    lp = float(log_probs[v].item())
                    rs_new = cand.rs + lp
                    l_new = cand.length + 1
                    val = compute_score_or_cost(rs_new, l_new, sigma_parent, tau)
                    expanded.append((val, pi, v, rs_new))

            if not expanded:
                # If a sentence cannot expand, keep as-is (early stop)
                continue

            reverse = True if objective == "max" else False
            expanded.sort(key=lambda x: x[0], reverse=reverse)
            expanded = expanded[:width_w]

            for (val, pi, v, rs_new) in expanded:
                selected_global.append({
                    "si": si,
                    "pi": pi,
                    "v": v,
                    "rs_new": rs_new,
                    "val": val,
                })

        if not selected_global:
            break

        # ---- Batched forward for all selected expansions across K sentences ----
        B = len(selected_global)

        parent_pasts = []
        parent_counts = []
        parent_tokens_list = []
        parent_rs_list = []
        new_tokens = []
        parent_lengths = []
        sent_index = []

        for item in selected_global:
            si = item["si"]
            pi = item["pi"]
            parent = beams[si][pi]

            parent_pasts.append(parent.past_key_values)
            parent_counts.append(parent.counts)  # [L,E] GPU
            parent_tokens_list.append(parent.tokens)
            parent_rs_list.append(parent.rs)
            parent_lengths.append(parent.length)
            new_tokens.append(item["v"])
            sent_index.append(si)

        # Checkpoint B: Check parent pasts before and after stacking
        if debug and parent_pasts:
            debug_check_cache(parent_pasts[0], expect_B=1, tag="parent_past_0", enabled=debug)
        batched_past = _stack_past_key_values(parent_pasts)
        debug_check_cache(batched_past, expect_B=B, tag="batched_past_stacked", enabled=debug)

        # At step j: all parents have length==j (if beams are well-formed), but
        # we keep it robust and just build a constant-length mask by max length.
        max_parent_len = max(parent_lengths) if parent_lengths else j
        # BOS + parent_len + new_token
        cur_len = 1 + max_parent_len + 1
        attention_mask = torch.ones((B, cur_len), device=input_device, dtype=torch.long)

        token_ids = torch.tensor(new_tokens, device=input_device, dtype=torch.long).view(B, 1)

        next_logits_b, new_past_b, gate_outs_b = forward_step_batched(
            model=model,
            tracker=tracker,
            token_ids=token_ids,
            attention_mask=attention_mask,
            past_key_values=batched_past,
            max_router_out=max_router_out,
            debug=debug,
        )

        counts_b = torch.stack(parent_counts, dim=0).contiguous()  # [B,L,E]
        update_counts_from_gate_outputs_batch_gpu(counts_b, gate_outs_b, moe_top_k=moe_top_k)

        # ---- Dispatch back to per-sentence beams (keep EXACT w per sentence) ----
        # First, collect next candidates per sentence
        next_beams: List[List[Candidate]] = [[] for _ in range(K)]

        # Checkpoint C: Check new_past before and after split
        debug_check_cache(new_past_b, expect_B=B, tag="new_past_b", enabled=debug)
        new_past_splits = _split_past_key_values(new_past_b, B)
        if debug and new_past_splits:
            debug_check_cache(new_past_splits[0], expect_B=1, tag="new_past_split0", enabled=debug)
        for bi in range(B):
            si = sent_index[bi]
            sliced_past = new_past_splits[bi]

            # reconstruct candidate
            # Note: parent_lengths may vary if some sentences had early stop; keep robust
            # by using stored parent token list.
            new_tok = int(new_tokens[bi])
            # rs_new already computed in selection_global
            rs_new = float(selected_global[bi]["rs_new"])

            cand_new = Candidate(
                tokens=parent_tokens_list[bi] + [new_tok],
                rs=rs_new,
                counts=counts_b[bi],
                length=len(parent_tokens_list[bi]) + 1,
                past_key_values=sliced_past,
                next_logits=next_logits_b[bi],
            )
            next_beams[si].append(cand_new)

        # For sentences that had no expansions, keep their old beams (early stop)
        for si in range(K):
            if not next_beams[si]:
                next_beams[si] = beams[si]
                continue

            # If more than w (can happen if sentence got multiple selected_global due to parent beams < w),
            # rank and trim to w using deferred sigma on the new candidate itself (consistent final selection).
            if len(next_beams[si]) > width_w:
                sig = sigma_from_counts_gpu(torch.stack([c.counts for c in next_beams[si]], dim=0))
                scored = []
                for idx, c in enumerate(next_beams[si]):
                    val = compute_score_or_cost(
                        rs_new=c.rs,
                        l_new=c.length,
                        sigma_parent=float(sig[idx].item()),
                        tau=tau,
                    )
                    scored.append((val, idx))
                reverse = True if objective == "max" else False
                scored.sort(key=lambda x: x[0], reverse=reverse)
                keep_idx = [idx for (_, idx) in scored[:width_w]]
                next_beams[si] = [next_beams[si][idx] for idx in keep_idx]

        beams = next_beams

    # ---- Pick best final candidate per sentence ----
    results: List[Dict[str, Any]] = []
    for si in range(K):
        candidates = beams[si]
        if not candidates:
            results.append({"text": ""})
            continue

        counts_stack = torch.stack([c.counts for c in candidates], dim=0)
        sig = sigma_from_counts_gpu(counts_stack)  # [n]
        best = None
        best_val = None
        for i, c in enumerate(candidates):
            val = compute_score_or_cost(c.rs, c.length, float(sig[i].item()), tau)
            if best is None:
                best, best_val = c, val
            else:
                if (objective == "min" and val < best_val) or (objective == "max" and val > best_val):
                    best, best_val = c, val

        text = tokenizer.decode(best.tokens, skip_special_tokens=True)
        results.append({
            "text": text,
            "input_ids": best.tokens,
            "stats": {
                "length": int(best.length),
                "rs": float(best.rs),
                "sigma": float(sigma_from_counts_gpu(best.counts.unsqueeze(0))[0].item()),
                "score_or_cost": float(best_val),
                "objective": objective,
                "num_layers": int(best.counts.shape[0]),
                "num_experts": int(best.counts.shape[1]),
            }
        })

    return results


# -----------------------------
# 10) Build dataset (NS sentences) + incremental saving
# -----------------------------
def build_ebss_dataset(
    model_path: str,
    output_path: str,
    ns: int,
    seq_len: int,
    w: int,
    tau: float,
    moe_top_k: int,
    top_token_k: int,
    objective: str,
    seed: int,
    trust_remote_code: bool,
    router_regex: str,
    exclude_regex: str,
    max_router_out: int,
    save_every: int,  # ignored (write per-sentence)
    max_retries_per_sentence: int,
    avoid_special: bool,
    avoid_eos_before_end: bool,
    sent_batch: int,
    debug: bool = False,
):
    print(f"[EBSS] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    # Multi-GPU model parallelism: use device_map="auto" to split model across GPUs
    print("[EBSS] Loading model with device_map='auto' for multi-GPU support...")

    # Try FlashAttention2 if available; fallback silently.
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="auto",  # Automatically distribute layers across available GPUs
            attn_implementation="sdpa",
        )
        print("[EBSS] Model loaded with device_map='auto' and attn_implementation=sdpa.")
    except Exception as e:
        print(f"[EBSS][WARN] sdpa unavailable or failed ({type(e).__name__}: {e}). Falling back to default attention.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="auto",  # Automatically distribute layers across available GPUs
        )
        print("[EBSS] Model loaded with device_map='auto'.")

    model.eval()
    input_device = _get_input_device(model)
    print(f"[EBSS] Input device: {input_device}")

    tracker = ExpertTracker(
        model=model,
        router_regex=router_regex,
        exclude_regex=exclude_regex,
        max_router_out=max_router_out,
        verbose=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ---- Build banned tokens once (CJK/fullwidth) ----
    cjk_re = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
    vocab_size = int(getattr(tokenizer, "vocab_size", None) or model.get_input_embeddings().weight.shape[0])

    print(f"[EBSS] Building CJK banned token list for vocab_size={vocab_size} ...")
    banned_token_ids = []
    special_ids = set(tokenizer.all_special_ids or [])
    for tid in range(vocab_size):
        if tid in special_ids:
            continue
        s = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if s and cjk_re.search(s):
            banned_token_ids.append(tid)

    banned_set = set(banned_token_ids)
    banned_ids_device = (
        torch.tensor(banned_token_ids, device=input_device, dtype=torch.long)
        if banned_token_ids else torch.empty((0,), device=input_device, dtype=torch.long)
    )
    print(f"[EBSS] Banned CJK-like tokens: {len(banned_token_ids)}")

    # ---- Open output in append mode, write per sentence ----
    f = open(output_path, "a", encoding="utf-8")

    written = 0
    t0 = time.time()
    pbar = tqdm(total=ns, desc=f"EBSS ({objective})", dynamic_ncols=True, unit="sent")

    # Use inference_mode globally for speed
    try:
        with torch.inference_mode():
            i = 0
            while i < ns:
                # generate up to this many in one batch
                k = min(sent_batch, ns - i)
                sent_list = None
                last_err = None

                for r in range(max_retries_per_sentence):
                    try:
                        sent_list = generate_sent_batch_ebss(
                            model=model,
                            tokenizer=tokenizer,
                            tracker=tracker,
                            input_device=input_device,
                            seq_len=seq_len,
                            width_w=w,
                            tau=tau,
                            moe_top_k=moe_top_k,
                            top_token_k=top_token_k,
                            objective=objective,
                            sent_batch=k,
                            banned_ids_device=banned_ids_device,
                            banned_set=banned_set,
                            avoid_special=avoid_special,
                            avoid_eos_before_end=avoid_eos_before_end,
                            base_seed=seed + i * 1000 + r,
                            max_router_out=max_router_out,
                            debug=debug,
                        )
                        if sent_list and isinstance(sent_list, list):
                            break
                    except RuntimeError as e:
                        last_err = repr(e)
                        if "out of memory" in str(e).lower():
                            raise
                        sent_list = None

                if sent_list is None:
                    pbar.set_postfix(fail="1", wrote=written)
                    if last_err:
                        print(f"[EBSS][DBG] batch starting at {i} failed. last_err={last_err}")
                    continue

                # write results (keep EXACT jsonl format you want)
                for sent in sent_list:
                    txt = sent.get("text", "")
                    if not isinstance(txt, str) or len(txt.strip()) == 0:
                        continue
                    f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                    written += 1
                    pbar.update(1)

                    # optional live stats
                    s = sent.get("stats", {})
                    dt = time.time() - t0
                    pbar.set_postfix(
                        wrote=written,
                        len=s.get("length", seq_len),
                        sigma=f"{s.get('sigma', 0.0):.4f}" if isinstance(s.get("sigma", 0.0), (int, float)) else "NA",
                        val=f"{s.get('score_or_cost', 0.0):.4f}" if isinstance(s.get("score_or_cost", 0.0), (int, float)) else "NA",
                        t=f"{dt:.0f}s",
                        bs=k,
                        bw=w,
                    )

                    if written >= ns:
                        break

                f.flush()
                i = written  # advance by actual written count

        print(f"\n[EBSS] Done. Wrote {written} samples to: {output_path}")

    finally:
        try:
            f.close()
        except Exception:
            pass
        pbar.close()
        tracker.remove()


# -----------------------------
# 11) CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("EBSS builder (paper workflow + batching + objective flag).")
    ap.add_argument("--model", type=str, required=True, help="HF model name or local path (MoE model).")
    ap.add_argument("--output", type=str, required=True, help="Output jsonl path.")

    ap.add_argument("--ns", type=int, default=128, help="Number of sentences.")
    ap.add_argument("--seqlen", type=int, default=128, help="Tokens per sentence (excluding BOS).")
    ap.add_argument("--w", type=int, default=4, help="Beam width w.")
    ap.add_argument("--tau", type=float, default=0.1, help="Balance temperature tau.")
    ap.add_argument("--moe_top_k", type=int, default=2, help="MoE routing top-k experts per token (for counting).")
    ap.add_argument("--top_token_k", type=int, default=64, help="Per-branch expansion top-k tokens (prob-guided pruning).")

    ap.add_argument("--objective", type=str, choices=["min", "max"], default="min",
                    help="min: minimize cost; max: maximize paper-written score (strict topk/max).")

    ap.add_argument("--seed", type=int, default=1234, help="Random seed.")
    ap.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for model/tokenizer.")

    # router detection
    ap.add_argument("--router_regex", type=str, default=r"(router|routing|gate|moe)",
                    help="Regex to match router/gate module names.")
    ap.add_argument("--exclude_regex", type=str, default=r"(gate_proj|up_proj|down_proj|shared_expert)",
                    help="Regex to exclude dense FFN projections (use 'gate_proj' not 'gate' to allow mlp.gate router).")
    ap.add_argument("--max_router_out", type=int, default=1024,
                    help="Max out_features/last_dim accepted as router logits dim (Qwen3-Next has 512 experts).")

    # robustness / IO
    ap.add_argument("--save_every", type=int, default=16, help="Flush every N sentences to jsonl (append mode).")
    ap.add_argument("--max_retries_per_sentence", type=int, default=5, help="Retry times per sentence/batch.")
    ap.add_argument("--avoid_special", action="store_true", help="Avoid generating special tokens.")
    ap.add_argument("--avoid_eos_before_end", action="store_true", help="Avoid EOS before the last token.")

    # NEW: sentence-level batching to fill GPU
    ap.add_argument("--sent_batch", type=int, default=8, help="Generate this many sentences in parallel (batch=sent_batch*w per step).")

    # Debug flag for cache shape checking
    ap.add_argument("--debug", action="store_true", help="Enable debug output for cache shape checking.")

    return ap.parse_args()


def main():
    args = parse_args()
    build_ebss_dataset(
        model_path=args.model,
        output_path=args.output,
        ns=args.ns,
        seq_len=args.seqlen,
        w=args.w,
        tau=args.tau,
        moe_top_k=args.moe_top_k,
        top_token_k=args.top_token_k,
        objective=args.objective,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        router_regex=args.router_regex,
        exclude_regex=args.exclude_regex,
        max_router_out=args.max_router_out,
        save_every=args.save_every,
        max_retries_per_sentence=args.max_retries_per_sentence,
        avoid_special=args.avoid_special,
        avoid_eos_before_end=args.avoid_eos_before_end,
        sent_batch=args.sent_batch,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
