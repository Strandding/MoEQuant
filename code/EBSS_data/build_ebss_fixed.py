#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EBSS (Expert-Balanced Self-Sampling) builder — paper-faithful workflow + engineering optimizations.
FIXED VERSION: Enhanced compatibility with Qwen3-Next and other models with non-standard cache structures.

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

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers.cache_utils import Cache, DynamicCache
except Exception:
    Cache = None
    DynamicCache = None

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
            if not isinstance(module, nn.Linear):
                continue
            if self.exclude_re is not None and self.exclude_re.search(name):
                continue
            if self.router_re is not None and not self.router_re.search(name):
                continue

            # Heuristic: router out_features should be small
            if getattr(module, "out_features", None) is not None:
                if module.out_features <= 1 or module.out_features > self.max_router_out:
                    continue

            self.layer_names.append(name)
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
# 3) Device helpers + KV batching (FIXED)
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
    # Duck typing: check for Cache-like attributes/methods
    # This handles custom Cache classes (e.g., HybridCache in Qwen3-Next)
    if hasattr(pkv, "get_seq_length") and hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return True
    return False


def _get_cache_tensor_shape(tensor: torch.Tensor) -> str:
    """Get a readable shape string for debugging."""
    if tensor is None:
        return "None"
    return f"[{', '.join(map(str, tensor.shape))}]"


def _normalize_kv_tensor_for_repeat(tensor: torch.Tensor, times: int) -> torch.Tensor:
    """
    Repeat a k/v tensor along batch dimension, handling various shapes.
    Supports 3D, 4D, and 5D tensors with flexible batch dimension handling.
    """
    if tensor is None:
        return None

    if times <= 1:
        return tensor

    ndim = tensor.ndim
    if ndim < 3:
        # Unexpected shape, try to add batch dim if needed
        if ndim == 2:
            tensor = tensor.unsqueeze(0)
            ndim = 3
        else:
            raise ValueError(f"Unexpected tensor shape for KV cache: {tensor.shape}")

    # Assume first dimension is batch
    # Create repeat pattern: (times, 1, 1, ...) for all other dimensions
    repeat_pattern = [times] + [1] * (ndim - 1)
    return tensor.repeat(*repeat_pattern)


def _normalize_kv_tensor_for_slice(tensor: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """
    Slice a k/v tensor at batch dimension, handling various shapes.
    """
    if tensor is None:
        return None

    # Assume first dimension is batch, slice and keep as [1, ...]
    return tensor[batch_idx:batch_idx+1].contiguous()


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
                # seq_len is typically at dim -2 (e.g., [B, H, S, D] or [B, S, H, D])
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
    FIXED: Better shape validation and error handling.
    Supports:
      - New HF Cache (transformers.cache_utils.Cache / DynamicCache)
      - Legacy tuple(num_layers) of (k,v)
    """
    if not pasts:
        raise ValueError("Empty past list")

    p0 = pasts[0]

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
                # Validate shapes before stacking
                num_layers = len(p0.key_cache)
                for li in range(num_layers):
                    # Get reference shape from first cache
                    ref_k = p0.key_cache[li]
                    ref_v = p0.value_cache[li]

                    if ref_k is None or ref_v is None:
                        continue

                    # Check all caches have compatible shapes
                    for pi, p in enumerate(pasts[1:], 1):
                        if li >= len(p.key_cache) or li >= len(p.value_cache):
                            raise ValueError(f"Cache {pi} missing layer {li}")

                        k = p.key_cache[li]
                        v = p.value_cache[li]

                        if k is None or v is None:
                            if ref_k is not None or ref_v is not None:
                                raise ValueError(f"Cache {pi} layer {li} has None while reference is not None")
                            continue

                        # Check shape compatibility (all dims except batch dim 0)
                        if k.shape[1:] != ref_k.shape[1:]:
                            raise ValueError(
                                f"Key shape mismatch at layer {li}: "
                                f"cache 0 has {_get_cache_tensor_shape(ref_k)}, "
                                f"cache {pi} has {_get_cache_tensor_shape(k)}"
                            )
                        if v.shape[1:] != ref_v.shape[1:]:
                            raise ValueError(
                                f"Value shape mismatch at layer {li}: "
                                f"cache 0 has {_get_cache_tensor_shape(ref_v)}, "
                                f"cache {pi} has {_get_cache_tensor_shape(v)}"
                            )

                # All shapes validated, proceed with stacking
                # Try to create new instance
                try:
                    cache = cls()
                except Exception:
                    # If empty instantiation fails, deepcopy and reset
                    cache = copy.deepcopy(p0)
                if hasattr(p0, "_seen_tokens"):
                    cache._seen_tokens = p0._seen_tokens
                cache.key_cache = [torch.cat([p.key_cache[i] for p in pasts], dim=0) for i in range(len(p0.key_cache))]
                cache.value_cache = [torch.cat([p.value_cache[i] for p in pasts], dim=0) for i in range(len(p0.value_cache))]
                return cache
            except Exception as e:
                print(f"[WARN] Cache stack failed: {e}")
                print(f"[DEBUG] Falling back to tuple conversion")
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
        for pi, p in enumerate(pasts):
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

                # Validate shape compatibility
                if ks and k.shape[1:] != ks[0].shape[1:]:
                    raise ValueError(
                        f"Key shape mismatch at layer {li}: "
                        f"past 0 has {_get_cache_tensor_shape(ks[0])}, "
                        f"past {pi} has {_get_cache_tensor_shape(k)}"
                    )
                if vs and v.shape[1:] != vs[0].shape[1:]:
                    raise ValueError(
                        f"Value shape mismatch at layer {li}: "
                        f"past 0 has {_get_cache_tensor_shape(vs[0])}, "
                        f"past {pi} has {_get_cache_tensor_shape(v)}"
                    )

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
    FIXED: Better handling of various tensor shapes.
    Supports Cache and legacy tuple.
    """
    if times <= 1:
        return past

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
                # Repeat each layer's key and value tensors with dynamic shape handling
                new_cache.key_cache = [_normalize_kv_tensor_for_repeat(k, times) if k is not None else None
                                       for k in past.key_cache]
                new_cache.value_cache = [_normalize_kv_tensor_for_repeat(v, times) if v is not None else None
                                         for v in past.value_cache]
                return new_cache
            except Exception as e:
                # If all else fails, convert to tuple, repeat, and convert back
                print(f"[WARN] Cache repeat failed: {e}, falling back to tuple conversion")
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
                out.append((_normalize_kv_tensor_for_repeat(k, times), _normalize_kv_tensor_for_repeat(v, times)))
        else:
            # Unexpected format, keep as-is
            out.append(layer_past)
    return tuple(out)


def _split_past_key_values(past_batched: Any, batch_size: int) -> List[Any]:
    """
    Split a batched past (batch=batch_size) into list of batch=1 past.
    FIXED: Better handling of various tensor shapes.
    Supports Cache and legacy tuple.
    """
    if batch_size <= 0:
        return []

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
                    # Slice each layer's key and value tensors with dynamic shape handling
                    new_cache.key_cache = [_normalize_kv_tensor_for_slice(k, bi) if k is not None else None
                                          for k in past_batched.key_cache]
                    new_cache.value_cache = [_normalize_kv_tensor_for_slice(v, bi) if v is not None else None
                                            for v in past_batched.value_cache]
                    out.append(new_cache)
                return out
            except Exception as e:
                print(f"[WARN] Cache split failed: {e}, falling back to tuple conversion")
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
                    one.append((_normalize_kv_tensor_for_slice(k_t, bi), _normalize_kv_tensor_for_slice(v_t, bi)))
            else:
                # Unexpected format, keep as-is
                one.append(layer_past)
        pasts.append(tuple(one))
    return pasts


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
) -> Tuple[torch.Tensor, Tuple, List[torch.Tensor]]:
    """
    One token step for a batch.
    Returns:
      next_logits: [B, vocab] on device
      new_past: past_key_values (batched) on device
      gate_outputs: list of tensors on device (per-layer), either from native outputs or hooks.
    """
    tracker.clear()
    out = model(
        input_ids=token_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    )
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
    past_bos = _repeat_past_key_values(past_bos_1, K)

    tok1 = torch.tensor(first_tokens, device=input_device, dtype=torch.long).view(K, 1)
    am1 = torch.ones((K, 2), device=input_device, dtype=torch.long)

    next_logits_1, past_1, gate_outs_1 = forward_step_batched(
        model=model,
        tracker=tracker,
        token_ids=tok1,
        attention_mask=am1,
        past_key_values=past_bos,
        max_router_out=max_router_out,
    )

    inferred_E = _infer_num_experts(model, gate_outs_1, max_router_out=max_router_out)
    L = max(1, len(tracker.layer_names))
    if inferred_E <= 1:
        inferred_E = max(2, inferred_E)

    counts1 = torch.zeros((K, L, inferred_E), device=input_device, dtype=torch.int32)
    update_counts_from_gate_outputs_batch_gpu(counts1, gate_outs_1, moe_top_k=moe_top_k)

    # split past_1 (batched K) into per-sentence batch-1 past
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

        batched_past = _stack_past_key_values(parent_pasts)

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
        )

        counts_b = torch.stack(parent_counts, dim=0).contiguous()  # [B,L,E]
        update_counts_from_gate_outputs_batch_gpu(counts_b, gate_outs_b, moe_top_k=moe_top_k)

        # ---- Dispatch back to per-sentence beams (keep EXACT w per sentence) ----
        # First, collect next candidates per sentence
        next_beams: List[List[Candidate]] = [[] for _ in range(K)]

        new_past_splits = _split_past_key_values(new_past_b, B)
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
    ap.add_argument("--exclude_regex", type=str, default=r"(gate_proj|up_proj|down_proj)",
                    help="Regex to exclude dense FFN projections.")
    ap.add_argument("--max_router_out", type=int, default=256,
                    help="Max out_features/last_dim accepted as router logits dim (num_experts usually small).")

    # robustness / IO
    ap.add_argument("--save_every", type=int, default=16, help="Flush every N sentences to jsonl (append mode).")
    ap.add_argument("--max_retries_per_sentence", type=int, default=5, help="Retry times per sentence/batch.")
    ap.add_argument("--avoid_special", action="store_true", help="Avoid generating special tokens.")
    ap.add_argument("--avoid_eos_before_end", action="store_true", help="Avoid EOS before the last token.")

    # NEW: sentence-level batching to fill GPU
    ap.add_argument("--sent_batch", type=int, default=8, help="Generate this many sentences in parallel (batch=sent_batch*w per step).")

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
    )


if __name__ == "__main__":
    main()
