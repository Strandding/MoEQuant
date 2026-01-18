#!/usr/bin/env python3
import torch
print(f"PyTorch version: {torch.__version__}")

# Try different import paths
print("\nTrying Replicate:")
try:
    from torch.distributed.tensor import Replicate
    print("✓ from torch.distributed.tensor import Replicate")
except ImportError as e:
    print(f"✗ from torch.distributed.tensor import Replicate - {e}")

try:
    from torch.distributed._tensor import Replicate
    print("✓ from torch.distributed._tensor import Replicate")
except ImportError as e:
    print(f"✗ from torch.distributed._tensor import Replicate - {e}")

try:
    from torch.distributed._tensor.placement_types import Replicate
    print("✓ from torch.distributed._tensor.placement_types import Replicate")
except ImportError as e:
    print(f"✗ from torch.distributed._tensor.placement_types import Replicate - {e}")

print("\nTrying distribute_module:")
try:
    from torch.distributed.tensor import distribute_module
    print("✓ from torch.distributed.tensor import distribute_module")
except ImportError as e:
    print(f"✗ from torch.distributed.tensor import distribute_module - {e}")

try:
    from torch.distributed.tensor.parallel import distribute_module
    print("✓ from torch.distributed.tensor.parallel import distribute_module")
except ImportError as e:
    print(f"✗ from torch.distributed.tensor.parallel import distribute_module - {e}")

# Check what's available
import torch.distributed.tensor as dtt
print(f"\nAvailable in torch.distributed.tensor: {dir(dtt)[:30]}")
