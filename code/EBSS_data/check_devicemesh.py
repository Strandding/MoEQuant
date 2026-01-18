#!/usr/bin/env python3
import torch
print(f"PyTorch version: {torch.__version__}")

# Try different import paths for DeviceMesh
print("\nTrying different import paths:")

try:
    from torch.distributed import DeviceMesh
    print("✓ from torch.distributed import DeviceMesh")
except ImportError as e:
    print(f"✗ from torch.distributed import DeviceMesh - {e}")

try:
    from torch.distributed._tensor import DeviceMesh
    print("✓ from torch.distributed._tensor import DeviceMesh")
except ImportError as e:
    print(f"✗ from torch.distributed._tensor import DeviceMesh - {e}")

try:
    from torch.distributed.device_mesh import DeviceMesh
    print("✓ from torch.distributed.device_mesh import DeviceMesh")
except ImportError as e:
    print(f"✗ from torch.distributed.device_mesh import DeviceMesh - {e}")

# Check what's available in torch.distributed
import torch.distributed as dist
print("\nAvailable in torch.distributed:")
attrs = [a for a in dir(dist) if 'mesh' in a.lower() or 'device' in a.lower()]
print(attrs[:20])
