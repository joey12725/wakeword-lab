#!/usr/bin/env python3
"""Verify CUDA availability for training."""

import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Capability:", torch.cuda.get_device_capability(0))
    
    # Test tensor on GPU
    x = torch.randn(100, 100).cuda()
    print("\n✓ Successfully created tensor on GPU!")
else:
    print("\n✗ CUDA not available. Training will use CPU (much slower).")
