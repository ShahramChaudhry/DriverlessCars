import torch

from config import DEVICE

print("Using device:", DEVICE)
x = torch.rand(3, 3, device=DEVICE)
print(x)
