import torch

def pad_masking(x: torch.Tensor) -> torch.Tensor:
  padded_positions = x == 0
  return padded_positions.unsqueeze(1)