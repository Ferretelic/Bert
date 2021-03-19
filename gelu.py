import math

import torch
import numpy as np
import matplotlib.pyplot as plt

class GELU(torch.nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))