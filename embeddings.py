import torch

class PositionalEmbedding(torch.nn.Module):
  def __init__(self, max_len: int, dim_model: int):
    super(PositionalEmbedding, self).__init__()
    self.positional_embedding = torch.nn.Embedding(max_len, dim_model)
    positions = torch.arange(0, max_len)
    self.register_buffer("positions", positions)

  def forward(self, sequence: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = sequence.size()
    positions = self.positions[:seq_len].unsqueeze(0).repeat(batch_size, 1)
    return self.positional_embedding(positions)

class SegmentEmbedding(torch.nn.Module):
  def __init__(self, dim_model: int):
    super(SegmentEmbedding, self).__init__()
    self.segment_embedding = torch.nn.Embedding(2, dim_model)

  def forward(self, segments: torch.Tensor) -> torch.Tensor:
    return self.segment_embedding(segments)