import torch

from bert import build_model

model = build_model(num_layers=6, dim_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1, max_len=512, vocabulary_size=100).cuda()

sequence = torch.tensor([[1, 2, 3, 4, 5], [2, 1, 3, 0, 0]]).cuda()
segment = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]]).cuda()
token_predictions, classification_output = model((sequence, segment))
print(token_predictions.size(), classification_output.size())