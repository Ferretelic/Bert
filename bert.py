import torch

from embeddings import PositionalEmbedding, SegmentEmbedding
from transformer import TransformerEncoder
from mask import pad_masking

class BERT(torch.nn.Module):
  def __init__(self, encoder: torch.nn.Module, token_embedding: torch.nn.Module, positional_embedding: torch.nn.Module, segment_embedding: torch.nn.Module, dim_model: int, vocabulary_size: int):
    super(BERT, self).__init__()

    self.encoder = encoder
    self.token_embedding = token_embedding
    self.positional_embedding = positional_embedding
    self.segment_embedding = segment_embedding
    self.token_prediction_layer = torch.nn.Linear(dim_model, vocabulary_size)
    self.classification_layer = torch.nn.Linear(dim_model, 2)

  def forward(self, inputs: tuple) -> torch.Tensor:
    sequence, segment = inputs
    token_embedded = self.token_embedding(sequence)
    positional_embedded = self.positional_embedding(sequence)
    segment_embedded = self.segment_embedding(segment)
    embedded_sources = token_embedded + positional_embedded + segment_embedded

    mask = pad_masking(sequence)
    encoded_sources = self.encoder(embedded_sources, mask)
    token_predictions = self.token_prediction_layer(encoded_sources)
    classification_embedding = embedded_sources[:, 0, :]
    classification_output = self.classification_layer(classification_embedding)
    return token_predictions, classification_output

class FineTuneModel(torch.nn.Module):
  def __init__(self, pretrained_model: torch.nn.Module, dim_model: int, num_classes: int):
    super(FineTuneModel, self).__init__()

    self.pretrained_model = pretrained_model
    new_classification_layer = torch.nn.Linear(dim_model, num_classes)
    self.pretrained_model.classification_layer = new_classification_layer

  def forward(self, inputs: tuple) -> torch.Tensor:
    sequence, segment = inputs
    token_predictions, classification_outputs = self.pretrained_model((sequence, segment))
    return classification_outputs

def build_model(num_layers: int, dim_model: int, num_heads: int, dim_feedforward: int, dropout: float, max_len: int, vocabulary_size: int):
  token_embedding = torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=dim_model)
  positional_embedding = PositionalEmbedding(max_len=max_len, dim_model=dim_model)
  segment_embedding = SegmentEmbedding(dim_model=dim_model)

  encoder = TransformerEncoder(
    num_layers = num_layers,
    dim_model = dim_model,
    num_heads = num_heads,
    dim_feedforward = dim_feedforward,
    dropout = dropout
  )

  bert = BERT(
    encoder = encoder,
    token_embedding = token_embedding,
    positional_embedding = positional_embedding,
    segment_embedding = segment_embedding,
    dim_model = dim_model,
    vocabulary_size = vocabulary_size
  )

  return bert