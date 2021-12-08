"""
Part 1:
Refactor from a pytorch nn.Module to a transformers.PreTrainedModel

Steps:
1. Create a class to inherit from transformers.PreTrainedModel
2. Change CustomSaModel to inherit from the above class
3. Refactor init to use CustomSaConfig in config.py

"""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from .config import CustomSaConfig


@dataclass
class CustomSaModelOutput(ModelOutput):
    loss: float = None
    logits: torch.Tensor = None


class CustomSaModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, vocab_size, embedding_dim, num_layers, dropout_rate):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, tokenized_ids, labels=None):
        """
        Args:
            tokenized_ids: Token IDs.
            labels: (Optional) Loss will be calculated if labels are provided.

        Returns:
            output (:class:`~tutorial.custom_sa.modeling.CustomSaModelOutput`)
        """

        embeddings = self.embedding(tokenized_ids)
        lstm_output, _ = self.lstm(embeddings)
        lstm_final_hidden = lstm_output[:, -1, :]  # use final hidden states only
        dropout_output = self.dropout(lstm_final_hidden)
        linear_output = self.linear(dropout_output)

        loss = None
        if labels is not None:
            loss = self.loss(linear_output, labels)

        return CustomSaModelOutput(loss=loss, logits=linear_output)
