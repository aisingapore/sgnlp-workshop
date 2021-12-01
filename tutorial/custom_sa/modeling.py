from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from tutorial.custom_sa.config import CustomSaConfig


@dataclass
class CustomSaModelOutput(ModelOutput):
    loss: float = None
    logits: torch.Tensor = None


class CustomSaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CustomSaConfig
    base_model_prefix = "custom_sa"

    def _init_weights(self, module):
        pass


class CustomSaModel(CustomSaPreTrainedModel):
    """This is a custom model for sentiment analysis.

    Args:
        config (:class:`~tutorial.custom_sa.config.CustomSaConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.
    """

    def __init__(self, config: CustomSaConfig):
        super().__init__(config)

        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate

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
        dropout_output = self.dropout(lstm_output)
        linear_output = self.linear(dropout_output)

        loss = None
        if labels:
            loss = self.loss(linear_output, labels)

        return CustomSaModelOutput(loss=loss, logits=linear_output)
