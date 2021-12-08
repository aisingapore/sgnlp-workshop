from transformers import PretrainedConfig


class CustomSaConfig(PretrainedConfig):
    """
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        hidden_dim (:obj:`int`, defaults to 64): Hidden dimension size.
        output_dim (:obj:`int`, defaults to 5): Output dimension size.
        vocab_size (:obj:`int`, defaults to 5000): Vocabulary size.
        embedding_dim (:obj:`int`, defaults to 64): Embedding dimension size.
        num_layers (:obj:`int`, defaults to 2): Number of LSTM layers.
        dropout_rate (:obj:`float`, defaults to 0.3): Dropout rate.
    """

    def __init__(
        self,
        hidden_dim=64,
        output_dim=5,
        vocab_size=5000,
        embedding_dim=64,
        num_layers=2,
        dropout_rate=0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
