from dataclasses import dataclass, field


@dataclass
class CustomSaTrainConfig:
    train_data_path: str = field(metadata={"help": "Training data path"})
    test_data_path: str = field(metadata={"help": "Test data path"})
    model_config_path: str = field(metadata={"help": "Model config path"})
    output_dir: str = field(metadata={"help": "Output directory"})
