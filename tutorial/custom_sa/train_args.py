from dataclasses import dataclass, field


@dataclass
class CustomSaTrainConfig:
    data_path: str = field(metadata={"help": "Data path"})
    model_config_path: str = field(metadata={"help": "Model config path"})
    output_dir: str = field(metadata={"help": "Output directory"})
    seed: str = field(metadata={"help": "Random seed"})
