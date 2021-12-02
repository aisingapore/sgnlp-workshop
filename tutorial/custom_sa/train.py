import argparse
import json
import os

import nltk
import pandas as pd


from tutorial.custom_sa.config import CustomSaConfig
from tutorial.custom_sa.modeling import CustomSaModel
from tutorial.custom_sa.preprocess import CustomSaPreprocessor
from tutorial.custom_sa.train_args import CustomSaTrainConfig


def parse_args():

    parser = argparse.ArgumentParser(description="Custom SA Model Training")
    parser.add_argument(
        "--train_config_path",
        type=str,
        required=True,
        help="Path to training config file.",
    )

    args = parser.parse_args()
    with open(args.train_config_path, "r") as f:
        config = json.load(f)

    train_config = CustomSaTrainConfig(**config)
    return train_config


def train_custom_sa(train_config: CustomSaTrainConfig):
    model_config = CustomSaConfig.from_json_file(train_config.model_config_path)
    os.makedirs(train_config.output_dir, exist_ok=True)

    # Load data
    train_data = pd.read_csv(train_config.train_data_path, sep="\t")
    test_data = pd.read_csv(train_config.train_data_path, sep="\t")

    # Build vocab
    tokenizer = nltk.word_tokenize
    vocab = CustomSaPreprocessor.build_vocab(
        train_data["Phrase"].values, tokenizer, model_config.vocab_size
    )
    CustomSaPreprocessor.save_vocab(vocab, train_config.output_dir)




if __name__ == "__main__":
    train_config = parse_args()
    train_custom_sa(train_config)
