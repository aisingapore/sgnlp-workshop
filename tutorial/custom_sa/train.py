import argparse
import json
import os
import logging

import nltk
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset

from .config import CustomSaConfig
from .modeling import CustomSaModel
from .preprocess import CustomSaPreprocessor
from .train_args import CustomSaTrainConfig


logging.basicConfig(level=logging.INFO)


class KaggleSentimentDataset(Dataset):
    def __init__(self, df, preprocessor):
        self.df = df
        self.sentences = self.df["Phrase"].values
        self.labels = self.df["Sentiment"].values
        self.labels = torch.LongTensor(self.labels)
        self.preprocessor = preprocessor
        self.tokenized_ids = preprocessor(self.sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"tokenized_ids": self.tokenized_ids[idx], "labels": self.labels[idx]}


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

    # Build vocab from train data
    data = pd.read_csv(train_config.data_path, sep="\t")
    # Take a sample for test run in tutorial
    sample_size = 3000
    data = data.sample(sample_size, random_state=train_config.seed)

    train_data, val_data = train_test_split(
        data[["Phrase", "Sentiment"]], train_size=0.8, random_state=train_config.seed
    )

    tokenizer = nltk.word_tokenize
    vocab = CustomSaPreprocessor.build_vocab(
        data["Phrase"].values, tokenizer, model_config.vocab_size
    )
    CustomSaPreprocessor.save_vocab(vocab, train_config.output_dir)

    # Prepare data
    preprocessor = CustomSaPreprocessor(vocab, tokenizer)

    train_dataset = KaggleSentimentDataset(train_data, preprocessor)
    val_dataset = KaggleSentimentDataset(val_data, preprocessor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=32)

    # Train model
    model = CustomSaModel(model_config)

    epochs = 10
    optimizer = torch.optim.Adam(model.parameters())

    # Keep best model based on val f1 score
    best_val_f1_score = None
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")

        epoch_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):

            outputs = model(**batch)

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        epoch_train_loss = epoch_train_loss / (step + 1)

        epoch_val_loss = 0
        labels = []
        preds = []
        model.eval()
        for step, batch in enumerate(val_dataloader):
            outputs = model(**batch)

            batch_preds = torch.argmax(
                torch.softmax(outputs.logits, dim=1), dim=1
            ).tolist()
            preds.extend(batch_preds)
            labels.extend(batch["labels"].tolist())

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_val_loss += loss.item()
        epoch_val_loss = epoch_val_loss / (step + 1)
        val_f1_score = sklearn.metrics.f1_score(labels, preds, average="macro")
        if best_val_f1_score is None:
            best_val_f1_score = val_f1_score
        else:
            if val_f1_score > best_val_f1_score:
                best_val_f1_score = val_f1_score
                model.save_pretrained(
                    os.path.join(train_config.output_dir, "best_val_f1")
                )

        logging.info(
            f"Train loss: {epoch_train_loss:.3f}, Val loss: {epoch_val_loss:.3f}, Val f1 score: {val_f1_score:.3f}"
        )


if __name__ == "__main__":
    train_config = parse_args()
    train_custom_sa(train_config)
