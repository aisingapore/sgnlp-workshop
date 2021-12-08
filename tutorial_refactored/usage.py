import torch
from custom_sa.modeling import CustomSaModel, CustomSaConfig
from custom_sa.preprocess import CustomSaPreprocessor

vocab = torch.load("output/vocab.pt")
preprocessor = CustomSaPreprocessor(vocab)

config = CustomSaConfig.from_pretrained("output/best_val_f1/config.json")
model = CustomSaModel.from_pretrained("output/best_val_f1/pytorch_model.bin", config=config)

sentences = [
    "A comedy-drama of nearly epic proportions rooted in a sincere performance by the title character undergoing midlife crisis",
    "Trouble Every Day is a plodding mess"
]

input_tensor = preprocessor(sentences)
output = model(input_tensor)

print(output)
