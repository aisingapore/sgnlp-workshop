import json

import torch


class CustomSaPostprocessor:
    def __init__(self, labels_map_path):
        self.labels_map_path = labels_map_path
        with open(labels_map_path) as f:
            self.labels_map = json.load(f)
            self.labels_map = {int(k): v for k, v in self.labels_map.items()}

    def __call__(self, model_output, sentences=None):
        label_ids = torch.argmax(
            torch.softmax(model_output.logits, dim=1), dim=1
        ).tolist()
        label_desc = [self.labels_map[label_id] for label_id in label_ids]
        if sentences is not None:
            return [
                {"sentence": sentence, "label": desc}
                for sentence, desc in zip(sentences, label_desc)
            ]
        return label_desc
