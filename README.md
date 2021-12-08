# SG-NLP Workshop

## Setup

Create a virtual environment and install using `requirements.txt`

```
# e.g. using conda
conda create -n sgnlp-workshop python=3.7
conda activate sgnlp-workshop
pip install -r requirements.txt
```

## Running scripts

```
# Train script
python -m tutorial_refactored.custom_sa.train --train_config_path config/train_config.json

# Usage script
python tutorial_refactored/usage.py
```

## Tutorial steps

1. Refactor to add the config class (`config.py` and `modeling.py`)
2. Refactor train script to add train args class (`train.py` and `train_args.py`)
3. Run train script and save outputs
4. Write a usage script (`usage.py`)
