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

## Tutorial overview

SGnlp models make use of Huggingface's transformers model interface.
Going through the tutorial will help you understand the additional classes that are needed to
package a model that follows the transformers package interface. To understand more about adding
a Huggingface's transformers model, refer to this [link](https://huggingface.co/docs/transformers/add_new_model).

This tutorial contains a simple sentiment analysis model written in pytorch and data from Kaggle
in the `sample_data` folder. 

The `tutorial` folder contains the pre-refactored code and the `tutorial_refactored` code is the
outcome of the refactoring, where we make the necessary changes to change our pytorch model to fit
the transformers model interface.

## Tutorial steps

1. Refactor to add the config class (`config.py` and `modeling.py`)
2. Refactor train script to add train args class (`train.py` and `train_args.py`)
3. Run train script and save outputs
4. Write a usage script (`usage.py`)
