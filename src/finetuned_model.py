import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoConfig, AutoTokenizer,
                          RobertaForSequenceClassification, pipeline)


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

SEED=42
seed_all(SEED)

data_tl = pd.read_csv("data/processed/data_tl.csv")

model_id = "roberta-base"
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

sentiment_analyzer = pipeline(
    model=model_id,
    tokenizer= "FacebookAI/roberta-base",
    framework="pt",
    task="text-classification",
    device=1
)

dataset = Dataset.from_pandas(data_eval[['id', 'text', 'sentiment', 'predictor', 'confidence']])
print(dataset)

#* Training
# TODO: Split the dataset
# TODO: Train the model
# TODO: Evaluate the model post fine-tuning