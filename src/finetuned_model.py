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

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

SEED=42
seed_all(SEED)

train_df = pd.read_csv("data/processed/data_tl.csv")
test_df = pd.read_csv("data/processed/data_eval.csv")

# Split the evaluation data into test and dev sets
dev_df = test_df[:200]
test_df = test_df[200:]
dev_df.reset_index()
test_df.reset_index()

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

train_dataset = Dataset.from_pandas(train_df[['id', 'text', 'sentiment', 'predictor', 'confidence']])
dev_dataset = Dataset.from_pandas(dev_df[['id', 'text', 'sentiment', 'predictor', 'confidence']])
test_dataset = Dataset.from_pandas(test_df[['id', 'text', 'sentiment', 'predictor', 'confidence']])

print(train_dataset)
print(dev_dataset)
print(test_dataset)

#* Training
# TODO: Train the model
# TODO: Evaluate the model post fine-tuning