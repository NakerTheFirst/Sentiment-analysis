import random

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoConfig, AutoTokenizer,
                          RobertaForSequenceClassification, Trainer,
                          TrainingArguments, pipeline)


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

SEED=42
seed_all(SEED)

train_df = pd.read_csv("data/processed/data_tl.csv")
test_df = pd.read_csv("data/processed/data_eval.csv")

#* Split the data into test/dev sets with 60/40 ratio
dev_df = test_df[:200]
test_df = test_df[200:]

#* Load the model, tokenizer and config
model_id = "roberta-base"
config = AutoConfig.from_pretrained(model_id, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = RobertaForSequenceClassification.from_pretrained(model_id, config=config, torch_dtype="auto")

model.train()

sentiment_analyzer = pipeline(
    model=model_id,
    tokenizer= "FacebookAI/roberta-base",
    framework="pt",
    task="text-classification",
    device=1,
    num_labels=3
)

train_dataset = Dataset.from_pandas(train_df[['id', 'text', 'label', 'predictor', 'confidence']])
dev_dataset = Dataset.from_pandas(dev_df[['id', 'text', 'label', 'predictor', 'confidence']])
test_dataset = Dataset.from_pandas(test_df[['id', 'text', 'label', 'predictor', 'confidence']])

#* Tokenise the data
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

#* Training
metric = evaluate.load("accuracy")

training_args = TrainingArguments(
    output_dir="test_trainer", 
    log_level='debug', 
    eval_strategy="steps"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    compute_metrics=metric
)

trainer.train()

# TODO: Train the model
# TODO: Evaluate the model
# TODO: Tune the model
# TODO: Evaluate the model post hyperparameter tuning
