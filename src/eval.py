import gc
import random

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # This method 
    
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
torch.cuda.empty_cache()
gc.collect()

SEED = 42
seed_all(SEED)

#* Select the model to evaluate
model_path = "./models/alfa0/"

#* Read data and convert to Huggingface datasets
dev_df = pd.read_csv("data/processed/dev_df.csv")

dev_dataset = Dataset.from_pandas(dev_df[["text", "label"]])

metric = evaluate.load("accuracy")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#* Tokenize data
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True)

#* Set up training arguments for evaluation
eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=16,
    report_to="none",
)

#* Create trainer
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_dev_dataset,
    compute_metrics=compute_metrics,
)

# * Evaluate
results = trainer.evaluate()
print(results)
