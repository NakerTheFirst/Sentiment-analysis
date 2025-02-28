import gc
import random

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoConfig, AutoTokenizer,
                          RobertaForSequenceClassification, Trainer,
                          TrainingArguments)


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


torch.cuda.empty_cache()
gc.collect()

SEED = 42
seed_all(SEED)

train_df = pd.read_csv("data/processed/data_tl.csv")
test_df = pd.read_csv("data/processed/data_eval.csv")

# *Split the data into train/dev/test sets using 40/30/30 ratio
additional_train_df = test_df[300:]
test_df = test_df[:300]

train_df = pd.concat([train_df, additional_train_df])

train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df['id'] = range(1, len(train_df)+1)

dev_df = test_df[:150]
test_df = test_df[150:]

train_df.to_csv("data/processed/train_df.csv")
dev_df.to_csv("data/processed/dev_df.csv")
test_df.to_csv("data/processed/test_df.csv")

# Enable CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n\nUsing device: {device}\n\n")

# * Load the model, tokenizer and config
model_id = "roberta-base"
config = AutoConfig.from_pretrained(model_id, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = RobertaForSequenceClassification.from_pretrained(
    model_id, config=config, torch_dtype="auto"
).to(device)
model.train()

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
dev_dataset = Dataset.from_pandas(dev_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

# * Tokenise the data
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# * Training
metric = evaluate.load("accuracy")

training_args = TrainingArguments(
    output_dir="a0_wow1",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    skip_memory_metrics=True,
    disable_tqdm=False,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=0,
    save_total_limit=0,
    save_only_model=False,
    num_train_epochs=10,
    learning_rate=4e-5,
    weight_decay=0.1,
    adam_beta1=0.95,
    adam_beta2=0.999,
    logging_steps=100,
    log_level="warning",
)

eval_args = TrainingArguments(
    output_dir=None,
    per_device_eval_batch_size=16,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the model
model_save_path = "./models/alfa0"
trainer.save_model(model_save_path)  # Save model, tokeniser and config
tokenizer.save_pretrained(model_save_path)  

# TODO: Rename finetuned_model.py into train.py
# TODO: Refactor base_model.py into train.py
# TODO: Change Roberta into Roberta for seq classification