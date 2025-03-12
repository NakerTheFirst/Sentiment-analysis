import gc
import random

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (AutoConfig, AutoTokenizer,
                          RobertaForSequenceClassification, Trainer,
                          TrainingArguments)


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def freeze_layers(model, num_layers_to_freeze=9):
    """
    Freeze the embeddings and first n layers of the model,
    keep only the last (12-n) layers trainable
    
    Args:
        model: The RoBERTa model
        num_layers_to_freeze: Number of encoder layers to freeze, counting from bottom
    """
    # Freeze embeddings
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze the specified number of encoder layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        if i < num_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    # Print trainable parameters summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model


SEED = 42
seed_all(SEED)

torch.cuda.empty_cache()
gc.collect()

# Load data for cross-validation
eval_df = pd.read_csv("data/processed/data_eval.csv")
print(f"LinkedIn data shape: {eval_df.shape}")

# Display class distribution
print("Label distribution:")
print(eval_df['label'].value_counts())

# Load your already fine-tuned model
base_model_path = "./models/alfa0"
config = AutoConfig.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Load evaluation metric
metric = evaluate.load("accuracy")

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

# Store results
fold_results = []

# Perform cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(eval_df, eval_df['label'])):
    print(f"\n--- Processing Fold {fold+1}/{n_folds} ---")
    
    # Split data for this fold
    fold_train = eval_df.iloc[train_idx].reset_index(drop=True)
    fold_dev = eval_df.iloc[test_idx].reset_index(drop=True)
    
    # Verify label distribution in this fold
    print(f"Training set size: {len(fold_train)}")
    print(f"Dev set size: {len(fold_dev)}")
    print("Train label distribution:", fold_train['label'].value_counts().to_dict())
    print("Dev label distribution:", fold_dev['label'].value_counts().to_dict())
    
    # Convert to datasets
    fold_train_dataset = Dataset.from_pandas(fold_train[["text", "label"]])
    fold_dev_dataset = Dataset.from_pandas(fold_dev[["text", "label"]])
    
    # Tokenize datasetsa
    tokenized_fold_train = fold_train_dataset.map(tokenize_function, batched=True)
    tokenized_fold_dev = fold_dev_dataset.map(tokenize_function, batched=True)
    
    # Load a fresh copy of the pre-trained model for each fold
    fold_model = RobertaForSequenceClassification.from_pretrained(
        base_model_path, config=config, torch_dtype="auto"
    ).to(device)
    
    fold_model = freeze_layers(fold_model, num_layers_to_freeze=10)
    
    # Training arguments for the second fine-tuning stage
    fold_training_args = TrainingArguments(
        output_dir=f"./models/cv_frozen_k3_{fold+1}",
        per_device_train_batch_size=8,  # Smaller batch for limited data
        per_device_eval_batch_size=8,
        num_train_epochs=8,  
        learning_rate=5e-5, 
        weight_decay=0.01, 
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=0,
        save_total_limit=1,
        load_best_model_at_end=True,
        save_only_model=True,
        metric_for_best_model="accuracy",
        report_to="none", 
        max_grad_norm=1.0,
    )
    
    fold_trainer = Trainer(
        model=fold_model,
        args=fold_training_args,
        train_dataset=tokenized_fold_train,
        eval_dataset=tokenized_fold_dev,
        compute_metrics=compute_metrics,
    )
    
    # Fine-tune the model on this fold
    fold_trainer.train()
    
    # Evaluate
    fold_eval_results = fold_trainer.evaluate()
    print(f"Fold {fold+1} Results: {fold_eval_results}")
    fold_results.append(fold_eval_results)
    
    # Save the model for this fold
    # fold_trainer.save_model(f"./models/cv_fold_delta_{fold+1}")
    
    # Clear CUDA cache between folds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Calculate average performance across folds
avg_accuracy = np.mean([result["eval_accuracy"] for result in fold_results])
avg_loss = np.mean([result["eval_loss"] for result in fold_results])
std_accuracy = np.std([result["eval_accuracy"] for result in fold_results])

print("\n--- Cross-Validation Summary ---")
print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Average Loss: {avg_loss:.4f}")
print("\nPer-fold results:")
for i, result in enumerate(fold_results):
    print(f"Fold {i+1}: Accuracy = {result['eval_accuracy']:.4f}, Loss = {result['eval_loss']:.4f}")

# Identify the best performing fold
best_fold = np.argmax([result["eval_accuracy"] for result in fold_results]) + 1
print(f"\nBest performing model: Fold {best_fold} with accuracy {fold_results[best_fold-1]['eval_accuracy']:.4f}")
best_fold = np.argmax([result["eval_accuracy"] for result in fold_results]) + 1
