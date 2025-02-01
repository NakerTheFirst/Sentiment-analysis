import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import AutoConfig, RobertaForSequenceClassification, pipeline


def map_sentiment_scores(label: str) -> int:
    """Maps RoBERTa's sentiment labels to your numeric format"""
    mapping = {
        'LABEL_0': 3,  # NEGATIVE -> 3
        'LABEL_1': 2,  # NEUTRAL -> 2
        'LABEL_2': 1   # POSITIVE -> 1
    }
    return mapping[label]

def analyze_sentiment(text: str) -> tuple[int, float]:
    """Analyze sentiment of text using RoBERTa model
    
    Returns:
        tuple[int, float]: (prediction, confidence_score)
    """
    result = sentiment_analyzer(text, truncation=True)[0]
    prediction = map_sentiment_scores(result['label'])
    confidence = result['score']
    
    return prediction, confidence

# Load the data
with open(r"data/processed/data_internal.bin", "rb") as data_file:
    data_eval = pickle.load(data_file)

data_tl = pd.read_csv("data/processed/data_tl.csv")

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline(
    model="roberta-base",
    tokenizer= "FacebookAI/roberta-base",
    framework="pt",
    task="text-classification",
    device=1
)

model_id = "roberta-base"
config = AutoConfig.from_pretrained(model_id)
model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)

# Apply sentiment analysis to your DataFrame
sentiment_results = [analyze_sentiment(text) for text in data_eval['text']]
data_eval['predictor'] = [result[0] for result in sentiment_results]
data_eval['confidence'] = [result[1] for result in sentiment_results]

# Explicitly cast predictors and actuals to int
data_eval['predictor'] = data_eval['predictor'].astype(int)
data_eval['sentiment'] = data_eval['sentiment'].astype(int)

labels = [1, 2, 3]  # Positive, neutral, negative

# Accuracy
accuracy = accuracy_score(data_eval['sentiment'], data_eval['predictor'])
balanced_accuracy = balanced_accuracy_score(data_eval['sentiment'], data_eval['predictor'])

# Calculate confidence
confidence = data_eval['confidence'].mean()

print(f"\nAccuracy: {accuracy}")
print(f"Balanced accuracy: {balanced_accuracy}")
print(f"Mean confidence: {confidence}")

# TODO: Save the evaluation data 
# TODO: Visualise evaluation data (is this really necessary?) 
# TODO: Train = fine-tune the model
# TODO: Evaluate the model

# Save dataset to CSV
data_eval.to_csv('data/processed/data_eval_predictions.csv', index=False)

#* Training
# training_args = TrainingArguments(
#     output_dir=repository_id,
#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     evaluation_strategy="epoch",
#     logging_dir=f"{repository_id}/logs",
#     logging_strategy="steps",
#     logging_steps=10,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     warmup_steps=500,
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     save_total_limit=2,
#     report_to="tensorboard",
#     push_to_hub=True,
#     hub_strategy="every_save",
#     hub_model_id=repository_id,
#     hub_token=HfFolder.get_token(),
# )
