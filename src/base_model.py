import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, multilabel_confusion_matrix
from transformers import pipeline, AutoConfig, AutoTokenizer, RobertaForSequenceClassification
import torch

def map_sentiment_scores(label: str) -> int:
    """Maps RoBERTa's sentiment labels to numeric format"""
    mapping = {
        'LABEL_0': 0,  # POSITIVE -> 0
        'LABEL_1': 1,  # NEUTRAL -> 1
        'LABEL_2': 2   # NEGATIVE -> 2
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
# with open(r"data/processed/data_internal.bin", "rb") as data_file:
    # data_eval = pickle.load(data_file)

data_eval = pd.read_csv('data/processed/data_eval.csv')

model_id = "roberta-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(model_id, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = RobertaForSequenceClassification.from_pretrained(
    model_id, config=config, torch_dtype="auto"
).to(device)

sentiment_analyzer = pipeline(
    task="text-classification",
    model=model,
    config=config,
    tokenizer= tokenizer,
    framework="pt",
    device=device
)

# Predict sentiment and confidence
sentiment_results = [analyze_sentiment(text) for text in data_eval['text']]
data_eval['predictor'] = [result[0] for result in sentiment_results]
data_eval['confidence'] = [result[1] for result in sentiment_results]

# Calculate metrics
accuracy = accuracy_score(data_eval['label'], data_eval['predictor'])
balanced_accuracy = balanced_accuracy_score(data_eval['label'], data_eval['predictor'])
confidence = data_eval['confidence'].mean()

print(f"\nAccuracy: {accuracy}")
print(f"Balanced accuracy: {balanced_accuracy}")
print(f"Mean confidence: {confidence}")

labels = [0, 1, 2]
cm = multilabel_confusion_matrix(data_eval['label'], data_eval['predictor'], labels=labels)
print(cm)

# Save dataset to CSV
# data_eval.to_csv('data/processed/base_model_predictions.csv', index=False)
