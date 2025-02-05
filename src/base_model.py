import pickle

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import pipeline


def map_sentiment_scores(label: str) -> int:
    """Maps RoBERTa's sentiment labels to numeric format"""
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
model_id = "roberta-base"

sentiment_analyzer = pipeline(
    model=model_id,
    tokenizer= "FacebookAI/roberta-base",
    framework="pt",
    task="text-classification",
    device=1
)

# Predict sentiment and confidence
sentiment_results = [analyze_sentiment(text) for text in data_eval['text']]
data_eval['predictor'] = [result[0] for result in sentiment_results]
data_eval['confidence'] = [result[1] for result in sentiment_results]

# Explicitly cast predictors and actuals to int
data_eval['predictor'] = data_eval['predictor'].astype(int)
data_eval['sentiment'] = data_eval['sentiment'].astype(int)

# Calculate metrics
accuracy = accuracy_score(data_eval['sentiment'], data_eval['predictor'])
balanced_accuracy = balanced_accuracy_score(data_eval['sentiment'], data_eval['predictor'])
confidence = data_eval['confidence'].mean()

print(f"\nAccuracy: {accuracy}")
print(f"Balanced accuracy: {balanced_accuracy}")
print(f"Mean confidence: {confidence}")

# Save dataset to CSV
data_eval.to_csv('data/processed/base_model_predictions.csv', index=False)
