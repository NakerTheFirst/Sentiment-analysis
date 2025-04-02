# LinkedIn Opinion Analysis
## Project Overview
This project conducts sentiment analysis on public opinions about OpenAI shared on LinkedIn. Using natural language processing (NLP) techniques, the system analyzes posts and comments to categorize sentiment as positive, neutral, or negative. The project was developed as part of an engineering thesis with a focus on educational purposes.

## Features
- Data collection from LinkedIn using Apify scraper
- Robust text preprocessing to handle URLs, slang, multilingual content
- Manual labelling functionality for creating training datasets
- Transfer Learning with RoBERTa model for sentiment classification
- Cross-validation to evaluate model performance
- Data visualization with histograms and word clouds

## Technical Architecture
The project follows a complete machine learning pipeline:
1. **Data Collection**: Scraping LinkedIn posts and comments containing "openai"
2. **Preprocessing**: Text cleaning, language detection, URL tokenization
3. **Labelling**: Manual sentiment annotation interface
4. **Model Training**: Fine-tuning RoBERTa with Transfer Learning
5. **Evaluation**: Using cross-validation and accuracy metrics
6. **Visualization**: Displaying sentiment distributions and word frequencies

## Data Processing
- English language detection using Lingua
- URL standardization and removal of duplicates
- Sentiment categorization (Positive, Neutral, Negative)

## Models
The sentiment analysis uses:
- Base model: RoBERTa
- Transfer Learning: Fine-tuning on domain-specific data
- Layer freezing: 10 bottom layers frozen to preserve general language understanding

## Concepts Explored
The project explores several key data science and NLP concepts:
- **Web Scraping**: Ethical data collection techniques from social media platforms
- **Natural Language Processing**: Text preprocessing, tokenization, and sentiment analysis
- **Transfer Learning**: Adapting pre-trained language models to new domains
- **Feature Engineering**: Extracting relevant features from text data
- **Model Evaluation**: Cross-validation techniques and performance metrics
- **Class Imbalance**: Handling uneven distribution of sentiment classes
- **Hyperparameter Tuning**: Optimizing model parameters for better performance
- **Data Visualization**: Techniques for presenting text data analysis results

## Requirements
- Python 3.10
- PyTorch
- Transformers (Hugging Face)
- pandas
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- Apify client
- Lingua

## Installation
```
pip install -r requirements.txt
```
Additionally, the project requires pytorch package with GPU compatible Nvidia CUDA version, which can differ across devices. Check your GPU's CUDA version with: `nvidia-smi` and install [compatible pytorch package](https://pytorch.org/).
