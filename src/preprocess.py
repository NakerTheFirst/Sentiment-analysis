import pickle

import pandas as pd
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger

# Silence the unreliable language detection warning
polyglot_logger.setLevel("ERROR")

datasets = []
data_temp = []
datasets_num = 8

# Define the regex pattern for URLs
url_pattern = (
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def clean_dataset_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the text data, remove trailing and leading whitespace, 
        tokenise URLs into "<URL>" tags, drop duplicate rows based on "text" column

    Args:
        df (pd.DataFrame): Dataset which text is to be cleaned

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    
    # Clean and standardise the text
    df["text"] = (
        df["text"]
        .str.strip()
        .str.replace("\s+", " ")  # type: ignore
        .str.replace(url_pattern, "<URL>", regex=True)
    )

    # Drop duplicates after text standardisation
    df = df.drop_duplicates(subset=["text"], keep="first")

    # Drop entries that are just URLs
    df = df[df["text"] != "<URL>"]

    return df


def save_relevant_cols(
    dataset: list, item: dict, parent_post: dict | None = None
) -> None:
    """Extract relevant columns from an item and save them in specified dataset

    content_type data encoding:
        Unknown -> 0
        LinkedIn post image -> 1
        LinkedIn post video -> 2
        LinkedIn comment image -> 3
        LinkedIn comment video -> 4

    Args:
        dataset (list): Dataset to which append extracted data
        item (dict): A raw post or a raw comment from which the data is extracted
        parent_post (dict | None, optional): Parent post of a comment. Defaults to None.
    """

    # Posts
    if not parent_post:
        likes_count = int(item["numLikes"])

        if item["type"] == "image":
            content_type = 1
        elif item["type"] == "linkedinVideo":
            content_type = 2
        else:
            content_type = 0

    # Comments
    if parent_post:
        likes_count = None

        if parent_post["type"] == "image":
            content_type = 3
        elif parent_post["type"] == "linkedinVideo":
            content_type = 4
        else:
            content_type = 0

    dataset.append(
        {
            "id": None,
            "type": content_type,
            "text": item["text"].strip(),
            "likes_num": likes_count,
            "sentiment": None,
            "confidence": None,
        }
    )


# Load the raw data
for x in range(datasets_num):
    with open(rf"data/raw/data_raw{x+1}.bin", "rb") as data_file:
        datasets.append(pickle.load(data_file))
        data_file.seek(0)

# Preprocess the data
for dataset_index in range(datasets_num):
    for post in datasets[dataset_index]:
        post_text = post["text"]

        # Detect post language
        post_detector = Detector(post_text, quiet=True)
        post_lang_name = post_detector.language.name

        # Save reliably language detected, english, person-written, non-repost posts with keyword "openai"
        if (
            "openai" in post_text.lower()
            and post["authorType"] == "Person"
            and not post["isRepost"]
            and post_lang_name == "English"
            and post_detector.reliable
        ):
            save_relevant_cols(data_temp, post)

        for comment in post["comments"]:
            comment_text = comment["text"]

            comment_detector = Detector(comment_text, quiet=True)
            comment_lang_name = comment_detector.language.name

            # Save reliably language detected, english, non-repost post comments with keyword "openai"
            if (
                "openai" in comment_text.lower()
                and comment_detector.reliable
                and comment_lang_name == "English"
                and not post["isRepost"]
            ):
                save_relevant_cols(data_temp, comment, post)

# Create dataframe, clean text data, tokenise URLs, drop duplicates
in_interim = pd.DataFrame(data_temp)
in_interim = clean_dataset_text(in_interim)

# Shuffle the data
in_interim = in_interim.sample(frac=1).reset_index(drop=True)

# Split the df 
in_to_label = in_interim.iloc[:500] 
in_unlabelled = in_interim.iloc[500:] 

# Reset the index for both dfs
in_to_label = in_to_label.reset_index(drop=True)
in_unlabelled = in_unlabelled.reset_index(drop=True)

in_to_label["id"] = range(1, len(in_to_label) + 1)
in_unlabelled["id"] = range(1, len(in_unlabelled) + 1)

# Save the preprocessed, unlabelled data
# with open(r"data/interim/in_unlabelled.bin", "wb") as filehandler:
    # pickle.dump(in_unlabelled, filehandler)

# Save the preprocessed, data to be labelled
# with open(r"data/interim/in_labelled.bin", "wb") as filehandler:
    # pickle.dump(in_to_label, filehandler)

# Preprocess the Transfer Learning data as to match text and sentiment columns

# Read the CSV
df = pd.read_csv("data/raw/social_media_sentiment_dataset.csv", encoding="ISO-8859-1")

# Rename columns
df = df.rename(columns={
    'tweet_text': 'text',
    'emotion_in_tweet_is_directed_at': 'sentiment_focus',
    'is_there_an_emotion_directed_at_a_brand_or_product': 'sentiment'
})

# Create a mapping dict
sentiment_mapping = {
    'Positive emotion': 1,
    'No emotion toward brand or product': 2,
    'I can\'t tell': 2, 
    'Negative emotion': 3
}

# Replace values in the existing sentiment column
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# Verify the conversion
# print(df['sentiment'].value_counts())
# print(df.head())

# Check initial size
print(f"Original size: {len(df)}")

# Remove sentiment_focus column, drop duplicates & NaN values
df = df.drop(columns=["sentiment_focus"])
df = df.drop_duplicates()
df = df.dropna(subset=["text", "sentiment"])
print(f"Size after removing duplicates and NaN values: {len(df)}\n")

# Reset index after dropping rows
df = df.reset_index(drop=True)
df['id'] = range(1, len(df)+1)
df = df[['id', 'text', 'sentiment']]

# Read the internal labelled dataset
with open(r"data/interim/in_labelled.bin", "rb") as data_file:
    in_labelled = pickle.load(data_file)

in_labelled = in_labelled[['id', 'text', 'sentiment', 'confidence']]

with open(r"data/interim/in_labelled_processed.bin", "wb") as filehandler:
    pickle.dump(in_labelled, filehandler)
    