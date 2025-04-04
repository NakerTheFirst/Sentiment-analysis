import pickle

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

# Build the language detector
languages = [Language.ENGLISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

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
        .str.replace(r"{link}", "<URL>", regex=True)
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


#* Preprocess internal dataset
# Load the raw data
for x in range(datasets_num):
    with open(rf"data/raw/data_raw{x+1}.bin", "rb") as data_file:
        datasets.append(pickle.load(data_file))
        data_file.seek(0)

# Preprocess the data
for dataset_index in range(datasets_num):
    for post in datasets[dataset_index]:
        post_text = post["text"]

        # Detect post language using lingua
        detected_language = detector.detect_language_of(post_text)
        is_reliable = detected_language is not None
        is_english = is_reliable and detected_language == Language.ENGLISH

        # Save reliably language detected, english, person-written, non-repost posts with keyword "openai"
        if (
            "openai" in post_text.lower()
            and post["authorType"] == "Person"
            and not post["isRepost"]
            and is_english
            and is_reliable
        ):
            save_relevant_cols(data_temp, post)

        for comment in post["comments"]:
            comment_text = comment["text"]

            # Detect comment language using lingua
            detected_comment_language = detector.detect_language_of(comment_text)
            is_comment_reliable = detected_comment_language is not None
            is_comment_english = is_comment_reliable and detected_comment_language == Language.ENGLISH

            # Save reliably language detected, english, non-repost post comments with keyword "openai"
            if (
                "openai" in comment_text.lower()
                and is_comment_reliable
                and is_comment_english
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

#! TODO: DO NOT EVER uncomment this, else all the data labells will be lost
# Save the preprocessed data to be labelled
# with open(r"data/interim/in_labelled.bin", "wb") as filehandler:
    # pickle.dump(in_to_label, filehandler)

with open(r"data/interim/in_labelled.bin", "rb") as data_file:
    in_labelled = pickle.load(data_file)

in_labelled['predictor'] = None
data_internal = in_labelled[['id', 'text', 'sentiment', 'predictor', 'confidence']]
data_internal = data_internal.rename(columns={'sentiment': 'label'})

data_internal['label'] = data_internal['label'].map({1: 0, 2: 1, 3: 2})
data_internal['label'] = data_internal['label'].astype(int)

#! TODO: Uncomment before submitting the final project version to APD - this preprocesses in_labelled.bin data into data_eval.csv
# Save the processed internal labelled dataset as .csv
# data_internal.to_csv("data/processed/data_eval.csv", index=False)

#* Preprocess the TL dataset 
tl_df = pd.read_csv("data/raw/social_media_sentiment_dataset.csv", encoding="MacRoman")
tl_df = tl_df.rename(columns={
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
tl_df['sentiment'] = tl_df['sentiment'].map(sentiment_mapping)

# Tokenise the ULRs
tl_df = clean_dataset_text(tl_df)

tl_df = tl_df.drop(columns=["sentiment_focus"])
tl_df = tl_df.dropna(subset=["text", "sentiment"])
tl_df = tl_df.reset_index(drop=True)

tl_df['id'] = range(1, len(tl_df)+1)
tl_df['confidence'] = None
tl_df['predictor'] = None

tl_df = tl_df.rename(columns={'sentiment': 'label'})
tl_df = tl_df[['id', 'text', 'label', 'predictor', 'confidence']]

tl_df['label'] = tl_df['label'].map({1: 0, 2: 1, 3: 2})
tl_df["label"] = tl_df["label"].astype(int)

tl_df = tl_df.sample(frac=1).reset_index(drop=True)

#! TODO: Uncomment before submitting the final project version to APD - this preprocesses data_eval.csv into data_tl.csv
# Save the processed TL dataset
# tl_df.to_csv("data/processed/data_tl.csv", index=False)
