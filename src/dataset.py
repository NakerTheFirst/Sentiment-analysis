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
            "sentiment": None,
            "likes_num": likes_count,
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

# Create dataframe, drop duplicates, replace URLs with <URL> token, drop URL-only content
data_interim = pd.DataFrame(data_temp)
data_interim = data_interim.drop_duplicates()
data_interim["text"] = data_interim["text"].str.replace(
    url_pattern, "<URL>", regex=True
)
data_interim = data_interim.drop(data_interim[data_interim["text"] == "<URL>"].index)

# Shuffle & reindex the data
data_interim = data_interim.sample(frac=1).reset_index(drop=True)
data_interim["id"] = range(1, len(data_interim) + 1)

print(data_interim.head(), "\n")
print(f"Length of dataset: {len(data_interim)}\n")
# print(data_interim.loc[data_interim["text"].str.contains("<URL>")])

# TODO: Save the df to data/interim
