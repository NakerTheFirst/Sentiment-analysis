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

# Create dataframe, drop duplicates, replace URLs with <URL> token, drop URL-only content
in_interim = pd.DataFrame(data_temp)
in_interim = in_interim.drop_duplicates()
in_interim["text"] = in_interim["text"].str.replace(url_pattern, "<URL>", regex=True)
in_interim = in_interim.drop(in_interim[in_interim["text"] == "<URL>"].index)

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
with open(r"data/interim/in_unlabelled.bin", "wb") as filehandler:
    pickle.dump(in_unlabelled, filehandler)

# Save the preprocessed, data to be labelled
with open(r"data/interim/in_to_label.bin", "wb") as filehandler:
    pickle.dump(in_to_label, filehandler)
