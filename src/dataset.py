import pickle

import pandas as pd
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger

# Silence the unreliable language detection warning
polyglot_logger.setLevel("ERROR")

datasets = []
data_temp = []
datasets_num = 8

# Data type encoding
# Unknown -> 0
# LinkedIn post image -> 1
# LinkedIn post video -> 2
# LinkedIn comment image -> 3
# LinkedIn comment video -> 4


def append_item(dataset: list, post_type: str, is_comment: bool, text: str) -> None:
    """Classify type of an element and append it to a dataset with specified type and text

    Args:
        dataset (list): Dataset to which append the data
        post_type (str): LinkedIn post type, can be "image" or "video"
        is_comment (bool): Whether the item is a LinkedIn comment
        text (str): "text" value of specified LinkedIn post/comment
    """
    if post_type == "image" and not is_comment:
        content_type = 1
    elif post_type == "video" and not is_comment:
        content_type = 2
    elif post_type == "image" and is_comment:
        content_type = 3
    elif post_type == "video" and is_comment:
        content_type = 4
    else:
        content_type = 0

    # Data type encoding
    # Unknown -> 0
    # LinkedIn post image -> 1
    # LinkedIn post video -> 2
    # LinkedIn comment image -> 3
    # LinkedIn comment video -> 4

    dataset.append({"id": None, "type": content_type, "text": text, "sentiment": None})


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

        # Save reliably language detected english, person-written, non-repost posts with keyword "openai"
        if (
            "openai" in post_text.lower()
            and post["authorType"] == "Person"
            and not post["isRepost"]
            and post_lang_name == "English"
            and post_detector.reliable
        ):
            append_item(data_temp, post["type"], False, post_text.strip())

        for comment in post["comments"]:
            comment_text = comment["text"]

            comment_detector = Detector(comment_text, quiet=True)
            comment_lang_name = comment_detector.language.name

            # Save reliably language detected english, non-repost post comments with keyword "openai"
            if (
                "openai" in comment_text.lower()
                and comment_detector.reliable
                and comment_lang_name == "English"
                and not post["isRepost"]
            ):
                append_item(data_temp, post["type"], True, comment_text.strip())


df = pd.DataFrame(data_temp)
df = df.drop_duplicates().reset_index(drop=True)

print(len(df))
# print(df.loc[0])

# TODO: Create a new column for posts: numLikes
# TODO: Save df to data/interim
