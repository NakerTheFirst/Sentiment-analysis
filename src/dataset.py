import pickle

import pandas as pd

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
        if post["authorType"] == "Person" and not post["isRepost"]:
            append_item(data_temp, post["type"], False, post["text"])

        for comment in post["comments"]:
            comment_text = comment["text"]

            if "openai" in comment_text.lower() and not post["isRepost"]:
                append_item(data_temp, post["type"], True, comment_text.strip())


df = pd.DataFrame(data_temp)
df = df.drop_duplicates()

print(df)
