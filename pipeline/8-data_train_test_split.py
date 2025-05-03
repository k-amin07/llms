import json
import numpy as np
import os
import random


from sklearn.model_selection import train_test_split

# Picks a balanced set of comments from the data (equal number of toxic and non-toxic) and creates train/test splits

current_dir = os.path.dirname(os.path.abspath(__file__))
data = []
with open(
    os.path.join(current_dir, "toxicity_ratings_embeddings_llms.jsonl"), "r"
) as tox_file:
    data = [json.loads(line) for line in tox_file]

data.sort(key=lambda x: x["is_toxic"])

non_toxic_data = list(filter(lambda x: x["is_toxic"] == 0, data))
toxic_data = list(filter(lambda x: x["is_toxic"] == 1, data))

min_length = min(len(toxic_data), len(non_toxic_data))

for run in range(1, 6):
    non_toxic_data_sample = random.sample(non_toxic_data, min_length)
    toxic_data_sample = random.sample(toxic_data, min_length)
    non_toxic_train, non_toxic_test = train_test_split(
        non_toxic_data_sample, test_size=0.2, random_state=run
    )

    toxic_train, toxic_test = train_test_split(
        toxic_data_sample, test_size=0.2, random_state=run
    )
    total_train, total_test = [], []

    total_train.extend(non_toxic_train)
    total_train.extend(toxic_train)
    total_test.extend(non_toxic_test)
    total_test.extend(toxic_test)

    print(
        "Run {}:\n- Total Toxic: {}\n- Total Non-Toxic: {}\n- Toxic Train: {}\n- Non-Toxic Train: {}\n- Toxic Test: {}\n- Non-Toxic Test: {}\n- Total Train: {}\n- Total Test: {}\n".format(
            run,
            len(toxic_data),
            len(non_toxic_data),
            len(toxic_train),
            len(non_toxic_train),
            len(toxic_test),
            len(non_toxic_test),
            len(total_train),
            len(total_test),
        )
    )

    with open(
        os.path.join(current_dir, f"data/run_{run}_train.jsonl"), "w"
    ) as train_file:
        for row in total_train:
            train_file.write(json.dumps(row) + "\n")

    with open(
        os.path.join(current_dir, f"data/run_{run}_test.jsonl"), "w"
    ) as test_file:
        for row in total_test:
            test_file.write(json.dumps(row) + "\n")
