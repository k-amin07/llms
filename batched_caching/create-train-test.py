import json
import numpy as np
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

data = []
current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, "deduped_with_embeddings.json"), "r") as tox_file:
    data = json.load(tox_file)

data.sort(key=lambda x: x["perspective_score"])
perspective_scores = np.array([item["perspective_score"] for item in data])
bins = np.arange(0, 1.1, 0.1)
indices = np.digitize(perspective_scores, bins, right=False)
grouped_data = defaultdict(list)


for item, idx in zip(data, indices):
    bin_start = round(bins[idx - 1], 1)  # Get the bin start value
    grouped_data["{}".format(bin_start)].append(item)


grouped_data = dict(grouped_data)
grouped_data_keys = [
    "0.0",
    "0.1",
    "0.2",
    "0.3",
    "0.4",
    "0.5",
    "0.6",
    "0.7",
    "0.8",
    "0.9",
]
for run in range(1, 6):
    total_train_data = []
    total_test_data = {}

    for key in grouped_data_keys:
        data = grouped_data[key]
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=None)
        total_train_data += train_data
        total_test_data[key] = test_data

    with open(
        os.path.join(current_dir, "data/run_{}_train.json".format(run)), "w+"
    ) as train_file:
        json.dump(total_train_data, train_file, indent=4)
        # train_file.write(json.dumps(total_train_data))
    with open(
        os.path.join(current_dir, "data/run_{}_test.json".format(run)), "w+"
    ) as test_file:
        json.dump(total_test_data, test_file, indent=4)
        # test_file.write(json.dumps(total_test_data))
