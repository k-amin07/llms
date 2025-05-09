import json
import numpy as np
import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "deduped_with_embeddings.json"), "r") as tox_file:
    data = json.load(tox_file)

# Bin the data
data.sort(key=lambda x: x["perspective_score"])
perspective_scores = np.array([item["perspective_score"] for item in data])
bins = np.arange(0, 1.1, 0.1)
indices = np.digitize(perspective_scores, bins, right=False)
grouped_data = defaultdict(list)

for item, idx in zip(data, indices):
    bin_start = round(bins[idx - 1], 1)
    grouped_data[f"{bin_start}"].append(item)

grouped_data = dict(grouped_data)

# Bin groups
non_toxic_bins = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
toxic_bins = ["0.7", "0.8", "0.9"]

# Compute bin lengths dynamically
non_toxic_bin_lengths = [len(grouped_data[bin_key]) for bin_key in non_toxic_bins]
toxic_bin_lengths = [len(grouped_data[bin_key]) for bin_key in toxic_bins]

# Compute balanced sample sizes
min_non_toxic = min(non_toxic_bin_lengths)
min_toxic = min(toxic_bin_lengths)
overall_min = min(min_non_toxic, min_toxic)

num_toxic_bins = len(toxic_bins)
num_non_toxic_bins = len(non_toxic_bins)

max_total_per_group = overall_min * num_toxic_bins
TOXIC_SAMPLES_PER_BIN = overall_min
NON_TOXIC_SAMPLES_PER_BIN = max_total_per_group // num_non_toxic_bins

print(f"TOXIC_SAMPLES_PER_BIN = {TOXIC_SAMPLES_PER_BIN}")
print(f"NON_TOXIC_SAMPLES_PER_BIN = {NON_TOXIC_SAMPLES_PER_BIN}")


# Sampling helper
def safe_sample(data_list, sample_size):
    return random.sample(data_list, min(sample_size, len(data_list)))


# Create output dir
os.makedirs(os.path.join(current_dir, "data"), exist_ok=True)

# Repeatable random runs
for run in range(1, 6):
    total_train_data = []
    total_test_data = {}

    # Sample non-toxic bins
    for bin_key in non_toxic_bins:
        sampled = safe_sample(grouped_data[bin_key], NON_TOXIC_SAMPLES_PER_BIN)
        train_data, test_data = train_test_split(
            sampled, test_size=0.2, random_state=run
        )
        total_train_data.extend(train_data)
        total_test_data[bin_key] = test_data

    # Sample toxic bins
    for bin_key in toxic_bins:
        sampled = safe_sample(grouped_data[bin_key], TOXIC_SAMPLES_PER_BIN)
        train_data, test_data = train_test_split(
            sampled, test_size=0.2, random_state=run
        )
        total_train_data.extend(train_data)
        total_test_data[bin_key] = test_data

    # Save files
    with open(
        os.path.join(current_dir, f"data/run_{run}_train.json"), "w"
    ) as train_file:
        json.dump(total_train_data, train_file, indent=4)

    with open(os.path.join(current_dir, f"data/run_{run}_test.json"), "w") as test_file:
        json.dump(total_test_data, test_file, indent=4)
