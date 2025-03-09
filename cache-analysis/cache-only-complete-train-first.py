import json
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.extensions.llmcache import SemanticCache
from sklearn.model_selection import train_test_split

### Train test split the zakir dataset
### Put the traint data into the cache
### Check accuracy for test data


###### TO DO
###### ALSO SAVE THE QUERY, LIKERT SCORE, TOXICITY RATING BASED ON LIKERT SCORE, TOXICITY RATING BASED ON CACHE FOR EACH CACHE THRESHOLD

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAITextVectorizer(
    model="text-embedding-ada-002",
    api_config={"api_key": api_key},
)

data = []
current_dir = os.path.dirname(os.path.abspath(__file__))
redis_url = "redis://localhost:6379"

llm_cache = SemanticCache(
    name="zakir-cache-complete-train-first",
    prefix="zakir-cache-complete-train-first",
    redis_url=redis_url,
    distance_threshold=0.1,
    vectorizer=client,
)

with open(os.path.join(current_dir, "toxicity_ratings.json"), "r") as tox_file:
    lines = tox_file.readlines()
    for line in lines:
        line = line.strip()
        data_json = json.loads(line)
        data_json["perspective_score"] = float(data_json["perspective_score"])
        data.append(data_json)

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

distance_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# distance_thresholds = [0.7604798184] # 1 minus this is optimal so far
res = {}
res_per_bin = {}

for key in grouped_data_keys:
    res_per_bin[key] = {}
    for thold in distance_thresholds:
        sim = 1 - thold
        res_per_bin[key][sim] = {"hits": 0, "total_queries": 0, "correct": 0}

for thold in distance_thresholds:
    sim = 1 - thold
    res[sim] = {"hits": 0, "total_queries": 0, "correct": 0}

total_train_data = []
total_test_data = {}

for key in grouped_data_keys:
    data = grouped_data[key]
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=None)
    total_train_data += train_data
    total_test_data[key] = test_data

print("Processing train_data")
for object in total_train_data:
    comment = object["comment"]
    ratings = object["ratings"]
    ## normalized toxic score is toxic_score - min/(max - min). Max is 4, min is 0 so simplifies to toxic_score/4
    ## average is sum of these toxic scores/num raters
    median_likert_score = np.median(np.array([i["toxic_score"] for i in ratings]))
    is_toxic = 1 if median_likert_score >= 3 else 0
    llm_cache.store(prompt=comment, response=is_toxic)

for key in grouped_data_keys:
    print("Processing test_data for group {}".format(key))
    for object in total_test_data[key]:
        comment = object["comment"]
        ratings = object["ratings"]
        ## normalized toxic score is toxic_score - min/(max - min). Max is 4, min is 0 so simplifies to toxic_score/4
        ## average is sum of these toxic scores/num raters
        median_likert_score = np.median(np.array([i["toxic_score"] for i in ratings]))
        is_toxic = 1 if median_likert_score >= 3 else 0
        for thold in distance_thresholds:
            sim = 1 - thold
            llm_cache.set_threshold(thold)
            cache_response = None
            if resp := llm_cache.check(prompt=comment):
                cache_response = int(resp[0]["response"])
                res[sim]["hits"] += 1
                res_per_bin[key][sim]["hits"] += 1

            res[sim]["total_queries"] += 1
            res_per_bin[key][sim]["total_queries"] += 1
            if (cache_response is not None) and (is_toxic == cache_response):
                res[sim]["correct"] += 1
                res_per_bin[key][sim]["correct"] += 1

df = pd.DataFrame.from_dict(res, orient="index").reset_index()
df.rename(columns={"index": "threshold"}, inplace=True)
df.to_csv("complete-train-first-output.csv", index=False)

rows = []
for bin_key, thresholds in res_per_bin.items():
    for threshold, values in thresholds.items():
        row = {
            "bin": bin_key,
            "threshold": threshold,
            "hits": values["hits"],
            "correct": values["correct"],
        }
        rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv("complete-train-first-res_per_bin.csv", index=False)

print(df)
