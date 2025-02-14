import os
import json
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from collections import defaultdict
from redisvl.extensions.llmcache import SemanticCache
import numpy as np

from redisvl.utils.vectorize import OpenAITextVectorizer

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAITextVectorizer(
    model="text-embedding-ada-002",
    api_config={"api_key": api_key},
)

data = []
num_lines = 1000
current_dir = os.path.dirname(os.path.abspath(__file__))
redis_url = "redis://localhost:6379"

llmcache = SemanticCache(
    name="eval-zakir",
    prefix="zakir",
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


# according to the paper: "we deem a perspective score of > 0.75 as toxic and all other comments as benign"
# so pick each comment, pass to a small LLM and ask it to rate it as toxic. If the comment is marked as toxic, it
# check if the persepective score is above 0.75

## NVM: perspective API can diverge from participants ratings.

# Compute average ratings per comment and normalize it to a 0-1 scale
# So computed_average - min/max - min =  computed_average - 0 / 4 - 0
# use the same threshold - >0.75 = toxic, less than that = benign

smallLLMs = {
    # "llama": ChatOpenAI(
    #     model="llama3.2:3b",
    #     api_key="ollama",
    #     base_url="http://localhost:11434/v1",
    #     model_kwargs={"response_format": {"type": "json_object"}},
    # ),
    # "gemma": ChatOpenAI(
    #     model="gemma:2b",
    #     api_key="ollama",
    #     base_url="http://localhost:11434/v1",
    #     model_kwargs={"response_format": {"type": "json_object"}},
    # ),
    # "qwen": ChatOpenAI(
    #     model="qwen2.5:3b",
    #     api_key="ollama",
    #     base_url="http://localhost:11434/v1",
    #     model_kwargs={"response_format": {"type": "json_object"}},
    # ),
    "4o": ChatOpenAI(
        model_name="gpt-4o-mini",
        model_kwargs={"response_format": {"type": "json_object"}},
    )
}

small_llm_keys = [
    # "llama",
    # "gemma",
    # "qwen",
    "4o"
]

distance_thresholds = [0.1, 0.2, 0.25, 0.3]
res = {"cache_accuracy": {}}
rmse_array = []

embeddings = []
for key in grouped_data_keys:
    res[key] = []
    rmse_per_llm = {llm: [] for llm in small_llm_keys}
    # we have a 100k rows, divided into 10 bins
    # picking a 1000 total, 100 from each bin
    data = grouped_data[key][:100]
    correct_predictions = {llm: 0 for llm in small_llm_keys}
    correct_cache_predictions = {
        key: {
            llm: {threshold: 0 for threshold in distance_thresholds}
            for llm in small_llm_keys
        }
    }
    cache_hits = {
        llm: {threshold: 0 for threshold in distance_thresholds}
        for llm in small_llm_keys
    }
    total_predictions = 0
    for object in data:
        comment = object["comment"]
        ratings = object["ratings"]

        ## normalized toxic score is toxic_score - min/(max - min). Max is 4, min is 0 so simplifies to toxic_score/4
        ## average is sum of these toxic scores/num raters
        average = sum([i["toxic_score"] / 4 for i in ratings]) / len(ratings)
        # Using median likert score to compute toxicity as described in the paper
        median_likert_score = np.median(np.array([i["toxic_score"] for i in ratings]))
        is_toxic = 1 if median_likert_score >= 3 else 0
        total_predictions += 1
        res[key].append(
            {
                "comment": comment,
                "median_likert_score": median_likert_score,
                "is_toxic": is_toxic,
            }
        )

        for llm in small_llm_keys:
            resp_valid = False
            while resp_valid == False:
                try:
                    print("{} processing comment: {}".format(llm, comment))
                    prompt = """
        Rate this comment "{}" with an appropriate score number from from the following:
        0: not toxic
        1: slightly toxic
        2: moderately toxic
        3: very toxic
        4: highly toxic. 
        Also give a reason for your rating from one of the following
        a. threatening
        b. insulting
        c. profane
        d. attack on a person or group based on their identity
        e. sexual harassment
        f. not toxic
        Return a JSON containing the rating and reason keys""".format(
                        comment
                    )

                    llm_resp = smallLLMs[llm].invoke(comment + prompt).content
                    print(llm_resp.strip())
                    llm_resp = json.loads(llm_resp)
                    llm_score = llm_resp["rating"]
                    llm_reason = llm_resp["reason"]
                    # if llm_resp.content.startswith("f"):
                    #     llm_score = 0
                    # else:
                    #     llm_score = int(llm_resp.content[0])
                    res[key][-1][llm] = llm_score
                    squared_error = (average - (llm_score / 4)) ** 2
                    rmse_per_llm[llm].append(squared_error)
                    res[key][-1]["{}_normalized".format(llm)] = round(llm_score, 2)
                    res[key][-1]["{}_reasoning".format(llm)] = llm_reason
                    llm_is_toxic = 1 if llm_score >= 3 else 0
                    if llm_is_toxic == is_toxic:
                        correct_predictions[llm] += 1
                    for thold in distance_thresholds:
                        llmcache.set_threshold(thold)
                        cache_response = None
                        if resp := llmcache.check(prompt=comment):
                            cache_response = int(resp[0]["response"])
                            cache_hits[llm][thold] += 1
                        if (cache_response is not None) and (
                            llm_is_toxic == cache_response
                        ):
                            correct_cache_predictions[key][llm][thold] += 1
                    resp_valid = True
                    llmcache.store(prompt=comment, response=llm_is_toxic)
                except Exception as e:
                    print(str(e))
                    pass
    res[key + "_rmse"] = {
        llm: np.sqrt(sum(errors) / len(errors)) if errors else 0
        for llm, errors in rmse_per_llm.items()
    }
    res[key + "_accuracy"] = {
        llm: (
            correct_predictions[llm] / total_predictions if total_predictions > 0 else 0
        )
        for llm in small_llm_keys
    }

    res["cache_accuracy"][key] = {
        bin: {
            llm: {
                threshold: correct_cache_predictions[key][llm].get(threshold, 0)
                / total_predictions
                for threshold in correct_cache_predictions[key][llm]
            }
            for llm in small_llm_keys
        }
        for bin in correct_cache_predictions
    }

    # res[key + "_cache_accuracy"] = {
    #     llm: (
    #         correct_cache_predictions[llm] / total_predictions
    #         if total_predictions > 0
    #         else 0
    #     )
    #     for llm in small_llm_keys
    # }


import pandas as pd

rows = []  # List to store flattened data

for key in grouped_data_keys:
    for entry in res[key]:  # Iterate through each comment's data
        row = {
            "key": key,
            "comment": entry["comment"],
            "median_likert_score": entry["median_likert_score"],
            "is_toxic": entry["is_toxic"],
        }

        # Add LLM scores and reasoning
        for llm in small_llm_keys:
            row[f"{llm}_score"] = entry.get(llm, None)
            row[f"{llm}_reasoning"] = entry.get(f"{llm}_reasoning", "")

        rows.append(row)

# Convert to DataFrame and save as CSV
df = pd.DataFrame(rows)
df.to_csv("llm_results.csv", index=False)

print("Comment-level results saved as llm_results.csv")


rmse_rows = []

for key in grouped_data_keys:
    rmse_row = {"key": key}

    for llm in small_llm_keys:
        rmse_row[llm] = res[key + "_rmse"].get(llm, 0)

    rmse_rows.append(rmse_row)

# Convert to DataFrame and save as CSV
df_rmse_acc = pd.DataFrame(rmse_rows)
df_rmse_acc.to_csv("rmse_zakir.csv", index=False)

accuracy_rows = []
for key in grouped_data_keys:
    accuracy_row = {"key": key}

    for llm in small_llm_keys:
        accuracy_row[llm] = res[key + "_accuracy"].get(llm, 0)

    accuracy_rows.append(accuracy_row)

# Convert to DataFrame and save as CSV
df_rmse_acc = pd.DataFrame(accuracy_rows)
df_rmse_acc.to_csv("accuracy_zakir.csv", index=False)

print("RMSE and accuracy results saved as rmse_zakir.csv and accuracy_zakir.csv")
