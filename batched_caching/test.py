import argparse
import json
import numpy as np
import os
from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.extensions.llmcache import SemanticCache
import pandas as pd

import asyncio


parser = argparse.ArgumentParser(description="Process toxicity dataset cache results.")
parser.add_argument("-f", "--file_name", type=str, help="Path to the input JSON file")
args = parser.parse_args()

api_key = os.environ.get("OPENAI_API_KEY")

client = OpenAITextVectorizer(
    model="text-embedding-3-small", api_config={"api_key": api_key}
)
redis_url = "redis://localhost:6379"

llm_cache = SemanticCache(
    name="zakir-cache",
    prefix="zakir-cache",
    redis_url=redis_url,
    distance_threshold=0.1,
    vectorizer=client,
)

## Restructure this to use batches.
# pick a batch of queries, test them at threshold of 0.95, then test them at 0.9 etc
# repeat for the next batch

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
res = {}
res_per_bin = {}
res_actual_matches = {}
batch_size = 25

for thold in distance_thresholds:
    sim = 1 - thold
    res[sim] = {"hits": 0, "total_queries": 0, "correct": 0}
    res_per_bin[sim] = {}
    res_actual_matches[sim] = []

for key in grouped_data_keys:
    for thold in distance_thresholds:
        sim = 1 - thold
        res_per_bin[sim][key] = {"hits": 0, "total_queries": 0, "correct": 0}


async def process_test_data(test_data):
    for thold in distance_thresholds:
        sim = 1 - thold
        for key in grouped_data_keys:
            stored_tasks = []
            count = 1
            cache_dict = []
            for object in test_data[key]:
                comment = object["comment"]
                ratings = object["ratings"]
                median_likert_score = np.median(
                    np.array([i["toxic_score"] for i in ratings])
                )
                is_toxic = 1 if median_likert_score >= 3 else 0
                cache_dict.append((comment, is_toxic))

                # if resp := llm_cache.check(prompt=comment):
                #     print(resp)
                stored_tasks.append(
                    llm_cache.acheck(prompt=comment, distance_threshold=thold)
                )

                res[sim]["total_queries"] += 1
                res_per_bin[sim][key]["total_queries"] += 1

                if (len(stored_tasks) % batch_size == 0) or count == len(
                    test_data[key]
                ):
                    responses = await asyncio.gather(
                        *stored_tasks, return_exceptions=True
                    )
                    print(
                        "Processed {} queries for key {} at threshold {}".format(
                            count, key, thold
                        )
                    )
                    stored_tasks = []
                    for index, resp in enumerate(responses):
                        if isinstance(resp, Exception):
                            print(resp)
                            continue
                        cache_response = None
                        if resp:
                            query, is_toxic = cache_dict[index]
                            cache_response = int(resp[0]["response"])
                            matched_comment = resp[0]["prompt"]
                            vector_distance = resp[0]["vector_distance"]
                            if is_toxic != cache_response:
                                res_actual_matches[sim].append(
                                    {
                                        "query": query,
                                        "matched_comment": matched_comment,
                                        "query_is_toxic": is_toxic,
                                        "matched_comment_is_toxic": cache_response,
                                        "cosine_similarity": 1 - vector_distance,
                                    }
                                )
                            res[sim]["hits"] += 1
                            res_per_bin[sim][key]["hits"] += 1
                        if (cache_response is not None) and (
                            is_toxic == cache_response
                        ):
                            res[sim]["correct"] += 1
                            res_per_bin[sim][key]["correct"] += 1
                count += 1

            if len(stored_tasks):
                responses = await asyncio.gather(*stored_tasks)
                for index, resp in enumerate(responses):
                    cache_response = None
                    if resp:
                        query, is_toxic = cache_dict[index]
                        cache_response = int(resp[0]["response"])
                        matched_comment = resp[0]["prompt"]
                        vector_distance = resp[0]["vector_distance"]
                        if is_toxic != cache_response:
                            res_actual_matches[sim].append(
                                {
                                    "query": query,
                                    "matched_comment": matched_comment,
                                    "query_is_toxic": is_toxic,
                                    "matched_comment_is_toxic": cache_response,
                                    "cosine_similarity": 1 - vector_distance,
                                }
                            )
                        res[sim]["hits"] += 1
                        res_per_bin[sim][key]["hits"] += 1
                    if (cache_response is not None) and (is_toxic == cache_response):
                        res[sim]["correct"] += 1
                        res_per_bin[sim][key]["correct"] += 1


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/run_{args.file_name}_test")
with open(file_path, "r") as test_file:
    test_data = json.load(test_file)


asyncio.run(process_test_data(test_data))
df = pd.DataFrame.from_dict(res, orient="index").reset_index()
df.rename(columns={"index": "threshold"}, inplace=True)
df.to_csv("run-{}-output.csv".format(args.file_name), index=False)

rows = []
for threshold, bins in res_per_bin.items():
    for bin, values in bins.items():
        row = {
            "bin": bin,
            "threshold": threshold,
            "hits": values["hits"],
            "correct": values["correct"],
        }
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("run-{}-res_per_bin.csv".format(args.file_name), index=False)

actual_match_rows = []
for threshold, matches in res_actual_matches.items():
    for match in matches:
        row = {
            "threshold": threshold,
            "query": match["query"],
            "matched_comment": match["matched_comment"],
            "query_is_toxic": match["query_is_toxic"],
            "matched_comment_is_toxic": match["matched_comment_is_toxic"],
            "cosine_similarity": match["cosine_similarity"],
        }
        actual_match_rows.append(row)

df2 = pd.DataFrame(actual_match_rows)
df2.to_csv("run-{}-incorrect_queries.csv".format(args.file_name), index=False)
