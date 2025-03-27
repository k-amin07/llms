import argparse
import json
import os
from redisvl.extensions.llmcache import SemanticCache
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import asyncio

parser = argparse.ArgumentParser(description="Process toxicity dataset cache results.")
parser.add_argument("-f", "--file_name", type=str, help="Path to the input JSON file")
args = parser.parse_args()

redis_url = "redis://localhost:6379"

llm_cache = SemanticCache(
    name="precomp",
    prefix="precomp",
    redis_url=redis_url,
)

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
batch_size = 100

metrics = {
    thold: {
        key: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
        }
        for key in grouped_data_keys
    }
    for thold in distance_thresholds
}

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
                vector = object["vector"]
                is_toxic = object["is_toxic"]
                cache_dict.append((comment, is_toxic))

                stored_tasks.append(
                    llm_cache.acheck(vector=vector, distance_threshold=thold)
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


def process_test_data_sync(test_data):
    for thold in distance_thresholds:
        sim = 1 - thold
        for key in grouped_data_keys:
            count = 1
            for object in test_data[key]:
                comment = object["comment"]
                vector = object["vector"]
                is_toxic = object["is_toxic"]
                res[sim]["total_queries"] += 1
                res_per_bin[sim][key]["total_queries"] += 1
                if resp := llm_cache.check(vector=vector, distance_threshold=thold):
                    cache_response = int(resp[0]["response"])
                    if is_toxic == 1 and cache_response == 1:
                        metrics[thold][key]["TP"] += 1
                    elif is_toxic == 0 and cache_response == 1:
                        metrics[thold][key]["FP"] += 1
                    elif is_toxic == 1 and cache_response == 0:
                        metrics[thold][key]["FN"] += 1
                    elif is_toxic == 0 and cache_response == 0:
                        metrics[thold][key]["TN"] += 1

                    matched_comment = resp[0]["prompt"]
                    vector_distance = resp[0]["vector_distance"]
                    if is_toxic != cache_response:
                        res_actual_matches[sim].append(
                            {
                                "query": comment,
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

                else:
                    if is_toxic == 1:
                        metrics[thold][key]["FN"] += 1
                    else:
                        metrics[thold][key]["TN"] += 1
                count += 1
                if count % 100 == 0 or count == len(test_data[key]):
                    print(
                        "Processed {} queries for key {} at threshold {}".format(
                            count, key, thold
                        )
                    )

    def calculate_metrics(TP, FP, FN):
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return precision, recall, f1

    for thold, key_data in metrics.items():
        for key, counts in key_data.items():
            TP, FP, FN = counts["TP"], counts["FP"], counts["FN"]
            precision, recall, f1 = calculate_metrics(TP, FP, FN)
            print(
                f"Threshold: {thold}, Key: {key} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )

    for thold in distance_thresholds:
        agg_TP = sum(metrics[thold][key]["TP"] for key in grouped_data_keys)
        agg_FP = sum(metrics[thold][key]["FP"] for key in grouped_data_keys)
        agg_FN = sum(metrics[thold][key]["FN"] for key in grouped_data_keys)
        precision, recall, f1 = calculate_metrics(agg_TP, agg_FP, agg_FN)
        print(
            f"Threshold: {thold} (Aggregate) - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/run_{args.file_name}_test.json")
with open(file_path, "r") as test_file:
    test_data = json.load(test_file)

process_test_data_sync(test_data)
# asyncio.run(process_test_data(test_data))
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
