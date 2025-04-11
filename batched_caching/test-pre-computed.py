import argparse
import json
import os
from redisvl.extensions.llmcache import SemanticCache
import pandas as pd
from langchain_openai import ChatOpenAI
from datetime import date
import json

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

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    model_kwargs={"response_format": {"type": "json_object"}},
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

distance_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
res = {}
res_per_bin = {}
res_actual_matches = {}
res_llm_matches = {}
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
    res_llm_matches[sim] = []

for key in grouped_data_keys:
    for thold in distance_thresholds:
        sim = 1 - thold
        res_per_bin[sim][key] = {"hits": 0, "total_queries": 0, "correct": 0}


def get_prompt(query, cache_resp):
    return """Here is a pair comments taken from social media. Comment 1: "{}", Comment 2: "{}". Are these comments semantically similar? also state the reason. Return the output as a json with keys "is_similar", and "reason\"""".format(
        query, cache_resp
    )


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
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
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
                    if is_toxic == cache_response:
                        metrics[thold][key]["TP"] += 1
                        total_TP += 1
                    else:
                        metrics[thold][key]["FP"] += 1
                        total_FP += 1

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
                    if resp := llm_cache.check(vector=vector, distance_threshold=0.3):
                        matched_comment = resp[0]["prompt"]
                        vector_distance = resp[0]["vector_distance"]
                        prompt = get_prompt(comment, matched_comment)
                        llm_resp = llm.invoke(prompt).content
                        llm_resp = json.loads(llm_resp)
                        is_similar = llm_resp["is_similar"]
                        if is_similar == True:
                            cache_response = int(resp[0]["response"])
                            if cache_response == is_toxic:
                                metrics[thold][key]["FN"] += 1
                                total_FN += 1
                                res_llm_matches[sim].append(
                                    {
                                        "query": comment,
                                        "matched_comment": matched_comment,
                                        "is_toxic": is_toxic,
                                        "cosine_similarity": 1 - vector_distance,
                                    }
                                )
                            else:
                                metrics[thold][key]["TN"] += 1
                                total_TN += 1
                        else:
                            metrics[thold][key]["TN"] += 1
                            total_TN += 1
                    else:
                        metrics[thold][key]["TN"] += 1
                        total_TN += 1
                count += 1
                if count % 100 == 0 or count == len(test_data[key]):
                    print(
                        "Processed {} queries for key {} at threshold {}".format(
                            count, key, thold
                        )
                    )

    results = {}

    total_precision = (
        total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    )
    total_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    total_f1 = (
        (2 * total_precision * total_recall) / (total_precision + total_recall)
        if (total_precision + total_recall) > 0
        else 0
    )
    total_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)

    aggregate_results = {
        "Total TP": total_TP,
        "Total FP": total_FP,
        "Total TN": total_TN,
        "Total FN": total_FN,
        "Precision": total_precision,
        "Recall": total_recall,
        "F1": total_f1,
        "Accuracy": total_accuracy,
    }
    print(aggregate_results)
    for thold, key_dict in metrics.items():
        results[thold] = {}

        for key, counts in key_dict.items():
            TP = counts["TP"]
            FP = counts["FP"]
            FN = counts["FN"]
            TN = counts["TN"]

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            accuracy = (TP + TN) / (TP + TN + FP + FN)

            results[thold][key] = {
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score,
                "Accuracy": accuracy,
            }

    # Print results
    for thold, key_dict in results.items():
        print(f"Threshold: {thold}")
        for key, scores in key_dict.items():
            print(
                f"  Key: {key}, Precision: {scores['Precision']:.4f}, Recall: {scores['Recall']:.4f}, F1 Score: {scores['F1 Score']:.4f}"
            )
    # Prepare aggregate results for CSV
    aggregate_df = pd.DataFrame([aggregate_results])

    # Prepare detailed results for CSV
    rows = []
    for thold, key_dict in results.items():
        for key, scores in key_dict.items():
            row = {
                "Threshold": thold,
                "Key": key,
                **scores,
            }
            rows.append(row)

    results_df = pd.DataFrame(rows)

    # Save to CSV
    folder = f"./results/{str(date.today())}/"
    os.makedirs(folder, exist_ok=True)

    aggregate_df.to_csv(
        f"{folder}/run-{args.file_name}-aggregate-metrics.csv", index=False
    )

    results_df.to_csv(
        f"{folder}/run-{args.file_name}-detailed-metrics.csv", index=False
    )

    with open(f"{folder}/run-{args.file_name}-metrics.json", "w") as f:
        json.dump(results, f, indent=4)


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/run_{args.file_name}_test.json")
with open(file_path, "r") as test_file:
    test_data = json.load(test_file)

folder = f"./results/{str(date.today())}/"
os.makedirs(folder, exist_ok=True)

process_test_data_sync(test_data)
# asyncio.run(process_test_data(test_data))
df = pd.DataFrame.from_dict(res, orient="index").reset_index()
df.rename(columns={"index": "threshold"}, inplace=True)
df.to_csv("{}run-{}-output.csv".format(folder, args.file_name), index=False)

rows = []
for threshold, bins in res_per_bin.items():
    for bin, values in bins.items():
        row = {
            "bin": bin,
            "threshold": threshold,
            "hits": values["hits"],
            "correct": values["correct"],
            "percentage_correct": (
                0 if values["hits"] == 0 else values["correct"] / values["hits"]
            ),
        }
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("{}run-{}-res_per_bin.csv".format(folder, args.file_name), index=False)

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
df2.to_csv("{}run-{}-incorrect_queries.csv".format(folder, args.file_name), index=False)
llm_cache.index.delete(drop=True)
