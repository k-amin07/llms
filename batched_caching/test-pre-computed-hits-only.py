import argparse
import json
import os
from redisvl.extensions.llmcache import SemanticCache
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
import json

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

distance_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
res_per_bin = {}
res_actual_matches = {}

for thold in distance_thresholds:
    sim = 1 - thold
    res_per_bin[sim] = {}
    res_actual_matches[sim] = []

for key in grouped_data_keys:
    for thold in distance_thresholds:
        sim = 1 - thold
        res_per_bin[sim][key] = {"hits": 0, "total_queries": 0, "correct": 0}


def process_test_data(test_data):
    metrics_by_threshold = {}
    roc_data_per_threshold = {}
    for thold in distance_thresholds:
        sim = 1 - thold
        llm_cache.set_threshold(thold)
        roc_data_per_threshold[thold] = {"y_true": [], "y_score": []}
        metrics_by_threshold[thold] = {}
        for key in grouped_data_keys:
            res_per_bin[sim][key]["total_queries"] += 1
            metrics_by_threshold[thold][key] = {}
            metrics_by_key = {
                "TP": 0,
                "FP": 0,
                "TN": 0,
                "FN": 0,
                "label_match_hits": 0,
                "label_mismatch_hits": 0,
                "total_hits": 0,
                "total_queries": 0,
                "metric_progression": [],  # Track how metrics evolve with more hits
            }
            hit_count = 0
            count = 0
            for object in test_data[key]:
                comment = object["comment"]
                vector = object["vector"]
                is_toxic = object["is_toxic"]
                metrics_by_key["total_queries"] += 1
                roc_y_true = []
                roc_y_score = []
                count += 1
                if count % 100 == 0 or count == len(test_data[key]):
                    print(
                        "Processed {} queries for key {} at threshold {}".format(
                            count, key, thold
                        )
                    )
                if resp := llm_cache.check(vector=vector, distance_threshold=thold):
                    res_per_bin[sim][key]["hits"] += 1
                    cache_label = int(resp[0]["response"])
                    metrics_by_key["total_hits"] += 1
                    if is_toxic == 1 and cache_label == 1:
                        metrics_by_key["TP"] += 1
                    elif is_toxic == 0 and cache_label == 1:
                        metrics_by_key["FP"] += 1
                    elif is_toxic == 1 and cache_label == 0:
                        metrics_by_key["FN"] += 1
                    elif is_toxic == 0 and cache_label == 0:
                        metrics_by_key["TN"] += 1

                    matched_comment = resp[0]["prompt"]
                    vector_distance = resp[0]["vector_distance"]
                    cosine_similarity = 1 - vector_distance
                    roc_data_per_threshold[thold]["y_true"].append(is_toxic)
                    roc_data_per_threshold[thold]["y_score"].append(cosine_similarity)
                    roc_y_true.append(is_toxic)
                    roc_y_score.append(cosine_similarity)
                    if is_toxic != cache_label:
                        res_actual_matches[sim].append(
                            {
                                "query": comment,
                                "matched_comment": matched_comment,
                                "query_is_toxic": is_toxic,
                                "matched_comment_is_toxic": cache_label,
                                "cosine_similarity": cosine_similarity,
                            }
                        )
                        metrics_by_key["label_mismatch_hits"] += 1
                    else:
                        metrics_by_key["label_match_hits"] += 1
                        res_per_bin[sim][key]["correct"] += 1

                    hit_count += 1
                    if hit_count % 100 == 0 or hit_count == len(test_data[key]):
                        tp, fp, fn, tn = (
                            metrics_by_key["TP"],
                            metrics_by_key["FP"],
                            metrics_by_key["FN"],
                            metrics_by_key["TN"],
                        )
                        total = tp + fp + fn + tn
                        precision = tp / (tp + fp) if (tp + fp) else 0
                        recall = tp / (tp + fn) if (tp + fn) else 0
                        f1 = (
                            2 * precision * recall / (precision + recall)
                            if (precision + recall)
                            else 0
                        )
                        accuracy = (tp + tn) / total if total else 0
                        coverage = (
                            metrics_by_key["total_hits"]
                            / metrics_by_key["total_queries"]
                        )
                        abstention_rate = 1 - coverage
                        match_rate = (
                            metrics_by_key["label_match_hits"]
                            / metrics_by_key["total_hits"]
                        )
                        try:
                            auc_score_progress = (
                                roc_auc_score(roc_y_true, roc_y_score)
                                if len(set(roc_y_true)) > 1
                                else 0
                            )
                        except ValueError:
                            auc_score_progress = 0
                        metrics_by_key["metric_progression"].append(
                            {
                                "hits": metrics_by_key["total_hits"],
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "coverage": coverage,
                                "abstention_rate": abstention_rate,
                                "label_match_rate": match_rate,
                                "auc": auc_score_progress,
                            }
                        )

            metrics_by_threshold[thold][key] = metrics_by_key

    aggregate_metrics = {}
    for thold in metrics_by_threshold:
        TP = sum(
            metrics_by_threshold[thold][key]["TP"]
            for key in metrics_by_threshold[thold]
        )
        FP = sum(
            metrics_by_threshold[thold][key]["FP"]
            for key in metrics_by_threshold[thold]
        )
        TN = sum(
            metrics_by_threshold[thold][key]["TN"]
            for key in metrics_by_threshold[thold]
        )
        FN = sum(
            metrics_by_threshold[thold][key]["FN"]
            for key in metrics_by_threshold[thold]
        )

        total = TP + FP + TN + FN
        hits = sum(
            metrics_by_threshold[thold][key]["total_hits"]
            for key in metrics_by_threshold[thold]
        )
        total_queries = sum(
            metrics_by_threshold[thold][key]["total_queries"]
            for key in metrics_by_threshold[thold]
        )
        coverage = hits / total_queries if total_queries else 0
        abstention_rate = 1 - coverage
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        accuracy = (TP + TN) / total if total else 0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0
        )
        label_match_rate = (TP + TN) / hits if hits else 0
        label_mismatch_rate = (FP + FN) / hits if hits else 0
        auc_score = 0
        if (
            thold in roc_data_per_threshold
            and len(set(roc_data_per_threshold[thold]["y_true"])) > 1
        ):
            y_true = roc_data_per_threshold[thold]["y_true"]
            y_score = roc_data_per_threshold[thold]["y_score"]
            auc_score = roc_auc_score(y_true, y_score)

        aggregate_metrics[thold] = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Coverage": coverage,
            "Abstention Rate": abstention_rate,
            "Label Match Rate": label_match_rate,
            "Label Mismatch Rate": label_mismatch_rate,
            "AUC": auc_score,
        }

    folder = f"./results/{str(date.today())}/run-{args.file_name}/"
    for thold, data in roc_data_per_threshold.items():
        y_true = data["y_true"]
        y_score = data["y_score"]
        if len(set(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        if recall[0] == 0.0 and precision[0] == 1.0:
            precision = precision[1:]
            recall = recall[1:]
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (Threshold {thold})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{folder}precision_recall_threshold_{thold}.png")
        plt.close()

    for thold, data in roc_data_per_threshold.items():
        y_true = data["y_true"]
        y_score = data["y_score"]
        if len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Threshold {thold})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{folder}roc_curve_threshold_{thold}.png")
        plt.close()
    # Combined
    plt.figure()
    for thold, data in roc_data_per_threshold.items():
        y_true = data["y_true"]
        y_score = data["y_score"]
        if len(set(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        if recall[0] == 0.0 and precision[0] == 1.0:
            precision = precision[1:]
            recall = recall[1:]
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"Th={thold}, AUC={pr_auc:.2f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision-Recall Curve Across Thresholds")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}combined_precision_recall.png")
    plt.close()

    plt.figure()
    for thold, data in roc_data_per_threshold.items():
        y_true = data["y_true"]
        y_score = data["y_score"]
        if len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"Th={thold}, AUC={roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curve Across Thresholds")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}combined_roc_curve.png")
    plt.close()

    macro_metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Coverage": [],
        "Abstention Rate": [],
        "Label Match Rate": [],
        "Label Mismatch Rate": [],
        "AUC": [],
    }

    for thold, metrics in aggregate_metrics.items():
        for k in macro_metrics:
            macro_metrics[k].append(metrics.get(k, 0))
    macro_avg = {k: sum(v) / len(v) for k, v in macro_metrics.items()}

    TP_micro = sum(m["TP"] for m in aggregate_metrics.values())
    FP_micro = sum(m["FP"] for m in aggregate_metrics.values())
    TN_micro = sum(m["TN"] for m in aggregate_metrics.values())
    FN_micro = sum(m["FN"] for m in aggregate_metrics.values())

    total = TP_micro + FP_micro + TN_micro + FN_micro
    precision_micro = TP_micro / (TP_micro + FP_micro) if (TP_micro + FP_micro) else 0
    recall_micro = TP_micro / (TP_micro + FN_micro) if (TP_micro + FN_micro) else 0
    accuracy_micro = (TP_micro + TN_micro) / total if total else 0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro)
        else 0
    )
    micro_avg = {
        "Accuracy": accuracy_micro,
        "Precision": precision_micro,
        "Recall": recall_micro,
        "F1": f1_micro,
    }

    df_agg = pd.DataFrame.from_dict(aggregate_metrics, orient="index")
    df_agg.index.name = "Threshold"
    df_agg = df_agg.sort_index()
    df_agg.loc["macro_avg"] = macro_avg
    df_agg.loc["micro_avg"] = micro_avg
    os.makedirs(folder, exist_ok=True)
    df_agg.to_csv(f"{folder}aggregate_metrics_per_threshold.csv")
    with open(f"{folder}metrics_by_threshold.json", "w") as f:
        json.dump(metrics_by_threshold, f, indent=2)
    with open(f"{folder}roc_data_per_threshold.json", "w") as f:
        json.dump(roc_data_per_threshold, f, indent=2)


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/run_{args.file_name}_test.json")
with open(file_path, "r") as test_file:
    test_data = json.load(test_file)

folder = f"./results/{str(date.today())}/run-{args.file_name}/"
os.makedirs(folder, exist_ok=True)

process_test_data(test_data)

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
df.to_csv("{}res_per_bin.csv".format(folder, args.file_name), index=False)

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
df2.to_csv("{}incorrect_queries.csv".format(folder, args.file_name), index=False)
