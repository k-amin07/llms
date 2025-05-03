# Computes inter-rater agreement for each llm and human rating

import json
import os
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import krippendorff
from itertools import combinations


def load_jsonl_file(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


def compute_fleiss_kappa_with_human(data):
    data_matrix = []
    for item in data:
        labels = [
            item["is_toxic"],
            item["llama"]["is_toxic"],
            item["qwen"]["is_toxic"],
            item["4o-mini"]["is_toxic"],
            item["gemma"]["is_toxic"],
        ]
        counts = [labels.count(0), labels.count(1)]
        data_matrix.append(counts)

    # Convert to numpy array
    data_matrix = np.array(data_matrix)

    # Compute Fleiss' Kappa
    kappa = fleiss_kappa(data_matrix)
    print(f"Fleiss' Kappa (5 raters): {kappa:.3f}")


def compute_fleiss_kappa_llm_only(data):
    data_matrix = []
    for item in data:
        labels = [
            item["llama"]["is_toxic"],
            item["qwen"]["is_toxic"],
            item["4o-mini"]["is_toxic"],
            item["gemma"]["is_toxic"],
        ]
        counts = [labels.count(0), labels.count(1)]
        data_matrix.append(counts)
    data_matrix = np.array(data_matrix)

    kappa = fleiss_kappa(data_matrix)
    print(f"Fleiss' Kappa (4 llms): {kappa:.3f}")


def compute_cohen_kappa(data):
    model_names = ["llama", "4o-mini", "gemma"]
    ground_truth = [item["is_toxic"] for item in data]
    for model in model_names:
        model_preds = [item[model]["is_toxic"] for item in data]
        kappa = cohen_kappa_score(ground_truth, model_preds)
        print(f"Cohen's Kappa ({model} vs human): {kappa:.3f}")

    # Compute pairwise between models
    print("\nCohen's Kappa: Model vs. Model")
    for model1, model2 in combinations(model_names, 2):
        preds1 = [item[model1]["is_toxic"] for item in data]
        preds2 = [item[model2]["is_toxic"] for item in data]
        kappa = cohen_kappa_score(preds1, preds2)
        print(f"{model1:8} vs {model2:8} : {kappa:.3f}")


def compute_krippendorff_alpha_with_human(data):
    raters = ["is_toxic", "llama", "4o-mini", "gemma"]
    annotations = []

    for rater in raters:
        row = []
        for item in data:
            if rater == "is_toxic":
                row.append(item[rater])
            else:
                row.append(item[rater]["is_toxic"])
        annotations.append(row)

    # Compute Krippendorff's alpha (for nominal data)
    alpha = krippendorff.alpha(
        reliability_data=annotations, level_of_measurement="nominal"
    )

    print(f"Krippendorff's Alpha (5 raters): {alpha:.3f}")


def compute_krippendorff_alpha_llm_only(data):
    raters = ["llama", "4o-mini", "gemma"]
    annotations = []

    for rater in raters:
        row = []
        for item in data:
            row.append(item[rater]["is_toxic"])
        annotations.append(row)

    # Compute Krippendorff's alpha (for nominal data)
    alpha = krippendorff.alpha(
        reliability_data=annotations, level_of_measurement="nominal"
    )

    print(f"Krippendorff's Alpha: {alpha:.3f}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    llm_ratings_file = os.path.join(
        current_dir, "toxicity_ratings_embeddings_llms.jsonl"
    )

    with open(llm_ratings_file, "r") as tox_file:
        data = [json.loads(line) for line in tox_file]

    compute_cohen_kappa(data)
    compute_fleiss_kappa_llm_only(data)
    compute_fleiss_kappa_with_human(data)
    compute_krippendorff_alpha_llm_only(data)
    compute_krippendorff_alpha_with_human(data)


if __name__ == "__main__":
    main()
