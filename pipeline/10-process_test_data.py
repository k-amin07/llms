import argparse
import json
import os
from redisvl.extensions.llmcache import SemanticCache


parser = argparse.ArgumentParser(description="Process toxicity dataset cache results.")
parser.add_argument("-f", "--file_name", type=str, help="Path to the test jsonl file")
args = parser.parse_args()

redis_url = "redis://localhost:6379"

llm_cache = SemanticCache(
    name="precomp",
    prefix="precomp",
    redis_url=redis_url,
)

distance_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/run_{args.file_name}_test.jsonl")

with open(file_path, "r") as test_file:
    test_data = [json.loads(line) for line in test_file]

res_actual_matches = {}

for thold in distance_thresholds:
    sim = 1 - thold
    res_actual_matches[sim] = []

count = 0
for object in test_data:
    count += 1
    comment = object["comment"]
    vector = object["vector"]
    is_toxic = object["is_toxic"]

    for thold in distance_thresholds:
        llm_cache.set_threshold(thold)
        if resp := llm_cache.check(vector=vector):
            cache_label = int(resp[0]["response"])
            matched_comment = resp[0]["prompt"]
            vector_distance = resp[0]["vector_distance"]
            cosine_similarity = 1 - vector_distance
            sim = 1 - thold
            object[sim] = {}
            object[sim]["matched_comment"] = matched_comment
            object[sim]["cosine_similarity"] = cosine_similarity
            object[sim]["cache_label"] = cache_label
    if count % 100 == 0 or count == len(test_data):
        print("Processed {} queries".format(count))


with open(file_path, "w") as out_file:
    for row in test_data:
        out_file.write(json.dumps(row) + "\n")
