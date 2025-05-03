import argparse
import json
import os
from redisvl.extensions.llmcache import SemanticCache

# from datetime import date
# import pandas as pd

# Populates train data into the cache

parser = argparse.ArgumentParser(description="Process toxicity dataset cache results.")
parser.add_argument("-f", "--file_name", type=str, help="Path to the train jsonl file")

args = parser.parse_args()

# query_hashes = []
redis_url = "redis://localhost:6379"

llm_cache = SemanticCache(
    name="precomp-balanced",
    prefix="precomp-balanced",
    redis_url=redis_url,
)

llm_cache.clear()
llm_cache.delete()
# llm_cache.index.delete(drop=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/run_{args.file_name}_train.jsonl")
with open(file_path, "r") as train_file:
    train_data = [json.loads(line) for line in train_file]


for idx, object in enumerate(train_data):
    comment = object["comment"]
    vector = object["vector"]
    is_toxic = object["is_toxic"]
    query_hash = llm_cache.store(
        prompt=comment,
        vector=vector,
        response=is_toxic,
    )
    # query_hashes.append({"query": comment, "hash": query_hash})
    if (idx + 1) % 100 == 0 or (idx + 1) == len(train_data):
        print("\t - Populated {} comments in cache".format(idx + 1))

# df = pd.DataFrame(query_hashes)
# folder = f"./results/{str(date.today())}/run-{args.file_name}/"

# os.makedirs(folder, exist_ok=True)

# df.to_csv(
#     "{}query_hashes.csv".format(folder, args.file_name),
#     index=False,
# )

print("\t - Cache Populated")
