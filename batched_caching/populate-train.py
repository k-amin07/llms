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
    model="text-embedding-ada-002",
    api_config={"api_key": api_key},
)
redis_url = "redis://localhost:6379"

llm_cache = SemanticCache(
    name="zakir-cache",
    prefix="zakir-cache",
    redis_url=redis_url,
    distance_threshold=0.1,
    vectorizer=client,
)

llm_cache.clear()

batch_size = 100

query_hashes = []


async def process_and_store(train_data):
    stored_tasks = []
    i = 1
    count = 1
    index = 0
    total_train_size = len(train_data)
    for object in train_data:
        comment = object["comment"]
        ratings = object["ratings"]
        median_likert_score = np.median(np.array([i["toxic_score"] for i in ratings]))
        is_toxic = 1 if median_likert_score >= 3 else 0
        stored_tasks.append(llm_cache.astore(prompt=comment, response=is_toxic))
        query_hashes.append({"query": comment, "hash": ""})
        if (len(stored_tasks) % batch_size == 0) or count == total_train_size:
            hashes = await asyncio.gather(*stored_tasks)
            for hash in hashes:
                query_hashes[index]["hash"] = hash
                index += 1

            print("Processed {} queries".format(min(i * batch_size, total_train_size)))
            i += 1
            stored_tasks = []
        count += 1
    await asyncio.gather(*stored_tasks)
    df = pd.DataFrame(query_hashes)
    df.to_csv("run-{}-query_hashes.csv".format(args.file_name), index=False)


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/run_{args.file_name}_train")
with open(file_path, "r") as train_file:
    train_data = json.load(train_file)


if __name__ == "__main__":
    asyncio.run(process_and_store(train_data))
