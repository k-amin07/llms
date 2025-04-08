import argparse
import json
import os
from redisvl.extensions.llmcache import SemanticCache
from datetime import date
import pandas as pd
import random

import asyncio

parser = argparse.ArgumentParser(description="Process toxicity dataset cache results.")
parser.add_argument("-f", "--file_name", type=str, help="Path to the input JSON file")
parser.add_argument(
    "-p",
    "--percentage",
    type=int,
    help="(Integer) Percentage of train data to store in cache",
    default=100,
)
args = parser.parse_args()


redis_url = "redis://localhost:6379"

llm_cache = SemanticCache(
    name="precomp",
    prefix="precomp",
    redis_url=redis_url,
)

llm_cache.index.delete(drop=True)

batch_size = 100

query_hashes = []


async def process_and_store(train_data):
    stored_tasks = []
    i = 1
    count = 1
    index = 0
    total_train_size = len(train_data)
    chosen_train_size = int((total_train_size * args.percentage) // 100)
    train_data = random.sample(train_data, k=chosen_train_size)
    print("Total train data: ", total_train_size)
    print("Populated train data", len(train_data))
    for object in train_data:
        comment = object["comment"]
        vector = object["vector"]
        is_toxic = object["is_toxic"]
        stored_tasks.append(
            llm_cache.astore(
                prompt=comment,
                vector=vector,
                response=is_toxic,
            )
        )
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
    folder = "./results/{}/".format(str(date.today()))

    if not os.path.exists(folder):
        os.mkdir(folder)
    df.to_csv(
        "{}/run-{}-query_hashes.csv".format(folder, args.file_name),
        index=False,
    )


def process_and_store_sync(train_data):
    i = 0
    total_train_size = len(train_data)
    chosen_train_size = int((total_train_size * args.percentage) // 100)
    train_data = random.sample(train_data, k=chosen_train_size)
    print("Total train data: ", total_train_size)
    print("Populated train data", len(train_data))
    for object in train_data:
        comment = object["comment"]
        vector = object["vector"]
        is_toxic = object["is_toxic"]
        query_hash = llm_cache.store(
            prompt=comment,
            vector=vector,
            response=is_toxic,
        )
        query_hashes.append({"query": comment, "hash": query_hash})
        if i % 100 == 0:
            print("Processed {} queries".format(i))
        i += 1
    df = pd.DataFrame(query_hashes)
    folder = "./results/{}/".format(str(date.today()))

    os.makedirs(folder, exist_ok=True)
    df.to_csv(
        "{}/run-{}-query_hashes.csv".format(folder, args.file_name),
        index=False,
    )


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f"data/run_{args.file_name}_train.json")
with open(file_path, "r") as train_file:
    train_data = json.load(train_file)

if __name__ == "__main__":
    # asyncio.run(process_and_store(train_data))
    process_and_store_sync(train_data)
