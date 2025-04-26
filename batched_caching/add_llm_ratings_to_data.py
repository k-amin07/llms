from langchain_openai import ChatOpenAI
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Adds llm ratings to test data")
parser.add_argument(
    "-f", "--file_name", type=str, help="JSON file name containing test data"
)
args = parser.parse_args()


def ask_llm(comment, llm):
    SYSTEM_PROMPT = "You are an online content moderator that helps identify toxic content. A toxic comment is defined as a rude, disrespectful, or unreasonable comment that is likely to make someone leave a discussion"
    user_prompt = """
    Consider the following comment:
    "{}"
    Return a JSON object with a two fields, "rating" that is a score from 1 to 10 depending on how toxic the comment is based on our definition, and "explanation" which contains a reason for the rating. The reason must not be longer than five words.
    """.format(
        comment
    )
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    return json.loads(response.content)


llms = {
    # "llama": ChatOpenAI(
    #     model="llama3.2:latest",
    #     api_key="ollama",
    #     base_url="http://localhost:11434/v1",
    #     model_kwargs={"response_format": {"type": "json_object"}},
    #     temperature=0,
    # ),
    # "gemma": ChatOpenAI(
    #     model="gemma3:4b-it-qat",
    #     api_key="ollama",
    #     base_url="http://localhost:11434/v1",
    #     model_kwargs={"response_format": {"type": "json_object"}},
    #     temperature=0,
    # ),
    # "qwen": ChatOpenAI(
    #     model="qwen2.5:3b",
    #     api_key="ollama",
    #     base_url="http://localhost:11434/v1",
    #     model_kwargs={"response_format": {"type": "json_object"}},
    #     temperature=0,
    # ),
    "4o": ChatOpenAI(
        model="gpt-4o-mini",
        model_kwargs={"response_format": {"type": "json_object"}},
        temperature=0,
    ),
}

llm_keys = llms.keys()

current_dir = os.path.dirname(os.path.abspath(__file__))
io_file = os.path.join(current_dir, "data/run_{}_test.json".format(args.file_name))
print(io_file)
with open(io_file, "r") as tox_file:
    data = json.load(tox_file)


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

total_data = sum([len(data[i]) for i in grouped_data_keys])


def get_llm_response(comment, llm):
    resp = ask_llm(comment, llm)
    llm_label = 1 if resp["rating"] >= 7 else 0
    return {
        "is_toxic": llm_label,
        "rating": resp["rating"],
        "explanation": resp["explanation"],
    }


for llm_key in llm_keys:
    print(f"=== Processing model: {llm_key} ===")
    llm = llms[llm_key]
    all_jobs = []
    key_index_map = []
    count = 0
    for key in grouped_data_keys:
        for idx, obj in enumerate(data[key]):
            if "llm_ratings" not in obj:
                obj["llm_ratings"] = {}

            if "4o" in obj["llm_ratings"]:
                print("Already processed")
                count += 1
                continue
            if count == 15000:
                break
            comment = obj["comment"]
            print("Processing comment # {}: {}".format(count, comment))
            resp = get_llm_response(comment=comment, llm=llm)
            data[key][idx]["llm_ratings"][llm_key] = resp
            count += 1

            if count % 100 == 0:
                print("Processed {} queries".format(count))
                with open(io_file, "w") as out_file:
                    json.dump(data, out_file, indent=2)

            # all_jobs.append(comment)
            # key_index_map.append((key, idx))
        if count == 15000:
            break
    # results = []

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     results = list(
    #         tqdm(
    #             executor.map(lambda c: get_llm_response(c, llm), all_jobs),
    #             total=len(all_jobs),
    #         )
    #     )
    #     for i, ((key, idx), result) in enumerate(zip(key_index_map, results)):
    #         data[key][idx]["llm_ratings"][llm_key] = result
    #         if (i + 1) % 100 == 0:
    #             with open(io_file, "w") as out_file:
    #                 json.dump(data, out_file, indent=2)
    #             print(f"[{llm_key}] Progress saved after {i + 1} queries.")
    # with open(io_file, "w") as out_file:
    #     json.dump(data, out_file, indent=2)

# i = 0
# for key in grouped_data_keys:
#     for obj in data[key]:
#         i += 1
#         comment = obj["comment"]
#         print("Processing {}/{} (key {})".format(i, total_data, key))
#         obj["llm_ratings"] = {}
#         with ThreadPoolExecutor(max_workers=len(llms)) as executor:
#             futures = [executor.submit(get_llm_response, k, comment) for k in llms]
#             results = [f.result() for f in futures]
#             obj["llm_ratings"] = dict(results)

#         for llm_key in llm_keys:
#             llm = llms[llm_key]
#             start = time.time()
#             resp = ask_llm(comment, llm)
#             print("\tTook {}s with llm {}".format(time.time() - start, llm_key))
#             llm_label = 1 if resp["rating"] >= 7 else 0
#             llm_rating = resp["rating"]
#             llm_explanation = resp["explanation"]
#             obj["llm_ratings"][llm_key] = {
#                 "is_toxic": llm_label,
#                 "rating": llm_rating,
#                 "explanation": llm_explanation,
#             }

with open(io_file, "w") as out_file:
    json.dump(data, out_file, indent=2)

print(f"Saved {len(data)} records to {io_file}")
