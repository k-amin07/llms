from langchain_openai import ChatOpenAI

import asyncio
import json
import re
import os


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


async def process_batch(batch_objs, llm, llm_key):
    SYSTEM_PROMPT = "You are an online content moderator helping to identify toxic content. A toxic comment is a rude, disrespectful, or unreasonable comment likely to make someone leave a discussion."
    batch_inputs = []
    for obj in batch_objs:
        user_prompt = f"""
            {SYSTEM_PROMPT}

            Now, consider the following comment posted online:
            "{obj['comment']}"

            Rules:
            - Return a JSON object ONLY, with two fields:
            - "rating": an integer from 1 (not toxic) to 10 (very toxic).
            - "explanation": a reason no longer than five words.
            - DO NOT explain, apologize, or add extra text. Only return the JSON object.

            Respond with valid JSON only.
            """
        batch_inputs.append(
            [
                {"role": "user", "content": user_prompt},
            ]
        )
    responses = await llm.abatch(batch_inputs)
    results = []
    for obj, resp in zip(batch_objs, responses):
        try:
            content = resp.content
            match = re.search(r"\{.*?\}", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(content)
            llm_label = 1 if parsed["rating"] >= 7 else 0
            results.append(
                {
                    "idx": obj["idx"],
                    "llm_key": llm_key,
                    "result": {
                        "is_toxic": llm_label,
                        "rating": parsed["rating"],
                        "explanation": parsed["explanation"],
                    },
                }
            )
        except Exception as e:
            print(f"Failed to parse response for comment idx={obj['idx']}: {e}")
            results.append({"idx": obj["idx"], "llm_key": llm_key, "result": None})
    return results


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "toxicity_ratings_embeddings.json")
    output_file = os.path.join(current_dir, "toxicity_ratings_embeddings_llms.json")

    if os.path.exists(output_file):
        input_file = output_file
    with open(input_file, "r") as tox_file:
        data = json.load(tox_file)

    llms = {
        "llama": ChatOpenAI(
            model="meta-llama/llama-3.1-8b-instruct",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model_kwargs={"response_format": {"type": "json_object"}},
            temperature=0,
        ),
        "gemma": ChatOpenAI(
            model="google/gemma-3-12b-it",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model_kwargs={"response_format": {"type": "json_object"}},
            temperature=0,
        ),
        "qwen": ChatOpenAI(
            model="qwen/qwen-2.5-7b-instruct",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model_kwargs={"response_format": {"type": "json_object"}},
            temperature=0,
        ),
    }

    BATCH_SIZE = 100

    for llm_key, llm in llms.items():
        print(f"=== Processing model: {llm_key} ===")
        comments_to_process = [
            {"idx": idx, "comment": obj["comment"]}
            for idx, obj in enumerate(data)
            if llm_key not in obj
        ]

        print(len(comments_to_process))

        for i in range(0, len(comments_to_process), BATCH_SIZE):
            batch_objs = comments_to_process[i : i + BATCH_SIZE]
            batch_results = await process_batch(batch_objs, llm, llm_key)
            for res in batch_results:
                if res["result"] is not None:
                    data[res["idx"]][res["llm_key"]] = res["result"]
            print(f"Processed {i + len(batch_objs)} comments for {llm_key}")
            with open(output_file, "w") as out_file:
                json.dump(data, out_file, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
