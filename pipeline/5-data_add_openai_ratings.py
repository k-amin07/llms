# Combines jsonl files from openai batch results into data

import json
import os
from glob import glob


def load_jsonl_file(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    batches_folder = os.path.join(current_dir, "./batches")
    input_files = sorted(glob(os.path.join(batches_folder, "batch_input_*.jsonl")))
    output_files = sorted(glob(os.path.join(batches_folder, "batch_output_*.jsonl")))

    # Load all batch inputs and outputs
    all_inputs = []
    for file in input_files:
        all_inputs.extend(load_jsonl_file(file))

    all_outputs = []
    for file in output_files:
        all_outputs.extend(load_jsonl_file(file))

    # Load main data
    toxicity_ratings_file = os.path.join(
        current_dir, "toxicity_ratings_embeddings_llama.json"
    )
    with open(toxicity_ratings_file, "r") as f:
        data = json.load(f)

    # Build a lookup from comment_x -> its idx in data
    comment_to_idx = {}
    for idx, item in enumerate(data):
        comment_id = f"comment_{idx}"
        comment_to_idx[comment_id] = idx

    # Process outputs
    for out in all_outputs:
        custom_id = out.get("custom_id")
        if not custom_id:
            continue  # skip malformed entries

        response_body = out.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            continue  # no choices

        content_str = choices[0].get("message", {}).get("content", "{}")

        try:
            content_json = json.loads(content_str)
            rating = content_json.get("rating")
            explanation = content_json.get("explanation")
        except Exception as e:
            print(f"Error parsing content for {custom_id}: {e}")
            continue

        if rating is None or explanation is None:
            print(f"Missing rating/explanation for {custom_id}")
            continue

        # Find where to insert
        idx = comment_to_idx.get(custom_id)
        if idx is None:
            print(f"Warning: No matching data entry for {custom_id}")
            continue

        is_toxic = 1 if rating >= 7 else 0

        data[idx]["4o-mini"] = {
            "is_toxic": is_toxic,
            "rating": rating,
            "explanation": explanation,
        }

    # Save back
    toxicity_output_file = os.path.join(
        current_dir, "toxicity_ratings_embeddings_openai.json"
    )
    with open("toxicity_ratings_embeddings_openai.json", "w") as f:
        json.dump(data, f, indent=2)

    print("Successfully updated toxicity_ratings_embeddings_openai.json!")


if __name__ == "__main__":
    main()
