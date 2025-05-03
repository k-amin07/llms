import os
import json

## Gemma ratings were computed on a separate machine. Combine the files into a single data file

current_dir = os.path.dirname(os.path.abspath(__file__))

gemma_path = os.path.join(current_dir, "toxicity_ratings_embeddings_gemma.json")
openai_path = os.path.join(current_dir, "toxicity_ratings_embeddings_openai.json")

with open(gemma_path, "r") as gfile:
    gemma_data = json.load(gfile)

with open(openai_path, "r") as ofile:
    openai_data = json.load(ofile)

if len(gemma_data) != len(openai_data):
    raise "Unequal data lengths"

for open_ai_data_obj, gemma_data_obj in zip(openai_data, gemma_data):
    open_ai_data_obj["gemma"] = {
        "is_toxic": gemma_data_obj["gemma"]["is_toxic"],
        "rating": gemma_data_obj["gemma"]["rating"],
        "explanation": gemma_data_obj["gemma"]["explanation"],
    }
    open_ai_data_obj["human_rating"] = int(open_ai_data_obj["human_rating"])

output_json_path = os.path.join(current_dir, "toxicity_ratings_embeddings_llms.json")
output_jsonl_path = os.path.join(current_dir, "toxicity_ratings_embeddings_llms.jsonl")

with open(output_json_path, "w+") as out_file:
    json.dump(openai_data, out_file, indent=2)

with open(output_jsonl_path, "w+") as out_jsonl_file:
    for row in openai_data:
        out_jsonl_file.write(json.dumps(row) + "\n")
