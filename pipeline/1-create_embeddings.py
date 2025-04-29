from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np

data = []
current_dir = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(current_dir, "toxicity_ratings.json"), "r") as tox_file:
    for line in tox_file:
        line = line.strip()
        data_json = json.loads(line)
        data_json["perspective_score"] = float(data_json["perspective_score"])
        data.append(data_json)


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

comments = []
for obj in data:
    comment = obj["comment"]
    comments.append(comment)


embeddings = model.encode(comments, show_progress_bar=True)

print("Data Length, Embeddings Length = ", len(data), len(embeddings))
print(len(data) == len(embeddings))

if len(data) != len(embeddings):
    raise ValueError("Mismatch between deduped data and embeddings length")

output_data = []
for obj, embedding in zip(data, embeddings):
    comment = obj["comment"]
    obj["vector"] = [float(x) for x in embedding]
    ratings = obj["ratings"]
    median_likert_score = np.median([r["toxic_score"] for r in ratings])
    is_toxic = 1 if median_likert_score >= 3 else 0
    out = {
        "comment": comment,
        "vector": obj["vector"],
        "human_rating": int(
            median_likert_score
        ),  # Each comment has 5 ratings so median likert score will always be an integer
        "is_toxic": is_toxic,
    }
    output_data.append(out)

output_data.sort(key=lambda x: x["is_toxic"])

output_path = os.path.join(current_dir, "toxicity_ratings_embeddings.json")

with open(output_path, "w") as out_file:
    json.dump(output_data, out_file, indent=2)

print(f"Saved {len(output_data)} records to {output_path}")
