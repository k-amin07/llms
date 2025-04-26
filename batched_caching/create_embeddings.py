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

for obj, embedding in zip(data, embeddings):
    obj["vector"] = [float(x) for x in embedding]
    ratings = obj["ratings"]
    median_likert_score = np.median([r["toxic_score"] for r in ratings])
    is_toxic = 1 if median_likert_score >= 3 else 0
    obj["is_toxic"] = is_toxic

output_path = os.path.join(current_dir, "toxicity_ratings_embeddings.json")

with open(output_path, "w") as out_file:
    json.dump(data, out_file, indent=2)

print(f"Saved {len(data)} records to {output_path}")
