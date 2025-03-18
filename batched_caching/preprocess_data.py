import json
import os

data = []
current_dir = os.path.dirname(os.path.abspath(__file__))
## remove duplicate comments from toxicity ratings
with open(os.path.join(current_dir, "toxicity_ratings.json"), "r") as tox_file:
    lines = tox_file.readlines()
    for line in lines:
        line = line.strip()
        data_json = json.loads(line)
        data_json["perspective_score"] = float(data_json["perspective_score"])
        data.append(data_json)

seen_comments = []
deduped_data = []

for obj in data:
    comment = obj["comment"]
    if comment not in seen_comments:
        deduped_data.append(obj)
        seen_comments.append(comment)

with open(
    os.path.join(current_dir, "toxicity_ratings_deduped.json"), "w+"
) as output_file:
    for obj in deduped_data:
        output_file.write(json.dumps(obj) + "\n")
