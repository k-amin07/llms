import os
import json

input_comments_file = "./toxicity_ratings_embeddings.json"
batch_input_dir = "./batches"  # New: output folder for batch files

system_prompt = (
    "You are an online content moderator that helps identify toxic content. "
    "A toxic comment is defined as a rude, disrespectful, or unreasonable comment "
    "that is likely to make someone leave a discussion."
)


def format_message(comment):
    return f"""
    Consider the following comment:
    "{comment}"
    Return a JSON object with two fields, "rating" from 1 to 10 depending on toxicity, 
    and "explanation" (max five words).
    """


# Make sure batch directory exists
os.makedirs(batch_input_dir, exist_ok=True)

# Load all comments
with open(input_comments_file, "r") as f:
    comments_data = json.load(f)

total_comments = len(comments_data)
print(f"Total comments loaded: {total_comments}")

# Split into batches of ≤10,000
batch_size = 10000
num_batches = 11

# Iterate over batches
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, total_comments)
    batch_comments = comments_data[start_idx:end_idx]

    batch_input_file_path = os.path.join(
        batch_input_dir, f"batch_input_{batch_idx+1}.jsonl"
    )

    # Write batch file
    with open(batch_input_file_path, "w") as f:
        for idx, obj in enumerate(batch_comments):
            comment = obj["comment"]
            batch_obj = {
                "custom_id": f"comment_{start_idx + idx}",  # global index
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": format_message(comment)},
                    ],
                },
            }
            f.write(json.dumps(batch_obj) + "\n")

    print(
        f"Batch input file '{batch_input_file_path}' created with {len(batch_comments)} requests."
    )
