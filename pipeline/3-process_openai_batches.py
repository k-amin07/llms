from openai import OpenAI
import openai
import os
import time

# OpenAI has hard limits on batch sizes which are currently only dependent on the model being used, not the API usage tier.
# This script uploads each bactch and waits for it to complete before moving to the next one.
# Currently, the processed results need to be manually downloaded

openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

current_dir = os.path.dirname(os.path.abspath(__file__))
initial_sleep = 30 * 60
retry_sleep = 10 * 60

for i in range(8, 12):
    batch_input_file_path = os.path.join(
        current_dir, "batches/batch_input_{}.jsonl".format(i)
    )

    # Upload batch input file
    uploaded_file = client.files.create(
        file=open(batch_input_file_path, "rb"), purpose="batch"
    )

    batch_input_file_id = uploaded_file.id

    # Submit batch job
    batch_job = openai.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    batch_id = batch_job.id

    print(f"Batch submitted! Batch ID: {batch_id}")
    print(f"Check status: https://platform.openai.com/batches/{batch_id}")

    print("Sleeping for 30 minutes before first status check...")
    time.sleep(initial_sleep)
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status
        print(f"Current status: {status}")
        if status in ["completed", "failed", "expired", "canceled"]:
            print(f"Batch {batch_id} finished with status: {status}")
            break
        else:
            print(f"Batch {batch_id} not finished yet. Sleeping for 10 more minutes...")
            time.sleep(retry_sleep)
