import time

from gptcache import cache, Config
from gptcache.manager import manager_factory
from gptcache.embedding import OpenAI
from gptcache.processor.post import temperature_softmax
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter import openai
from datetime import datetime

cache.set_openai_key()

emb = OpenAI()

data_manager = manager_factory("sqlite,faiss", vector_params={"dimension": emb.dimension})

cache.init(
    embedding_func=emb.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    post_process_messages_func=temperature_softmax
    )

prompts = ["Tell me a joke", "Tell me a joke" , "Tell me joke", "Can you tell me a joke"]
for prompt in prompts:
    startTime = datetime.now()
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature = 0.7, # making it 0.7 to match the manual cache example default.
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    print("LLM took {}".format(datetime.now() - startTime))
    print("Answer:", response["choices"][0]["message"]["content"])