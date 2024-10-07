from redisvl.extensions.llmcache import SemanticCache
from langchain_openai import ChatOpenAI
from datetime import datetime

redis_url = "redis://localhost:6379"

llmcache = SemanticCache(
    name="llmcache",                     # underlying search index name
    prefix="llmcache",                   # redis key prefix for hash entries
    redis_url=redis_url,  # redis connection url string
    distance_threshold=0.1               # semantic cache distance threshold
)

llm = ChatOpenAI(model_name="gpt-4o-mini", n=2)

prompts = ["Tell me a joke", "Tell me a joke" , "Tell me joke", "Can you tell me a joke"]

for prompt in prompts:
    startTime = datetime.now()
    if response := llmcache.check(prompt=prompt):
        print("Cache hit, took {}".format(datetime.now() - startTime))
        print(response[0]["response"])
    else:
        response = llm.invoke(prompt)

        llmcache.store(
            prompt=prompt,
            response=response.content,
        )
        print("Sent to llm, took {}".format(datetime.now() - startTime))
        response.pretty_print()
