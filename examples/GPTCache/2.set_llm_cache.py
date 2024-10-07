from langchain_community.cache import RedisSemanticCache
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from datetime import datetime

from langchain_community.embeddings import OllamaEmbeddings
# from gptcache.embedding import OpenAI

redis_url = "redis://localhost:6379"

llm = ChatOpenAI(model_name="gpt-4o-mini", n=2)
set_llm_cache(RedisSemanticCache(
    embedding=OllamaEmbeddings(),
    # embedding=OpenAI(),
    redis_url=redis_url
))

prompts = ["Tell me a joke", "Tell me a joke", "Tell me joke", "Can you tell me a joke"]

for prompt in prompts:
    startTime = datetime.now()
    response = llm.invoke(prompt)
    print("LLM took {}".format(datetime.now() - startTime))
    response.pretty_print()

