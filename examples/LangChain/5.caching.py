from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_community.cache import InMemoryCache



llm = ChatOpenAI(model_name="gpt-4o-mini", n=2)
# set_llm_cache(InMemoryCache())

# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

from datetime import datetime
startTime = datetime.now()
llm.predict("Tell me a joke")

print(datetime.now() - startTime)

startTime = datetime.now()
llm.predict("Tell me a joke")
print(datetime.now() - startTime)

# Output:
## 0:00:02.045725
## 0:00:00.000537