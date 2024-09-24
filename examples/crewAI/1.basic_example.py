from crewai import Agent, Task, Crew, Process

## We need to have the OPENAI_API_KEY env variable set, and it will use ChatGPT by default. We can also pass in llm="" to specify an llm
researcher = Agent(
    role = "Researcher",
    goal = "Research new AI insights",
    backstory = "You are an AI research assistant", # this is like the context for the Agent. With crewai, we use the keyword backstory instead
    verbose = True,
    allow_delegation = False # we dont want the researcher agent to delegate any of the work
)

## Now we need to create an Agent which acts as a writer
writer = Agent(
    role = "Writer",
    goal = "Write compelling and engaging blog posts about ai trends and insights",
    backstory = "You are an AI blog post writer who specializes in writing about AI topics",
    verbose = True,
    allow_delegation = False
)

## Now we define a task for our agents
task1 = Task(description="Investigate the latest AI trends", agent=researcher, expected_output="A detailed description of AI trends",output_file="research.md")
task2 = Task(description="Write a compelling blog post based on the latest AI trends", agent=writer, expected_output="A blog post style report of AI trends", output_file="blog_post.md")

## We can now create a crew
crew = Crew(
    agents = [researcher,writer],
    tasks=[task1,task2],
    verbose=True,
    process=Process.sequential ## Currently crew only supports sequential, they may add other later.
)

result = crew.kickoff()
