from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4', temperature=0.5, max_completion_tokens=10)

result = llm.invoke("What is famous in India?")

print(result)