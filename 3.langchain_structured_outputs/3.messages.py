## Messages.py
#### Storing messages along with retaining chat
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.3-70B-Instruct'
    )

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about Langchain?')
]

result = model.invoke(messages)
messages.append(AIMessage(content = result.content))

print(messages)