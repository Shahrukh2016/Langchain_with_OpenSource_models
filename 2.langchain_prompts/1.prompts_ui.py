from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="conversational"
)


model = ChatHuggingFace(llm=llm)

result = llm.invoke("Name random 5 indian female names?")

print(result)