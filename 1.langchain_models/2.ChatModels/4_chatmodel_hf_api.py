from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)

result = llm.invoke("Name random 5 indian female names?")

print(result)