from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

docs = [
    "LLMs are insane",
    "I am very passionate for Machine Learning",
    "RAGs are future"
]

result = embedding.embed_documents('Delhi is the capital of India')

print(str(result))