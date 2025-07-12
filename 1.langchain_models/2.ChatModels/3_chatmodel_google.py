from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(  model="models/gemini-1.5-pro-latest", temperature=0.7)

result = llm.invoke("What is famous in India?")

print(result)