from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_perplexity import ChatPerplexity
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

## Selecting LLM
model = ChatPerplexity(model="sonar")  # or another supported model


result = model.invoke("Hi, who are you ? Answer on one line")
print(result.content)
