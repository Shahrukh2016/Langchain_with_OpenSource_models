from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

## Step 1: Selecting LLM 
llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

## Initializing model
model = ChatHuggingFace(llm = llm)

## Loading stories pdf
loader = PyPDFLoader('7.langchain_document_loaders/deeplearning.pdf')
docs = loader.load()
# print(docs)

print(docs[0].page_content, end= '\n')
print(docs[0].metadata)