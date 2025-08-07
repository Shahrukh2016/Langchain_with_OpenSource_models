from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
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
loader = DirectoryLoader(
    path= '7.langchain_document_loaders',
    glob= '*.pdf',
    loader_cls= PyMuPDFLoader
)
docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

# print(docs[0].page_content, end= '\n')
# print(docs[0].metadata)