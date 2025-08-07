from langchain_community.document_loaders import TextLoader
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

## Step 2: Defining prompt
prompt = PromptTemplate(
    template= 'Write a summary of the following poem - \n {text}',
    input_variables= ['text']
)

## Step 3: Initializing parser
parser = StrOutputParser()


## Step 4: Building chain
chain = prompt | model | parser


## Step 5: Loading the docs/poems
loader = TextLoader('7.langchain_document_loaders/blackhole.txt', encoding= 'utf-8')
docs = loader.load()

## Sending to LLM
result = chain.invoke({'text' : docs[0].page_content})

## Printing the result
print(result)





