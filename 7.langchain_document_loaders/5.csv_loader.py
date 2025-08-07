from langchain_community.document_loaders import CSVLoader
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
    template= 'Answe the following question \n {question} from the following text - \n {text}',
    input_variables= ['question', 'text']
)

## Step 3: Initializing parser
parser = StrOutputParser()

## Step 4: Setting up webbase loader
path = '7.langchain_document_loaders\Social_Network_Ads.csv'
loader = CSVLoader(path)
docs = loader.load()

## Step 5: Building chain
chain = prompt | model | parser

## Printing doc
print(docs[0])

# Step 6: Sending to LLM and getting output
# result = chain.invoke({'question' : 'what is the price of product?', 'text' : docs[0].page_content})
# print(result)