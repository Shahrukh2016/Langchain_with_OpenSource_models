from langchain_community.document_loaders import WebBaseLoader
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
url = 'https://www.flipkart.com/apple-macbook-air-m4-16-gb-256-gb-ssd-macos-sequoia-mw0w3hn-a/p/itmf733f99c22ee6?pid=COMH9ZWQP4EP2XAT&lid=LSTCOMH9ZWQP4EP2XAT2OHHOE&marketplace=FLIPKART&q=macbook+air+m4&store=6bo%2Fb5g&srno=s_1_1&otracker=AS_QueryStore_OrganicAutoSuggest_1_6_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_6_na_na_na&fm=organic&iid=e764b811-039e-434c-82d8-2117e298565b.COMH9ZWQP4EP2XAT.SEARCH&ppt=hp&ppn=homepage&ssid=a34punw2kg0000001753379030631&qH=a3dc101ea3bce06d'
loader = WebBaseLoader(url)
docs = loader.load()

## Step 5: Building chain
chain = prompt | model | parser

# Step 6: Sending to LLM and getting output
result = chain.invoke({'question' : 'what is the price of product?', 'text' : docs[0].page_content})
print(result)