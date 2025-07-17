from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

## Initializing model
model = ChatHuggingFace(llm = llm)

## Building prompt
prompt = PromptTemplate(
    template= 'Generate 5 interesting facts about {topic}',
    input_variables= ['topic']
)

## Creating parser to extract string information of the returned query
parser = StrOutputParser()

## Constructing chain
chain = prompt | model | parser

## Sending to LLM
result = chain.invoke({'topic' : 'Google'})

## Pringting result
print(result)

## Visualizing the chain
chain.get_graph().print_ascii()