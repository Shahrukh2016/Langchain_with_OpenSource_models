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

## Building prompts
prompt1 = PromptTemplate(
    template= 'Generate a detailed report on {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= 'Create a 5 pointer summary on the following  \n {text}',
    input_variables= ['text']
)


## Creating parser to extract string information of the returned query
parser = StrOutputParser()

## Constructing chain
chain = prompt1 | model | parser | prompt2 | model | parser

## Sending to LLM
result = chain.invoke({'topic' : 'Google'})

## Pringting result
print(result)

## Visualizing the chain
chain.get_graph().print_ascii()