from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

## Selecting LLM
llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

## Initializing model
model = ChatHuggingFace(llm = llm)

## Building prompt template 1
prompt1 = PromptTemplate(
    template= 'Write a joke about the {topic}',
    input_variables= ['topic']
)

## Building prompt template 2
prompt2 = PromptTemplate(
    template= 'Explain the following joke - {text}',
    input_variables= ['text']
)


## Creating output parser
parser = StrOutputParser()

## Defining chain using Runnable
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

## Sending request to llm
result = chain.invoke({'topic' : 'Content creaters'})

## Printing response
print(result)