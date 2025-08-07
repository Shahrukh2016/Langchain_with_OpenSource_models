from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_perplexity import ChatPerplexity
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

## Selecting LLM 1
llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

## Initializing model 1
model1 = ChatHuggingFace(llm = llm)

## Selecting LLM 2
model2 = ChatPerplexity(model= 'sonar')

## Buildig Prompt 1
prompt1 = PromptTemplate(
    template= 'Generate a tweet about the {topic}',
    input_variables= ['topic']
)

## Buildig Prompt 2
prompt2 = PromptTemplate(
    template= 'Generate a linkedin post about the {topic}',
    input_variables= ['topic']
)

## Creating output parser
parser = StrOutputParser()

## Defining chain using Runnable
parallel_chain = RunnableParallel(
    {
        'tweet' : RunnableSequence(prompt1, model1, parser),
        'linkedin' : RunnableSequence(prompt2, model2, parser)
    }
)

## Sending to LLM
result = parallel_chain.invoke({'topic' : 'AI'})

## Pringting respons
for platform in result.keys():
    print(f'{platform}: {result[platform]}', end = '\n')