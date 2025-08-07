from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_perplexity import ChatPerplexity
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
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
    template= 'Create a joke about the {topic}',
    input_variables= ['topic']
)

## Buildig Prompt 2
prompt2 = PromptTemplate(
    template= 'Explain the following joke: {topic}',
    input_variables= ['topic']
)

## Creating output parser
parser = StrOutputParser()

## Defining chains using Runnable
##### Chain 1 that prints joke
joke_gen_chain = RunnableSequence(prompt1 ,model1, parser)

##### Chain 2 that prints thr joke and give the explanation as well
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2, model2, parser)
})

##### Chain 3 that connects both the chains
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)


## Sending to LLM
result = parallel_chain.invoke({'topic' : 'cricket'})

## Pringting respons
for platform in result.keys():
    print(f'{platform}: {result[platform]}', end = '\n')