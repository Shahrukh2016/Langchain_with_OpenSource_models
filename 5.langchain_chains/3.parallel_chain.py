from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

## Initializing model 1
llm1 = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)
model1 = ChatHuggingFace(llm = llm1)


## Initializing model 2
llm2 = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct-v0.2'
)
model2 = ChatHuggingFace(llm = llm2)

## Forming Prompt 1
prompt1 = PromptTemplate(
    template= 'Generate short and simple notes of the following topic \n {topic}',
    input_variables=['topic']

)

## Forming Prompt 2
prompt2 = PromptTemplate(
    template= 'Generate 5 short questions answers from the following text \n {text}',
    input_variables=['text']
)


## Forming Prompt 3
prompt3 = PromptTemplate(
    template= 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
) 

## Initializing parser
parser = StrOutputParser()

## Describing runnable
parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

## Defining merged cain
merge_chain = prompt3 | model1 | parser

## Final chain that connects parallel chain to merge chain
chain = parallel_chain | merge_chain

## Sending to LLM
result = chain.invoke({'topic' : 'Attention is all you need', 'text' : prompt1})

## Printing result
print(result)

## Checking the graph
chain.get_graph().print_ascii()