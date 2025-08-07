from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_perplexity import ChatPerplexity
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

## Defining function to apply in runnable lamba in order to perform some data processing steps
def word_count(text):
    return len(text.split())

## Selecting LLM 1
llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

## Initializing model 1
model1 = ChatHuggingFace(llm = llm)

## Selecting LLM 2
model2 = ChatPerplexity(model= 'sonar')

## Buildig Prompt 1 to generate the report
prompt1 = PromptTemplate(
    template= 'Write a detailed report on the {topic}',
    input_variables= ['topic']
)

## Buildig Prompt 2 to summarize the report if the words exceeds 500
prompt2 = PromptTemplate(
    template= 'Summarize the following text: \n {text}',
    input_variables= ['text']
)

## Creating output parser
parser = StrOutputParser()


## Forming sequential chain to generate joke
report_gen_chain = RunnableSequence(prompt1, model1, parser)

## Creating a branch chain for the conditions
branch_chain = RunnableBranch(
 (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model2, parser)),
 RunnablePassthrough()
)

## Merging the above 2 chains to get the final chain
final_chain = RunnableSequence(report_gen_chain, branch_chain)

## Sending to LLM and printing the response back
result = final_chain.invoke({'topic' : 'black hole'})
print(result)