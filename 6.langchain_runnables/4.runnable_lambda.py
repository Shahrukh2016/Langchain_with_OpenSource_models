from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_perplexity import ChatPerplexity
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
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

## Buildig Prompt
prompt = PromptTemplate(
    template= 'Create a joke about the {topic}',
    input_variables= ['topic']
)

## Creating output parser
parser = StrOutputParser()


## Forming sequential chain to generate joke
joke_gen_chain = RunnableSequence(prompt, model1, parser)

## Forming parallel chain to make
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})

## OR we can use in this way 
# parallel_chain = RunnableParallel({
#     'joke' : RunnablePassthrough(),
#     'word_count' : RunnableLambda(lambda x: len(x.split()))
# })

## Resulting chain
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

## Sending to LLM and printing output
result = final_chain.invoke({'topic' : 'laptop'})
print(result)