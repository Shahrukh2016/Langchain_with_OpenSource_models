from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

## Initializing model 1
llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)
model = ChatHuggingFace(llm = llm)


## Initializing parser 1
parser1 = StrOutputParser()

## Restricting the llm output using output parser
class Feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description= 'Give the sentiment of the feedback')

## Initializing parser 2
parser2 = PydanticOutputParser(pydantic_object= Feedback)

## Building prompt 1 that classifies the sentiment of the feedback
prompt1 = PromptTemplate(
    template= 'Classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}',
    input_variables= ['feedback'],
    partial_variables= {'format_instruction' : parser2.get_format_instructions()}
) 

## Building classifier chain for system
classifier_chain = prompt1 | model | parser2

# ## Sending to LLM and print the results
# result = classifier_chain.invoke({'feedback' : 'This is a terrible smartphone'}).sentiment
# print(result)

## Building prompt 2 that generates response for the positive feedback
prompt2 = PromptTemplate(
    template= 'Write an appropriate response to this positive feedback \n {feedback}',
    input_variables= ['feedback']
) 

## Building prompt 3 that generates response for the negative feedback
prompt3 = PromptTemplate(
    template= 'Write an appropriate response to this negative feedback \n {feedback}',
    input_variables= ['feedback']
) 

## Forming a runnable branch chains just like IF ELSE
branch_chains = RunnableBranch(
    (lambda x: x.sentiment == 'positive' , prompt2 | model | parser1),
    (lambda x: x.sentiment == 'negative' , prompt3 | model | parser1),
    RunnableLambda(lambda x: 'could not find sentiment')
)

## Forming a final chain that can process the pipeline
chain = classifier_chain | branch_chains

## Sending to LLM and printing the response
result = chain.invoke({'feedback' : 'This product is horrible'})
print(result)

## Visualizing the chain
print(chain.get_graph().print_ascii())