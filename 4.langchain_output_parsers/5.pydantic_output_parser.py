from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

## Initializing model
model = ChatHuggingFace(llm = llm)

## Creating pydantic class
class Person(BaseModel):
    name: str = Field(description= 'Name of the person')
    age: int = Field(gt=18, description= 'Age of the person')
    city: str = Field(description= 'Name of the city the person belongs to')

## Defining Parser
parser = PydanticOutputParser(pydantic_object= Person)

## Forming template
template = PromptTemplate(
    template= 'Generate the name, age, city of a fictional {place} person \n {format_instruction}',
    input_variables= ['place'],
    partial_variables= {'format_instruction' : parser.get_format_instructions()}
)

#### ----   Without chain   ----- ####
# ## Setting up the prompt
# prompt = template.invoke({'place' : 'indian'})

# ## Sending to LLM
# result = model.invoke(prompt)

# ## Restructuring response
# final_result = parser.parse(result.content)

# ## Printing output
# print(final_result)


#### ----   With chain   ----- ####
## Defining chain
chain = template | model | parser

## Sending it to LLM
result = chain.invoke({'place' : 'indian'})

## Printing result
print(result)