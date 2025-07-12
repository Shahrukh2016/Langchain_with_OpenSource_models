from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

model = ChatHuggingFace(llm = llm)

# Creating parser object
parser = JsonOutputParser()

# Forming Template
template = PromptTemplate(
    template= 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

#### ----   Without chain   ----- ####
# # Building Prompt
# prompt = template.format()

# # Sending to LLM and get response
# result = model.invoke(prompt)

# # Setting up chain and parser to extract the json only
# final_result = parser.parse(result.content)
# print(final_result['name'])


#### ----   With chain   ----- ####
# Setting up chain
chain = template | model | parser

# Sending to LLM and get response
result = chain.invoke({})

# Printing result
print(result)
