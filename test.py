from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.1-8B-Instruct'
)

model = ChatHuggingFace(llm = llm)


print(model.invoke('Hi'))

# # Creating parser object
# jsonparser = JsonOutputParser()
# strparser = StrOutputParser

# ## Forming template 1
# template1 = PromptTemplate(
#     template= 'Write a 10 pointers specifications about the given topic - \n {text}',
#     input_variables=['text']
# )

# # Forming Template 2
# template2 = PromptTemplate(
#     template= 'Give the one word keyword from each of the pointer of the given summary - \n {summary} \n {format_instruction}',
#     input_variables=['summary'],
#     partial_variables={'format_instruction' : jsonparser.get_format_instructions()}
# )

# #### ----   With chain   ----- ####
# # Setting up chain
# chain = template1 | model | strparser | template2 | model | jsonparser

# # Sending to LLM and get response
# result = chain.invoke({'text' : 'warm hole'})

# # Printing result
# print(result)
