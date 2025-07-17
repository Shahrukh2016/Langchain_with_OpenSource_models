from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

model = ChatHuggingFace(llm = llm)

# Creating Schema
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# Initiallizing parser
parser = StructuredOutputParser.from_response_schemas(schema)

# Forming prompt template
template = PromptTemplate(
    template= 'Give 3 facts about {topic} \n {format_instructions}',
    input_variables= ['topic'],
    partial_variables= {'format_instructions' : parser.get_format_instructions()}
)

#### ----   Without chain   ----- ####
# # Filling the topic in prompt template
# prompt = template.invoke({'topic' : 'black hole'})

# # Sending to LLM and getting back response
# result = model.invoke(prompt)

# # parsing the complex result
# final_result = parser.parse(result.content)

# # Printing result
# print(final_result)


#### ----   Without chain   ----- ####
chain = template | model | parser

result = chain.invoke({'topic' : 'black hole'})

print(result)

