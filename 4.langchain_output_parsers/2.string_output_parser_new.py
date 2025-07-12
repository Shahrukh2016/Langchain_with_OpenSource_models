from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'meta-llama/Llama-3.3-70B-Instruct'
)

model = ChatHuggingFace(llm = llm)

# Template 1 -> Detailed report
template1 = PromptTemplate(
    template= 'Write a detailed report on {topic}',
    input_variables= ['topic']
)

# Templtate 2 -> Summary
template2 = PromptTemplate(
    template= 'Write a 5 line summary on the following text. /n {text}',
    input_variables= ['text']
)

# Parser
parser = StrOutputParser()


# Forming Chain
chain = template1 | model | parser | template2 | model | parser

# Executing Chain
result = chain.invoke({'topic' : 'black hole'})

# Printing result
print(result)