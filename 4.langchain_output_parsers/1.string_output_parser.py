from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
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


prompt1 = template1.invoke({'topic' : 'black hole'})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result1.content})

result2 = model.invoke(prompt2)

print(result2.content)
