from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    # SystemMessage(content = 'You are a helpful {domain} assistant'),
    # HumanMessage(content = 'Explain in simple terms, what is {topic}')
    ('system', 'You are a helpful {domain} assistant'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain' : 'doctor', 'topic' : 'catarct'})
print(prompt)