from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.3-70B-Instruct'
    )

model = ChatHuggingFace(llm = llm)

#############################################################################################

## Chatbot.py
#### Chatbot without retaining chat
# while True:
#   user_input = input('You:' )
#   if user_input == 'exit':
#     break
#   else:
#     result = model.invoke(user_input)
#     print(f'AI: {result.content}')


#############################################################################################

## Chatbot.py
#### Chatbot with retaining chat
# chat_history = []
# while True:
#   user_input = input('You:' )
#   chat_history.append(user_input)
#   if user_input == 'exit':
#     break
#   else:
#     result = model.invoke(chat_history)
#     chat_history.append(result.content)
#     print(f'AI: {result.content}')

#############################################################################################

## Chatbot.py
#### Chatbot with sender identification and retaining chat
chat_history = [
  SystemMessage(content = 'You are a helpful AI Assistant'),
]
while True:
  user_input = input('You:' )
  chat_history.append(HumanMessage(content = user_input))
  if user_input == 'exit':
    break
  else:
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print(f'AI: {result.content}')

print(chat_history)

#############################################################################################
