from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.3-70B-Instruct'
    )

model = ChatHuggingFace(llm = llm)

#############################################################################################
#### Chatbot with sender identification and retaining chat

# chat_history = [
#   SystemMessage(content = 'You are a helpful AI Assistant'),
# ]
# while True:
#   user_input = input('You:' )
#   chat_history.append(HumanMessage(content = user_input))
#   if user_input == 'exit':
#     break
#   else:
#     result = model.invoke(chat_history)
#     chat_history.append(AIMessage(content = result.content))
#     print(f'AI: {result.content}')

# print(chat_history)

#############################################################################################

# Custom CSS for WhatsApp-style chat bubbles
st.markdown("""
    <style>
    .user-bubble {
        background-color: #003366;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .ai-bubble {
        background-color: #556B2F;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 70%;
        float: left;
        clear: both;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful AI Assistant.")
    ]

st.title("💬 LangChain Chatbot")

# Text input for user prompt
prompt = st.text_input("Write your message...")

if prompt:
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    ai_response = model.invoke(st.session_state.chat_history)
    st.session_state.chat_history.append(AIMessage(content=ai_response.content))

# Display chat history with bubbles
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div class='user-bubble'>{msg.content}</div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f"<div class='ai-bubble'>{msg.content}</div>", unsafe_allow_html=True)