from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_perplexity import ChatPerplexity
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from datetime import datetime
import os

st.title("💬 Chat Me Now! - AI Assistant")

# ------------------- INITIAL SESSION STATE -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful AI Assistant.")]
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# ------------------- API & MODEL -------------------
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
load_dotenv()

model = ChatPerplexity(model= 'sonar', api_key=PERPLEXITY_API_KEY)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
    .user-bubble {
        background-color: #003366;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .ai-bubble {
        background-color: #556B2F;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 70%;
        float: left;
        clear: both;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 10px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- PROCESS MESSAGE -------------------
def process_message():
    user_prompt = st.session_state.chat_input.strip()
    if user_prompt:
        # Add user message
        st.session_state.chat_history.append(HumanMessage(content=user_prompt))

        # Get AI response
        ai_response = model.invoke(st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=ai_response.content))

        # Save log locally
        log_file = "chat_logs.csv"
        log_df = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_prompt,
            "ai": ai_response.content
        }])
        if os.path.exists(log_file):
            existing_df = pd.read_csv(log_file)
            updated_df = pd.concat([existing_df, log_df], ignore_index=True)
            updated_df.to_csv(log_file, index=False)
        else:
            log_df.to_csv(log_file, index=False)

    # Clear input (safe, before next render)
    st.session_state.chat_input = ""

# ------------------- DISPLAY CHAT HISTORY -------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"<div class='user-bubble'>{msg.content}</div>", unsafe_allow_html=True)
    elif isinstance(msg, AIMessage):
        st.markdown(f"<div class='ai-bubble'>{msg.content}</div>", unsafe_allow_html=True)

# ------------------- FIXED INPUT BOX -------------------
st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
st.text_input(
    "Type your message...",
    key="chat_input",
    value=st.session_state.chat_input,
    label_visibility="collapsed",
    on_change=process_message  # <-- callback avoids post-widget modification
)
st.markdown("</div>", unsafe_allow_html=True)
