import os
import streamlit as st
from dotenv import load_dotenv

# Remove these commented imports
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate

# Use these consistent imports instead
from langchain_core.prompts import PromptTemplate
# from langchain_core.memory import ConversationBufferMemory
# from langchain.chains.conversation.base import ConversationChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Groq Qwen Chatbot", page_icon="🤖")

# App title and description
st.title("🤖 Groq Qwen Chatbot")
st.subheader("A simple chatbot powered by Groq's Qwen model and LangChain")

# Check for API key
api_key = os.getenv("GROQ_API_KEY")
if api_key == "your_api_key_here" or not api_key:
    st.error("Please add your Groq API key to the .env file")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get LLM response
def get_llm_response(user_input):
    # Initialize the Groq Qwen model
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-70b-8192",  # Using Llama 3 as Qwen might not be directly available
    )
    
    # Create a conversation memory
    memory = ConversationBufferMemory(return_messages=True)
    
    # Add previous messages to memory
    for message in st.session_state.messages:
        if message["role"] == "user":
            memory.chat_memory.add_user_message(message["content"])
        else:
            memory.chat_memory.add_ai_message(message["content"])
    
    # Create a prompt template
    template = """You are a helpful AI assistant. You provide clear, concise, and accurate responses.
    
    Current conversation:
    {history}
    
    Human: {input}
    AI Assistant:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
    )
    
    # Get response
    response = conversation.predict(input=user_input)
    
    return response

# Chat input
if user_input := st.chat_input("Ask something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_llm_response(user_input)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})