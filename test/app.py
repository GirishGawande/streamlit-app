import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import streamlit as st
import os
os.environ["STREAMLIT_WATCHED_MODULES"] = "app"
from langchain.chains import ConversationChain
  # Disables problematic inspection

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Bharatiya Nagarik Suraksha Sanhita Chatbot", page_icon="📚")

# App title
st.title("Bharatiya Nagarik Suraksha Sanhita Chatbot")
st.write("Ask questions about the Bharatiya Nagarik Suraksha Sanhita, 2023")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to process PDF and create vector store
@st.cache_resource
def process_pdf():
    # Path to the PDF file
    pdf_path = "Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf"
    
    # Read PDF
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

# Function to get Groq LLM
def get_llm():
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="qwen/qwen1.5-32b-chat",  # Using Qwen model
        temperature=0.2
    )
    return llm

# Function to create retrieval chain
def get_retrieval_chain(vector_store):
    # Create LLM
    llm = get_llm()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert on the Bharatiya Nagarik Suraksha Sanhita, 2023. 
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    If the answer is not in the context, say "I don't have information about that in the Bharatiya Nagarik Suraksha Sanhita."
    """)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# Process PDF and create vector store
vector_store = process_pdf()

# Create retrieval chain
retrieval_chain = get_retrieval_chain(vector_store)

# Get user input
user_query = st.chat_input("Ask a question about Bharatiya Nagarik Suraksha Sanhita")

# Process user input
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Get response from retrieval chain
        response = retrieval_chain.invoke({"question": user_query})
        answer = response["answer"]
        
        # Display response
        message_placeholder.markdown(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})