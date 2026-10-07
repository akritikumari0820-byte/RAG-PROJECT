import os
import glob
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables
load_dotenv()

# Set up Streamlit Page Layout
st.set_page_config(page_title="Zyro Dynamics HR Helpdesk", page_icon="🤖", layout="centered")
st.title("🤖 Zyro Dynamics HR Helpdesk")
st.subheader("Intelligent RAG Chatbot")
st.write("Ask any questions regarding company HR policies, leave, reimbursements, or code of conduct.")

# 2. Cached Backend Initialization Lifecycle
@st.cache_resource
def initialize_rag_backend():
    """Initializes and caches the heavy embedding models, vector store, and LLM chains."""
    # Define Configurations matching your notebook
    LLM_Provider = 'groq'
    LLM_Model = 'openai/gpt-oss-20b'  # Injected from your notebook variable configuration
    CORPUS_PATH = "./zyro-dynamics-hr-corpus/"
    
    # Check for API Key
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY not found in environment variables! Please check your .env file.")
        st.stop()

    # Document Loading
    pdf_files = glob.glob(f"{CORPUS_PATH}/**/*.pdf", recursive=True)
    documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
        
    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    
    # Embeddings & Vector DB Generation
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # LLM Initialization
    llm = ChatGroq(
        model=LLM_Model,
        temperature=0.7,
        max_tokens=500,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Prompts & Chains Setup
    RAG_PROMPT = ChatPromptTemplate.from_template("""
    You are an HR assistance. Answers the question using ONLY the context below. If the answers isn't in the context, say you don't have that information.
    
    context : {context},
    Question : {question}
    """)
    
    GUARDIAL_PROMPT = ChatPromptTemplate.from_template("""
    You are a scope classifier for an HR assistant. Decide whether the question below is something an HR assistant should answer (leave, reimbursement, code of conduct, or other internal HR topics) or something out of scope (general knowledge, coding help, etc.). Response with exactly one word: IN_SCOPE or OUT_SCOPE.
    Question: {question}
    """)
    
    # Build LangChain Expressions
    guard_chain = GUARDIAL_PROMPT | llm | StrOutputParser()
    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    
    return retriever, guard_chain, rag_chain

# Trigger the background compilation of your backend pipeline
retriever, guard_chain, rag_chain = initialize_rag_backend()

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 3. Streamlit Frontend Interaction Logic

# Initialize session state for message history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation elements
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user queries via standard input box
if user_question := st.chat_input("Type your HR query here..."):
    
    # Render user query right away
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Generate bot response block
    with st.chat_message("assistant"):
        with st.spinner("Analyzing scope and internal policy databases..."):
            
            # Step 1: Run Guardrail Scope Check
            scope_rating = guard_chain.invoke({"question": user_question}).strip()
            
            if "OUT_SCOPE" in scope_rating:
                final_answer = ("I'm an HR assistance and can only help with questions about company HR "
                                "policies (leave, reimbursement, code of conduct, etc.). I don't have "
                                "information to answer that question.")
            else:
                # Step 2: Retrieve Documents and Execute QA Engine if it is In Scope
                docs = retriever.invoke(user_question)
                context_str = format_docs(docs)
                final_answer = rag_chain.invoke({"context": context_str, "question": user_question})
            
            st.markdown(final_answer)
            
    # Commit interaction records to conversation state history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
