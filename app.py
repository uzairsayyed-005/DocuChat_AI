import os
import streamlit as st
from dotenv import load_dotenv

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="Document Interaction Assistant",
    layout="wide",
    page_icon="📚"
)

# Now load other dependencies after page config
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.markdown("""
<style>
    .main { background-color: #f5f7fb; padding: 2rem; }
    .header { color: #2d3436; font-family: 'Helvetica Neue', sans-serif; margin-bottom: 1.5rem; }
    .chat-container { background-color: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 1.5rem; margin-top: 1rem; }
    .user-message { background-color: #4a90e2; color: white; border-radius: 15px 15px 0 15px; padding: 1rem; margin: 0.5rem 0; max-width: 80%; float: right; }
    .assistant-message { background-color: #f1f3f6; color: #2d3436; border-radius: 15px 15px 15px 0; padding: 1rem; margin: 0.5rem 0; max-width: 80%; }
    .stTextInput>div>div>input { border-radius: 25px; padding: 12px 20px; border: 2px solid #e0e0e0; }
    .stFileUploader { border: 2px dashed #4a90e2; border-radius: 15px; padding: 1rem; }
    .stButton>button { background-color: #4a90e2; color: white; border-radius: 25px; padding: 10px 25px; border: none; transition: all 0.3s; }
    .stButton>button:hover { background-color: #357abd; transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# Rest of the application code remains the same...
document_embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

with st.container():
    st.markdown("<h1 class='header'>Local RAG based Document Interaction Assistant</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: #666; font-size: 1.1rem; margin-bottom: 2rem;'>
        Transform static documents into conversational partners with our AI-powered interface.
    </div>
    """, unsafe_allow_html=True)


# initialize the ollama model make sure to paste the model name as it is
conversation_engine = OllamaLLM(
    model="",
    temperature=0.3
)

with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        session_id = st.text_input("Session ID", value="session_default", 
                                 help="Enter a unique session identifier to maintain conversation history")

if 'system_memory' not in st.session_state:
    st.session_state.system_memory = {}

with st.container():
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(" ", type="pdf", 
                                   help="Upload a PDF document to start interacting with it")

if uploaded_file:
    temp_doc_path = "./temp_processing.pdf"
    with open(temp_doc_path, 'wb') as file_buffer:
        file_buffer.write(uploaded_file.getvalue())
        doc_title = uploaded_file.name

    pdf_loader = PyPDFLoader(temp_doc_path)
    raw_pages = pdf_loader.load()
    
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=4500,
        chunk_overlap=600,
        length_function=len
    )
    processed_chunks = text_processor.split_documents(raw_pages)

    vector_index = FAISS.from_documents(
        documents=processed_chunks,
        embedding=document_embedding_model
    )
    semantic_retriever = vector_index.as_retriever()

    contextualization_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Analyze the conversation history and current question strictly based on the document content. "
         "Generate a standalone query only if it can be answered from the document. "
         "If unsure, return the original question unchanged."),
        MessagesPlaceholder("chat_chronology"),
        ("human", "{input}")
    ])

    contextual_retriever = create_history_aware_retriever(
        conversation_engine,
        semantic_retriever,
        contextualization_prompt
    )

    response_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a document analysis assistant. Follow these rules strictly:\n"
         "1. Answer ONLY using the document context provided\n"
         "2. Never invent answers or use external knowledge\n"
         "3. If information is missing, say 'This information is not found in the document'\n"
         "4. Keep responses under 2 sentences\n\n"
         "Document Context:\n{context}"),
        MessagesPlaceholder("chat_chronology"),
        ("human", "{input}")
    ])

    response_assembler = create_stuff_documents_chain(
        conversation_engine,
        response_template
    )

    def validate_retrieval(inputs):
        if not inputs.get("context", []):
            return {"answer": "This information is not found in the document"}
        return RunnablePassthrough()

    knowledge_chain = create_retrieval_chain(
        contextual_retriever,
        validate_retrieval | response_assembler
    )

    def memory_store(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.system_memory:
            st.session_state.system_memory[session_id] = ChatMessageHistory()
        return st.session_state.system_memory[session_id]

    conversational_agent = RunnableWithMessageHistory(
        knowledge_chain,
        memory_store,
        input_messages_key="input",
        history_messages_key="chat_chronology",
        output_messages_key="answer"
    )

    if "message_log" not in st.session_state:
        st.session_state.message_log = []

    with st.container():
        st.markdown("### 💬 Chat Interface")
        chat_container = st.container()
        
        with chat_container:
            for entry in st.session_state.message_log:
                if entry["role"] == "user":
                    st.markdown(f"""
                    <div class='user-message'>
                        👤 {entry["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='assistant-message'>
                        🤖 {entry["content"]}
                    </div>
                    """, unsafe_allow_html=True)

        if user_query := st.chat_input("Ask a question about the document..."):
            st.session_state.message_log.append({"role": "user", "content": user_query})
            
            with chat_container:
                st.markdown(f"""
                <div class='user-message'>
                    👤 {user_query}
                </div>
                """, unsafe_allow_html=True)

            conversation_log = memory_store(session_id)

            try:
                system_response = conversational_agent.invoke(
                    {"input": user_query},
                    config={"configurable": {"session_id": session_id}}
                )
                answer = system_response['answer']
                
                if any(phrase in answer.lower() for phrase in ["not found", "don't know", "no information"]):
                    answer = "This information is not found in the document"
            except Exception as e:
                answer = "This information is not found in the document"

            with chat_container:
                st.markdown(f"""
                <div class='assistant-message">
                    🤖 {answer}
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.message_log.append(
                {"role": "assistant", "content": answer}
            )