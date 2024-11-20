import os
import streamlit as st
from dotenv import load_dotenv
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.pgvector import PGVector
from tools.etl import ETL
import psycopg2

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Sidebar setup
with st.sidebar:
    st.title("📄 Document and Chat Settings")
    docs = st.file_uploader("Upload file(s):", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Process Files"):
        if not openai_api_key:
            st.warning("Please provide your OpenAI API Key.")
        elif docs:
            with st.spinner("Processing documents..."):
                save_path = "to_process/"
                os.makedirs(save_path, exist_ok=True)
                for doc in docs:
                    file_name = doc.name
                    file_path = os.path.join(save_path, file_name)
                    with open(file_path, "wb") as f:
                        f.write(doc.getbuffer())
                # Run ETL pipeline
                etl = ETL(data_dir=save_path)
                res = etl.run()
                st.success(f"Processed and embedded {res} documents.")
                # Mark embeddings as available
                st.session_state["documents_processed"] = True
                # Clean up temporary files
                for doc in docs:
                    os.remove(os.path.join(save_path, doc.name))
        else:
            st.warning("Please upload at least one document.")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What would you like to know about your documents?"}]
if "conversation" not in st.session_state:
    st.session_state["conversation"] = None
if "documents_processed" not in st.session_state:
    st.session_state["documents_processed"] = False

# Initialize OpenAI embeddings and PGVector retriever
def get_retriever():
    try:
        pg_vector = PGVector(
            connection_string=os.getenv("CONNECTION_STRING"),
            collection_name=os.getenv("COLLECTION_NAME"),
            embedding_function=OpenAIEmbeddings(),
            use_jsonb=True
        )
        return pg_vector.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    except psycopg2.OperationalError as e:
        st.error(f"Database connection error: {e}")
        return None

# Initialize conversation chain
def get_conversation_chain():
    if not openai_api_key:
        return None
    llm = ChatOpenAI(api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = get_retriever()
    if not retriever:
        return None
    return ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=retriever)

# Chat UI
st.title("💬 Chat with your documents")

if st.session_state["documents_processed"]:
    # Initialize conversation chain if not already initialized
    if st.session_state["conversation"] is None:
        st.session_state["conversation"] = get_conversation_chain()

    # Display chat history
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
        elif not st.session_state["conversation"]:
            st.info("Conversation chain is not initialized. Please check your setup.")
        else:
            # Add user message
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Generate response
            conversation = st.session_state["conversation"]
            response = conversation({"question": prompt})
            answer = response["answer"]

            # Add assistant message
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
else:
    st.warning("Please upload and process documents before using the chatbot.")