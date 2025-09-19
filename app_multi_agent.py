# app.py - REFACTORED FOR MULTI-AGENT SUPPORT, CHAT HISTORY, DELETION, CONTEXT DISPLAY & CLEAR CHAT

import streamlit as st
import os
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
import json 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- Configuration & Templates ---
load_dotenv()
DB_PATH = "multi_agent_chroma_db"
CHAT_HISTORY_FILE = "chat_history.json"
PROMPT_TEMPLATE = """
You are a helpful and precise expert on the document provided.

A user has asked the following question:
"{user_query}"

Here are the most relevant sections from the document.
---CONTEXT---
{context_string}
---END CONTEXT---

Your task is to formulate a clear and concise answer to the user's question, using ONLY the information found in the context blocks.

If the information in the context is not sufficient to answer the question, simply state: "I could not find a definitive answer in the provided document."
"""

# --- Helper functions for Chat History ---
def save_chat_history():
    """Saves the entire chat history from session state to a JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(st.session_state.messages, f)

def load_chat_history():
    """Loads the chat history from a JSON file, if it exists."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

# NEW: Function to clear the chat history for the active agent
def clear_active_chat_history():
    """Clears the chat history for the currently active agent."""
    active_agent = st.session_state.get('active_agent')
    if active_agent and active_agent in st.session_state.messages:
        st.session_state.messages[active_agent] = []
        save_chat_history()
        st.toast(f"Chat history for '{active_agent}' cleared.")

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    print("Initializing resources...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel('gemini-1.5-flash')
    except KeyError:
        st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        return None, None, None
    print("Resources initialized.")
    return embedding_model, client, llm

embedding_model, db_client, llm = load_resources()

# --- Agent Creation & Deletion Logic ---
def process_and_store_pdf(pdf_file, collection_name, client):
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    metadatas = [{"page_number": i+1} for i in range(len(doc))]
    if not any(page.get_text() for page in doc):
        st.error("Could not extract text from PDF.")
        doc.close()
        return False
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pages_content = [page.get_text() for page in doc]
    doc.close()
    chunks = text_splitter.create_documents(pages_content, metadatas=metadatas)
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_metadatas = [chunk.metadata for chunk in chunks]
    try:
        collection = client.create_collection(name=collection_name)
        collection.add(
            ids=[f"{collection_name}_{i}" for i in range(len(chunk_texts))],
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )
        return True
    except Exception as e:
        st.error(f"Error creating ChromaDB collection: {e}")
        return False

def delete_agent(agent_name_to_delete):
    """Deletes an agent's collection and chat history. Runs as a callback."""
    collection_name_to_delete = st.session_state.agent_list[agent_name_to_delete]
    
    try:
        db_client.delete_collection(name=collection_name_to_delete)
    except Exception as e:
        st.error(f"Error deleting collection: {e}")
        return

    del st.session_state.agent_list[agent_name_to_delete]
    if agent_name_to_delete in st.session_state.messages:
        del st.session_state.messages[agent_name_to_delete]
    
    save_chat_history()
    
    if st.session_state.active_agent == agent_name_to_delete:
        st.session_state.active_agent = None
    
    st.success(f"Agent '{agent_name_to_delete}' has been deleted.")

# --- UI & Application Logic ---
st.set_page_config(page_title="Multi-Agent Chat", layout="wide")
st.title("🤖 Multi-Agent Chat")
st.caption("Create agents, chat with them, and view the sources for each answer.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "agent_list" not in st.session_state:
    collections = db_client.list_collections()
    st.session_state.agent_list = {c.name: c.name for c in collections}

if "active_agent" not in st.session_state:
    st.session_state.active_agent = None

# --- Sidebar for Agent Management ---
with st.sidebar:
    st.header("Agents")
    for agent_name in list(st.session_state.agent_list.keys()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(agent_name, use_container_width=True, key=f"switch_{agent_name}"):
                st.session_state.active_agent = agent_name
                st.toast(f"Switched to agent: {agent_name}")
        with col2:
            st.button("🗑️", key=f"delete_{agent_name}", on_click=delete_agent, args=(agent_name,), use_container_width=True)

    st.divider()
    st.header("Create New Agent")
    uploaded_file = st.file_uploader("Upload a PDF to create an agent", type="pdf")
    if uploaded_file:
        new_agent_name = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")
        collection_name = "".join(e for e in new_agent_name if e.isalnum() or e in ('_', '-'))
        if st.button(f"Create '{new_agent_name}' Agent"):
            if collection_name in st.session_state.agent_list.values():
                st.warning(f"An agent named '{collection_name}' already exists.")
            else:
                with st.spinner(f"Creating agent..."):
                    success = process_and_store_pdf(uploaded_file, collection_name, db_client)
                    if success:
                        st.session_state.agent_list[new_agent_name] = collection_name
                        st.session_state.active_agent = new_agent_name
                        st.success("Agent created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to create agent.")

# --- Main Chat Interface ---
if not st.session_state.agent_list:
    st.info("👋 Welcome! Please create your first agent by uploading a PDF in the sidebar.")
else:
    if not st.session_state.active_agent and st.session_state.agent_list:
        st.session_state.active_agent = list(st.session_state.agent_list.keys())[0]
    
    active_agent_name = st.session_state.active_agent
    
    # MODIFIED: Added a button to clear the current chat history
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"Currently chatting with: **{active_agent_name}**")
    with col2:
        st.button("🧹 Clear Chat History", on_click=clear_active_chat_history, use_container_width=True)

    if active_agent_name not in st.session_state.messages:
        st.session_state.messages[active_agent_name] = []

    for message in st.session_state.messages[active_agent_name]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "context" in message:
                with st.expander("View Sources"):
                    for i, doc in enumerate(message["context"]):
                        st.info(f"Source {i+1}:\n\n{doc}")
                        st.divider()

    if prompt := st.chat_input(f"Ask {active_agent_name} a question..."):
        if not llm:
            st.error("LLM not initialized.")
        else:
            st.session_state.messages[active_agent_name].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                collection_name = st.session_state.agent_list[active_agent_name]
                try:
                    collection = db_client.get_collection(name=collection_name)
                    results = collection.query(query_texts=[prompt], n_results=5)
                    retrieved_docs = results['documents'][0]
                    context_string = "\n\n---\n\n".join(retrieved_docs)
                    final_prompt = PROMPT_TEMPLATE.format(user_query=prompt, context_string=context_string)
                    response = llm.generate_content(final_prompt)
                    final_response = response.text

                    with st.chat_message("assistant"):
                        st.markdown(final_response)
                        with st.expander("View Sources"):
                            for i, doc in enumerate(retrieved_docs):
                                st.info(f"Source {i+1}:\n\n{doc}")
                                st.divider()
                    
                    st.session_state.messages[active_agent_name].append({
                        "role": "assistant", 
                        "content": final_response,
                        "context": retrieved_docs
                    })
                    
                    save_chat_history()

                except Exception as e:
                    st.error(f"An error occurred: {e}")