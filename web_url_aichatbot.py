import os
import streamlit as st
from dotenv import load_dotenv
from data_ingestion import get_text_chunks
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables
load_dotenv()

# Constants
INDEX_DIR = "faiss_index"

# Streamlit page configuration
st.set_page_config(
    page_title="Web RAG AI with Agent",
    page_icon=":robot_face:",
    layout="centered"
)
st.title("Web RAG AI ChatBot 🤖")

# Custom system prompt for LLM
SYSTEM_PROMPT = """
You are a friendly, conversational AI assistant 😊 specialized in answering questions about genetic ai, based on the content of the provided web url.

– 👋 Whenever the user greets you (e.g., “hi,” “hello,” “how are you?”), respond with a warm and natural greeting.  
– 🙂 When the user introduces themselves, welcomes you, or asks about you, reply in a personable, upbeat tone.  
– 🎉 If they offer well wishes (“happy holidays,” “good luck,” etc.) or say goodbye, reply with an appropriate friendly farewell.  
– 📖 For any question directly related to web url or the content in the url, retrieve relevant context and give a clear, concise, accurate answer.  
– ❗ If the user’s question falls outside web url topics, politely say: “I’m sorry, I can only help with questions about based on web url.”  
– 🤔 If you don’t know the answer from the web url, say: “I’m not sure about that—let me know if you’d like to ask another question based on web url.”

Always keep your tone warm, helpful, and professional 🌟.
"""

# Sidebar: URL Input and Processing
st.sidebar.header("Load & Index Web Content")
sidebar_status = st.sidebar.empty()
url_input = st.sidebar.text_area(
    "Enter one or more URLs, one per line",
    help="Full web URLs to load and index"
)
process_trigger = st.sidebar.button("Build Retrieval Agent")

# Process URLs and build FAISS-based retrieval agent
if url_input and process_trigger:
    sidebar_status.info("Loading pages and indexing...")
    url_list = [u.strip() for u in url_input.splitlines() if u.strip()]
    with st.spinner("Building FAISS index and agent..."):
        # Load documents
        loader = WebBaseLoader(web_paths=url_list, bs_kwargs={"parse_only": None})
        docs = loader.load()
        raw_texts = [doc.page_content for doc in docs]
        # Join texts into one string for chunking
        full_text = "\n\n".join(raw_texts)
        # Chunk text
        chunks = get_text_chunks(full_text)
        # Create embeddings & FAISS
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        vectorstore.save_local(INDEX_DIR)
        # Create retriever tool
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        tool = create_retriever_tool(
            retriever,
            name="web_retrieval",
            description="Retrieves relevant excerpts from provided web pages."
        )
        # Build agent with custom system prompt
        llm = ChatOpenAI(temperature=0)
        # Wrap SYSTEM_PROMPT into a ChatPromptTemplate, including the agent scratchpad
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}"),
        ])
        agent = create_openai_tools_agent(
            llm=llm,
            tools=[tool],
            prompt=chat_prompt
        )
        st.session_state.agent_executor = AgentExecutor(agent=agent, tools=[tool])
    sidebar_status.success("Agent ready! Enter your queries below.")

# Initialize session state
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to the Web RAG AI Agent! How can I help?"}
    ]

chat_placeholder = st.empty()

# Chat Interface with custom alignment
def display_chat():
    with chat_placeholder.container():
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                _, col = st.columns([1, 5])
                with col:
                    st.chat_message("user").write(message["content"])
            else:
                col, _ = st.columns([5, 1])
                with col:
                    st.chat_message("assistant").write(message["content"])

display_chat()

# User input
user_input = st.chat_input("Ask me anything about the loaded web content...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    display_chat()
    if st.session_state.agent_executor:
        with st.spinner("Thinking..."):
            result = st.session_state.agent_executor.invoke({"input": user_input})
            response = result.get("output")
    else:
        response = "Please build the agent first by entering URLs and clicking 'Build Retrieval Agent'."
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    display_chat()
          













