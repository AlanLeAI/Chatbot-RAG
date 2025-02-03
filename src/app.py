import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from embedding import GPT2Embeddings
import chromadb.api
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

def get_vectorstore_from_url(url):
    model_name = "openai-community/gpt2"
    loader = WebBaseLoader(url)
    document = loader.load()

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap = 50)
    document_chunks = text_splitter.split_documents(document)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create a vectorstore from chunks
    vector_store = Chroma.from_documents(document_chunks, embeddings)

    return vector_store

def get_retriever_chain(vector_store):
    # chromadb.api.client.SharedSystemClient.clear_system_cache()
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    model.config.pad_token_id = tokenizer.eos_token_id 
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    llm = HuggingFacePipeline(pipeline=pipe)

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user", "{input}"),
        ("user", "Given the previous conversation, generate an answer relevant to the conversation")
    ])

    chain = create_history_aware_retriever(llm, retriever, prompt)
    return chain

def get_RAG_chain(retriever_chain):
    # chromadb.api.client.SharedSystemClient.clear_system_cache()
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    llm = HuggingFacePipeline(pipeline=pipe)


    prompt = ChatPromptTemplate.from_messages([
      ("system", "Give a short answer in two sentences to the user's questions based on the below context:\n\n{context}."),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever = get_retriever_chain(st.session_state.vector_store)
    rag_chain = get_RAG_chain(retriever) 
    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    print(response.keys())
    return response['answer']


# App Configuration
st.set_page_config(page_title="AlanAI")
st.title("Hi! I am an AI assistant.")


# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")


# User Input
user_query = st.chat_input("Type your message here...")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "How can I help you?")
    ]

if website_url is not None and website_url != "":
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    

if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content = user_query))
    st.session_state.chat_history.append(AIMessage(content = response))


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
