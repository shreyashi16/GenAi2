import streamlit as st
#import os
#from dotenv import load_dotenv
#langChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Pages Config
st.set_page_config(page_title="C++ RAG ChatBot")
st.title("C++ RAG ChatBot")
st.write("Ask any question related to C++ Introducation")

#Load Enviornment Variables
#cache document Loading
@st.cache_resource
def load_vectorstore():
    #Load Document
    loader = TextLoader("C++_Introduction.txt",encoding ="utf-8")
    documents = loader.load()

    #split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 20
    )
    final_documents = text_splitter.split_documents(documents)

    #Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )

    #create FAISS Vector Store
    db = FAISS.from_documents(final_documents, embeddings)
    return db

#load vector DB(only once)
db = load_vectorstore()

#user input
query = st.text_input("Enter your qiestion about C++ : ")

if query:
    docs = db.similarity_search(query, k=3)
    st.subheader("Retrived Context :")
    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i+1}**")
        st.write(doc.page_content)