import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


##Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

##Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    #split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap = 1000)
    docs = text_splitter.split_documents(documents)
    return docs

##Vector Embedding and Vector store
def get_vector_store(docs):
    
