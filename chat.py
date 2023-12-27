import streamlit as st
import random
import datetime
import time
import langchain
import tensorflow as tf
import pandas as pd
import numpy
import openai
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit.components.v1 as components
from config import OPENAI_API_KEY

persist_directory = 'docs/chroma/chatbot/'

embedding = OpenAIEmbeddings(api_key="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")

vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="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")


st.markdown('<h1 style="font-family:Lora;color:darkred;text-align:center;">💬 TeeZee Chatbot</h1>',unsafe_allow_html=True)
st.markdown('<i><h3 style="font-family:Arial;color:darkred;text-align:center;font-size:20px;padding-left:50px">Your AI Assistant To Answer Queries!</h3><i>',unsafe_allow_html=True)




if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("Hit me up with your queries!"):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.  
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

        # Run chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        result = qa_chain({"query": prompt})

        # Simulate stream of response with milliseconds delay
        full_response += result["result"]
        message_placeholder.markdown(full_response + "▌")
        time.sleep(0.05)
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history

    st.session_state.messages.append({"role": "assistant", "content": full_response})
