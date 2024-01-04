import streamlit as st
import time
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Set up OpenAI API key
openai_key = st.secrets["openai"]["key"]

# Set up embedding and vector store
embedding = OpenAIEmbeddings(api_key=openai_key)
persist_directory = 'docs/chroma/fulltimeclass/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Set up language model
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_key)

st.markdown('<h1 style="font-family:Lora;color:darkred;text-align:center;">💬 TeeZee Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<i><h3 style="font-family:Arial;color:darkred;text-align:center;font-size:20px;padding-left:50px">Your AI Assistant To Answer Queries!</h3><i>', unsafe_allow_html=True)

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

        # Use vectordb directly for retrieval
        result = vectordb.retrieve(prompt)

        # Simulate stream of response with milliseconds delay
        full_response += result
        message_placeholder.markdown(full_response + "▌")
        time.sleep(0.05)
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
