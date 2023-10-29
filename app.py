import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    StreamlitChatMessageHistory,
)

from callback import PrintRetrievalHandler, StreamHandler
from config import configure_retriever

load_dotenv()

st.set_page_config(page_title="Chatea con tus documentos", page_icon=":books:")
st.title("Chatea con tus documentos")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

selected_model = st.sidebar.selectbox(
    "Escoge el modelo",
    [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
    ],
    key="selected_model",
)

temperature = st.sidebar.slider(
    "Temperatura", min_value=0.1, max_value=2.0, value=0.1, step=0.01
)

if not openai_api_key:
    st.info(
        "Por favor, introduce tu API Key de OpenAI. Puedes crea una [aquí]("
        "https://beta.openai.com/account/api-keys)."
    )
    st.stop()

uploaded_files = st.sidebar.file_uploader(
    label="Sube tus archivos", type=["pdf", "docx"], accept_multiple_files=True
)


if not uploaded_files:
    st.sidebar.info("Al cargar los archivos iniciará el proceso")
    st.info("Por favor sube un documento para continuar.")
    st.stop()

retriever = configure_retriever(files=uploaded_files, api_key=openai_api_key)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True
)

llm = ChatOpenAI(
    model_name=selected_model,
    openai_api_key=openai_api_key,
    temperature=temperature,
    streaming=True,
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory
)

if len(msgs.messages) == 0 or st.sidebar.button("Limpiar historial"):
    msgs.clear()
    msgs.add_ai_message("En qué puedo ayudarte?")

avatars = {"human": "user", "ai": "assistant"}

for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Escribe tu pregunta aquí"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(
            user_query, callbacks=[retrieval_handler, stream_handler]
        )
